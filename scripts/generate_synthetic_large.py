#!/usr/bin/env python3
"""Streaming synthetic data generator for large CSV outputs.

Writes in chunks to avoid high memory usage and outputs a ZIP and SHA256.
"""
import argparse
import os
import json
import math
import random
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse helper logic from smoke generator but stream rows
FREQS_MHZ = list(range(4900, 6101, 5))
ENVIRONMENTS = ["Urban", "Rural"]
DENSITIES = ["Low", "Medium", "High"]
WEATHER_COND = ["Clear", "Cloudy", "Rain", "Fog", "Snow", "Storm"]
OBST_TYPES = ["None", "Tree", "Building", "Vehicle", "Crane", "Billboard"]
TX_ANT_GAIN_DB = 15.0
RX_ANT_GAIN_DB = 15.0
BANDWIDTHS_MHZ = [20, 40, 80, 160]


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def fspl_db(distance_m: float, freq_mhz: float) -> float:
    if distance_m < 1:
        distance_m = 1.0
    d_km = distance_m / 1000.0
    return 32.45 + 20 * math.log10(d_km) + 20 * math.log10(freq_mhz)


def snr_to_efficiency_bps_per_hz(snr_db: float) -> float:
    return 10.0 / (1.0 + math.exp(-(snr_db - 25.0) / 4.0))


def generate_row(i, forced=None):
    environment = random.choice(ENVIRONMENTS)
    density = random.choice(DENSITIES)
    lat_a = random.uniform(33.0, 36.0)
    lon_a = random.uniform(35.0, 37.0)
    el_a = random.uniform(5, 900)
    lat_b = lat_a + random.uniform(-0.15, 0.15)
    lon_b = lon_a + random.uniform(-0.15, 0.15)
    el_b = random.uniform(5, 900)
    distance_m = haversine_m(lat_a, lon_a, lat_b, lon_b)
    weather = random.choice(WEATHER_COND)
    humidity = random.uniform(20, 95)
    temp_c = random.uniform(-5, 42)

    fresnel_clear = random.uniform(10, 100)
    # allow forced overrides for stratified generation
    if forced is None:
        forced = {}
    environment = forced.get("environment_type", environment)
    density = forced.get("area_density", density)
    obstructed = random.random() < 0.35
    # if user forced image_obstruction_detected, respect it
    if "image_obstruction_detected" in forced:
        obstructed = bool(forced.get("image_obstruction_detected"))
    obst_type = random.choice(OBST_TYPES if obstructed else ["None"])
    noise_floor_dbm = random.uniform(-115, -82)
    noise_dbm = noise_floor_dbm + random.uniform(0, 8)
    tx_power_dbm = random.uniform(10, 30)
    channel_bw_mhz = random.choice(BANDWIDTHS_MHZ)
    num_avail = random.randint(10, 50)
    available_channels = sorted(random.sample(FREQS_MHZ, k=num_avail))

    spectral_scan = {ch: noise_floor_dbm + np.random.normal(loc=2.0, scale=2.0) for ch in available_channels}

    best = None
    for ch in available_channels:
        fspl = fspl_db(distance_m, ch)
        loss = 0.01 * (distance_m / 1000.0)
        rssi = (tx_power_dbm + TX_ANT_GAIN_DB + RX_ANT_GAIN_DB) - fspl - loss
        interference_dbm = spectral_scan[ch]
        snr = rssi - interference_dbm
        score = snr
        if (best is None) or (score > best[3]):
            best = (ch, rssi, snr, score)

    ch_best, rssi_best, snr_best, _ = best
    eff = snr_to_efficiency_bps_per_hz(max(-10.0, min(60.0, snr_best)))
    expected_throughput_mbps = (eff * channel_bw_mhz)

    return {
        "scenario_id": i,
        "device_a_coordinates": json.dumps([lat_a, lon_a, el_a]),
        "device_b_coordinates": json.dumps([lat_b, lon_b, el_b]),
        "link_distance_m": float(distance_m),
        "noise_dbm": float(noise_dbm),
        "noise_floor_dbm": float(noise_floor_dbm),
        "rssi_dbm": float(rssi_best),
        "snr_db": float(snr_best),
        "tx_power_dbm": float(tx_power_dbm),
        "channel_bandwidth_mhz": int(channel_bw_mhz),
        "channel_utilization_pct": float(random.uniform(0, 100)),
        "available_channels_mhz": json.dumps(available_channels),
        "spectral_scan_dbm": json.dumps(spectral_scan),
        "fresnel_clear_pct": float(fresnel_clear),
        "weather_temp_c": float(temp_c),
        "weather_humidity_pct": float(humidity),
        "weather_condition": weather,
        "image_obstruction_detected": bool(obstructed),
        "image_obstruction_type": obst_type,
        "environment_type": environment,
        "area_density": density,
        "recommended_channel_mhz": int(ch_best),
        "expected_throughput_mbps": float(expected_throughput_mbps),
    }


def sha256_of(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def apply_stratified_resample(rows, stratify_by, balance_mode="none", seed=None):
    """
    Stratify and resample a list of row dicts.

    - rows: iterable/list of dicts
    - stratify_by: single field name to group by
    - balance_mode: 'none' | 'oversample' | 'undersample'
    - seed: int or None for deterministic sampling/shuffle

    Returns a new list of rows (may increase/decrease length depending on balance_mode).
    This function is intended for shard-sized inputs and is not fully streaming.
    """
    if balance_mode not in ("none", "oversample", "undersample"):
        raise ValueError("balance_mode must be one of: none, oversample, undersample")

    rows = list(rows)
    if not stratify_by or balance_mode == "none":
        return rows

    rng = random.Random(seed)
    groups = {}
    for r in rows:
        key = r.get(stratify_by)
        groups.setdefault(key, []).append(r)

    sizes = {k: len(v) for k, v in groups.items()}
    if not sizes:
        return rows

    if balance_mode == "oversample":
        target = max(sizes.values())
    elif balance_mode == "undersample":
        target = min(sizes.values())
    else:
        target = None

    balanced = []
    for k, group_rows in groups.items():
        n = len(group_rows)
        if n == target:
            balanced.extend(group_rows)
        elif n < target:
            # oversample with replacement
            needed = target - n
            picks = [rng.choice(group_rows) for _ in range(needed)]
            balanced.extend(group_rows)
            balanced.extend(picks)
        else:
            # undersample without replacement
            sampled = rng.sample(group_rows, target)
            balanced.extend(sampled)

    rng.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by", type=str, default=None, help="Comma-separated fields to stratify on, e.g. environment_type,area_density")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"synthetic_wifi_5ghz_{args.n_samples:,}.csv"
    zip_path = outdir / csv_path.with_suffix(".zip").name

    # Stream write in chunks
    cols = list(generate_row(0).keys())
    written = 0
    strata = []
    if args.stratify_by:
        fields = [f.strip() for f in args.stratify_by.split(',')]
        # build simple cartesian product of likely values for the requested fields
        pool = {"environment_type": ENVIRONMENTS, "area_density": DENSITIES, "image_obstruction_detected": [True, False]}
        for f in fields:
            if f in pool:
                strata.append((f, pool[f]))
        # flatten to list of dicts
        if strata:
            combos = []
            def build(idx, cur):
                if idx >= len(strata):
                    combos.append(dict(cur))
                    return
                key, vals = strata[idx]
                for v in vals:
                    cur[key] = v
                    build(idx+1, cur)
                cur.pop(key, None)
            build(0, {})
        else:
            combos = []
    else:
        combos = []

    with open(csv_path, "w", encoding="utf-8") as fh:
        # header
        fh.write(",".join(cols) + "\n")
        while written < args.n_samples:
            chunk = min(args.chunk_size, args.n_samples - written)
            rows = []
            for i in range(chunk):
                idx = (written + i)
                if combos:
                    forced = combos[idx % len(combos)]
                    rows.append(generate_row(written + i, forced=forced))
                else:
                    rows.append(generate_row(written + i))
            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(fh, header=False, index=False)
            written += chunk

    # Create zip
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=csv_path.name)

    print(f"Saved {args.n_samples:,} rows -> {csv_path}")
    print("CSV SHA256:", sha256_of(csv_path))
    print("ZIP SHA256:", sha256_of(zip_path))


if __name__ == "__main__":
    main()
