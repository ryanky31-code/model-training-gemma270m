import math
import json
import random
import numpy as np
import pandas as pd
import zipfile
import hashlib
import os

# Small smoke parameters
N_SAMPLES = 50
OUT_DIR = '.'
CSV_PATH = os.path.join(OUT_DIR, 'synthetic_wifi_5ghz_outdoor_smoke.csv')
ZIP_PATH = os.path.join(OUT_DIR, 'synthetic_wifi_5ghz_outdoor_smoke.zip')

FREQS_MHZ = list(range(4900, 6101, 5))
ENVIRONMENTS = ["Urban", "Rural"]
DENSITIES    = ["Low", "Medium", "High"]
WEATHER_COND = ["Clear", "Cloudy", "Rain", "Fog", "Snow", "Storm"]
OBST_TYPES   = ["None", "Tree", "Building", "Vehicle", "Crane", "Billboard"]
TX_ANT_GAIN_DB = 15.0
RX_ANT_GAIN_DB = 15.0
BANDWIDTHS_MHZ = [20, 40, 80, 160]

random.seed(42)
np.random.seed(42)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

def snr_to_efficiency_bps_per_hz(snr_db: float) -> float:
    return 10.0 / (1.0 + math.exp(-(snr_db - 25.0) / 4.0))

def fspl_db(distance_m: float, freq_mhz: float) -> float:
    if distance_m < 1:
        distance_m = 1.0
    d_km = distance_m / 1000.0
    return 32.45 + 20 * math.log10(d_km) + 20 * math.log10(freq_mhz)

def weather_extra_loss_db(weather: str, distance_m: float) -> float:
    d_km = distance_m / 1000.0
    base = {"Clear": 0.0, "Cloudy": 0.2, "Fog": 0.6, "Rain": 0.8, "Snow": 0.9, "Storm": 1.5}[weather]
    return base * d_km

def density_obstruction_factor(env: str, density: str) -> float:
    return {"Urban": {"Low": 0.5, "Medium": 1.5, "High": 3.0},
            "Rural": {"Low": 0.1, "Medium": 0.4, "High": 0.8}}[env][density]

def fresnel_penalty_db(f_clear: float) -> float:
    if f_clear >= 90: return 0.0
    if f_clear >= 70: return 1.0
    if f_clear >= 50: return 3.0
    if f_clear >= 30: return 6.0
    return 10.0

def obstruction_penalty_db(obstructed: bool, obst_type: str) -> float:
    if not obstructed or obst_type == "None": return 0.0
    return {"Tree": 3.0, "Vehicle": 2.0, "Billboard": 4.0, "Building": 8.0, "Crane": 5.0}.get(obst_type, 2.0)


def generate_synthetic_row(scenario_id: int):
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

    f_range = {"Urban": {"Low": (60, 100), "Medium": (40, 90), "High": (15, 80)},
               "Rural": {"Low": (85, 100), "Medium": (65, 100), "High": (45, 100)}}
    fresnel_clear = random.uniform(*f_range[environment][density])

    obstructed = random.random() < (0.65 if (environment == "Urban" and density == "High") else 0.35 if environment == "Urban" else 0.2 if density != "Low" else 0.1)
    obst_type = random.choice(OBST_TYPES if obstructed else ["None"])

    nf_range = {"Urban": {"Low": (-105, -92), "Medium": (-100, -88), "High": (-95, -82)},
                "Rural": {"Low": (-115, -102), "Medium": (-110, -98), "High": (-108, -96)}}
    noise_floor_dbm = random.uniform(*nf_range[environment][density])
    noise_dbm = noise_floor_dbm + random.uniform(0, 8)

    tx_power_dbm = random.uniform(10, 30)
    channel_bw_mhz = random.choice(BANDWIDTHS_MHZ)
    num_avail = random.randint(10, 50)
    available_channels = sorted(random.sample(FREQS_MHZ, k=num_avail))

    util_map = {("Urban", "Low"): (20, 60), ("Urban", "Medium"): (40, 80), ("Urban", "High"): (60, 98),
                ("Rural", "Low"): (0, 20), ("Rural", "Medium"): (10, 40), ("Rural", "High"): (20, 55)}
    util_pct = random.uniform(*util_map[(environment, density)])
    spectral_scan = {}
    for ch in available_channels:
        congestion_bump = np.random.normal(loc=util_pct / 100 * 8.0, scale=1.5)
        spectral_scan[ch] = noise_floor_dbm + 2.0 + max(0.0, congestion_bump)

    best = None
    for ch in available_channels:
        fspl = fspl_db(distance_m, ch)
        loss = (weather_extra_loss_db(weather, distance_m)
                + density_obstruction_factor(environment, density) * (distance_m / 1000.0)
                + fresnel_penalty_db(fresnel_clear)
                + obstruction_penalty_db(obstructed, obst_type))
        rssi = (tx_power_dbm + TX_ANT_GAIN_DB + RX_ANT_GAIN_DB) - fspl - loss
        interference_dbm = spectral_scan[ch]
        snr = rssi - interference_dbm
        score = snr - 0.25 * (interference_dbm - noise_floor_dbm)
        if (best is None) or (score > best[3]):
            best = (ch, rssi, snr, score)

    ch_best, rssi_best, snr_best, _ = best
    eff = snr_to_efficiency_bps_per_hz(max(-10.0, min(60.0, snr_best)))
    expected_throughput_mbps = (eff * channel_bw_mhz * 1e6) / 1e6

    return {
        "scenario_id": scenario_id,
        "device_a_coordinates": json.dumps([lat_a, lon_a, el_a]),
        "device_b_coordinates": json.dumps([lat_b, lon_b, el_b]),
        "link_distance_m": float(distance_m),
        "noise_dbm": float(noise_dbm),
        "noise_floor_dbm": float(noise_floor_dbm),
        "rssi_dbm": float(rssi_best),
        "snr_db": float(snr_best),
        "tx_power_dbm": float(tx_power_dbm),
        "channel_bandwidth_mhz": int(channel_bw_mhz),
        "channel_utilization_pct": float(util_pct),
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

# Generate and write
rows = [generate_synthetic_row(i) for i in range(N_SAMPLES)]
df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)
with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(CSV_PATH, arcname=os.path.basename(CSV_PATH))

# SHA256

def sha256_of(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

print(f"Saved {len(df):,} rows -> {CSV_PATH}")
print('CSV SHA256:', sha256_of(CSV_PATH))
print('ZIP SHA256:', sha256_of(ZIP_PATH))
print(df.head(2).to_dict(orient='records'))
