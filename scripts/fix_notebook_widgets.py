#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path


def remove_widgets(obj):
    if isinstance(obj, dict):
        if 'widgets' in obj:
            del obj['widgets']
        for v in list(obj.values()):
            remove_widgets(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_widgets(item)


def fix_notebook_file(nb_path: Path, quiet: bool = False):
    try:
        data = json.loads(nb_path.read_text(encoding='utf-8'))
    except Exception as e:
        if not quiet:
            print(f'Skipping {nb_path}: could not read/parse ({e})')
        return False
    remove_widgets(data)
    nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    if not quiet:
        print(f'Fixed notebook: {nb_path}')
    return True


def fix_notebook(path: str, quiet: bool = False):
    p = Path(path)
    if p.is_dir():
        changed = 0
        for nb in p.rglob('*.ipynb'):
            if fix_notebook_file(nb, quiet=quiet):
                changed += 1
        if not quiet:
            print(f'Processed {changed} notebooks under {p}')
        return
    if p.is_file():
        fix_notebook_file(p, quiet=quiet)
        return
    # if path doesn't exist, try to glob
    matched = list(Path('.').glob(path))
    for m in matched:
        fix_notebook(str(m), quiet=quiet)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Strip widget metadata from notebooks (recursively for directories)')
    parser.add_argument('path', help='File or directory (or glob) to process')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    fix_notebook(args.path, quiet=args.quiet)
