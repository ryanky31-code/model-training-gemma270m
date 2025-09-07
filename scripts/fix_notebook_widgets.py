#!/usr/bin/env python3
import json
import sys
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


def fix_notebook(path):
    nb_path = Path(path)
    data = json.loads(nb_path.read_text(encoding='utf-8'))
    remove_widgets(data)
    nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Fixed notebook: {nb_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/fix_notebook_widgets.py path/to/notebook.ipynb')
        sys.exit(2)
    fix_notebook(sys.argv[1])
