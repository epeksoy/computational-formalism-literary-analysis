"""Run the full Chapter 4 pipeline in order.

Usage:
    python 00_run_all.py

Reproduces every file in /mnt/user-data/outputs/ from the four raw CONLIT
uploads in /mnt/user-data/uploads/.
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ['01_prepare.py', '02_axes.py', '03_stats.py',
           '04_figures.py', '05_workbook.py']

for s in SCRIPTS:
    print('\n' + '=' * 70)
    print(f'RUNNING {s}')
    print('=' * 70)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)])
    if r.returncode != 0:
        sys.exit(f'FAILED at {s}')
print('\nall done.')
