import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRANSFERMARKT_DIR = REPO_ROOT / 'data' / 'Transfermarkt'
LATEST_PLAYERS_DIR = TRANSFERMARKT_DIR / 'kaggle-latest'
LATEST_PLAYERS_CSV = LATEST_PLAYERS_DIR / 'players.csv'
KAGGLE_DATASET = 'davidcariboo/player-scores'


def parse_args():
    parser = argparse.ArgumentParser(description='Download the latest Kaggle players.csv snapshot.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force a fresh Kaggle download even if the local file exists.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    LATEST_PLAYERS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        '-m',
        'kaggle',
        'datasets',
        'download',
        KAGGLE_DATASET,
        '--file',
        'players.csv',
        '--path',
        str(LATEST_PLAYERS_DIR),
        '--unzip',
    ]

    if args.force:
        command.append('--force')

    print('Downloading latest Kaggle players.csv...')
    subprocess.run(command, check=True)

    if not LATEST_PLAYERS_CSV.exists():
        raise FileNotFoundError(f'Expected {LATEST_PLAYERS_CSV} after download')

    print('\n✅ Done.')
    print(f'   Latest players.csv → {LATEST_PLAYERS_CSV}')


if __name__ == '__main__':
    main()
