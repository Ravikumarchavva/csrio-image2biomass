from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
AUGUMENTED_DATA_DIR = ROOT_DIR / "data" / "augmented_processed_data"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "original_processed_data"

if __name__ == "__main__":
    print(f"Root directory is set to: {ROOT_DIR}")