from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "infoot")]

from bank import main


if __name__ == "__main__":
    main(ROOT, "cat")
