"""Run the OCR-enhanced Schedule E extractor against the input suite."""

from pathlib import Path

from .part2 import process_batch


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    process_batch(
        project_root / "input",
        project_root / "output",
        project_root / "work",
    )
