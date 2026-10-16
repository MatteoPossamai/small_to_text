#! /bin/python3
import os
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_DIR / "data"


def write_dataset(path: Path, samples: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for filename, label in samples:
            if any(character.isspace() for character in filename + label):
                raise ValueError("Dataset filenames and labels must not contain whitespace")
            output.write(f"{filename} {label}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./auto_label_dataset.py <PATH_TO_DATASET>")
        sys.exit(1)

    dataset_base_location = sys.argv[1]
    all_recordings: dict[str, list[str]] = {
        os.path.basename(directory): [
            filename for filename in files if not filename.endswith(".md")
        ]
        for directory, subdirectories, files in os.walk(dataset_base_location)
        if not subdirectories
    }

    training: list[tuple[str, str]] = []
    scoring: list[tuple[str, str]] = []

    for label, recordings in all_recordings.items():
        training_samples = len(recordings) * 9 // 10
        training.extend((filename, label) for filename in recordings[:training_samples])
        scoring.extend((filename, label) for filename in recordings[training_samples:])

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    write_dataset(DATASET_DIR / "training.txt", training)
    write_dataset(DATASET_DIR / "scoring.txt", scoring)
