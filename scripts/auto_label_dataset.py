#! /bin/python3
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

    dataset_base_location = Path(sys.argv[1])

    # This list already keeps recordings from the same person together.
    scoring_files: set[str] = set()
    with (dataset_base_location / "testing_list.txt").open(encoding="utf-8") as input:
        for line in input:
            scoring_files.add(line.strip())

    training: list[tuple[str, str]] = []
    scoring: list[tuple[str, str]] = []

    for label_dir in sorted(dataset_base_location.iterdir()):
        if not label_dir.is_dir():
            continue

        for recording in sorted(label_dir.glob("*.wav")):
            sample = (recording.name, label_dir.name)
            relative_path = f"{label_dir.name}/{recording.name}"
            if relative_path in scoring_files:
                scoring.append(sample)
            else:
                training.append(sample)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    write_dataset(DATASET_DIR / "training.txt", training)
    write_dataset(DATASET_DIR / "scoring.txt", scoring)
