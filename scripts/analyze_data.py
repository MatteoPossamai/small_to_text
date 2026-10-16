from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

REPO_DIR: Path = Path(__file__).resolve().parent.parent
WEIGHTS_REPO: Path = REPO_DIR / "weights"
SUB_FOLDERS: list[Path] = [WEIGHTS_REPO / "training", WEIGHTS_REPO / "scoring"]

if __name__ == "__main__":
    words: list[str] = list(os.listdir(SUB_FOLDERS[0]))
    all_files: list[str] = []

    for sub_folder in SUB_FOLDERS:
        for word in words:
            current_folder = sub_folder / word
            all_files.extend([str(s) for s in current_folder.glob("*.wav")])

    all_data = np.concatenate([np.fromfile(f, dtype="<i2") for f in all_files])

    plt.hist(all_data, bins=100)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
