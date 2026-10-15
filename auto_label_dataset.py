#! /bin/python3
import glob
import os
import sys

DIR_PATH: str = os.path.dirname(os.path.realpath(__file__))
DATASET_ANNOTATION_FILE = DIR_PATH + "/src/dataset_annotation.hpp"

IMPORTS = """
#include <string>
#include <tuple>
#include <vector>

using dataset_t = std::vector<std::tuple<std::string, std::string>>;

"""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./auto_label_dataset.py <PATH_TO_DATASET>")

    dataset_base_location: str = sys.argv[1]
    all_recordings: dict[str, list[str]] = {
        x[0].split("/")[-1]: list(filter(lambda x: not x.endswith(".md"), x[2]))
        for x in os.walk(dataset_base_location)
        if len(x[1]) == 0
    }

    training: list[tuple[str, str]] = []
    scoring: list[tuple[str, str]] = []

    for key, recordings in all_recordings.items():
        n_recordings = len(recordings)

        training_samples = n_recordings * 9 // 10
        training.extend(
            [(value, key) for value in all_recordings[key][:training_samples]]
        )
        scoring.extend(
            [(value, key) for value in all_recordings[key][training_samples:]]
        )

    training_entries: str = ",\n    ".join(
        f'{{"{filename}", "{label}"}}' for filename, label in training
    )

    scoring_entries: str = ",\n    ".join(
        f'{{"{filename}", "{label}"}}' for filename, label in scoring
    )

    TRAININGS = f"""
dataset_t trainings = {{
    {training_entries}
}}
    """

    SCORINGS = f"""
dataset_t scorings = {{
    {scoring_entries}
}}
    """

    with open(DATASET_ANNOTATION_FILE, "w") as f:
        f.write(IMPORTS + TRAININGS + SCORINGS)
