#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile


def decode_flac(source: Path, destination: Path) -> None:
    if destination.is_file():
        return
    with NamedTemporaryFile(dir=destination.parent, suffix=".wav", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(
            ["flac", "--decode", "--silent", "--force", "-o", str(temporary), str(source)],
            check=True,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_split(source: Path, destination: Path) -> None:
    transcripts = sorted(source.rglob("*.trans.txt"))
    if not transcripts:
        raise ValueError(f"No LibriSpeech transcripts found under {source}")

    wav_directory = destination / "wav"
    wav_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for transcript in transcripts:
        for line in transcript.read_text(encoding="utf-8").splitlines():
            utterance_id, text = line.split(maxsplit=1)
            normalized = " ".join(text.lower().split())
            flac = transcript.with_name(f"{utterance_id}.flac")
            wav = wav_directory / f"{utterance_id}.wav"
            decode_flac(flac, wav)
            rows.append(f"wav/{wav.name}\t{normalized}\n")

    manifest = destination / "utterances.tsv"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination, delete=False) as handle:
        temporary = Path(handle.name)
        handle.writelines(rows)
    temporary.replace(manifest)
    print(f"{manifest}: {len(rows)} utterances", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mini LibriSpeech for LAS training")
    parser.add_argument("train", type=Path, help="Extracted train-clean-5 directory")
    parser.add_argument("validation", type=Path, help="Extracted dev-clean-2 directory")
    parser.add_argument("output", type=Path, help="Where to put WAV files and manifests")
    args = parser.parse_args()

    prepare_split(args.train, args.output / "train")
    prepare_split(args.validation, args.output / "validation")


if __name__ == "__main__":
    main()
