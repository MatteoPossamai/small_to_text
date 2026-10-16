#!/bin/bash -xe

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname -- "$SCRIPT_DIR")"

if [ ! -f "$PARENT_DIR/build/compile_audios" ]; then
  cd $PARENT_DIR && cmake -S . -B build && cmake --build build --target compile_audios
fi

$PARENT_DIR/build/compile_audios  ~/Downloads/speech_commands_v0.02 ./weights
