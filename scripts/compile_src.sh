#!/bin/bash -xe

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname -- "$SCRIPT_DIR")"

force_build=false
if [[ "${1:-}" == "--build" ]]; then
  force_build=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--build]" >&2
  exit 2
fi

if [[ "$force_build" == true || ! -x "$PARENT_DIR/build/compile_audios" ]]; then
  cmake -S "$PARENT_DIR" -B "$PARENT_DIR/build"
  cmake --build "$PARENT_DIR/build" --target compile_audios
fi

"$PARENT_DIR/build/compile_audios" ~/Downloads/speech_commands_v0.02 ./weights
