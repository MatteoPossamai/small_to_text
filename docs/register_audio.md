# Register/Play audio

If you wanted to register or play new audios that can
be useful for the project, here are the commands
to do it:

## Record

```shell
 ❯ arecord -d 1 -f S16_LE -r 16000 -c 1 output.wav
```

Parameters:

- `-d 1`: Record for 1 second.
- `-f S16_LE`: Signed 16-bit little-endian PCM.
- `-r 16000`: Sample rate of 16 kHz.
- `-c 1`: Mono audio.

## Play

```shell
 ❯ aplay output.wav
```
