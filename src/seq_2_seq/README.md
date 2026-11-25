# Sequence to Sequence: LAS

Paper: [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211)
This is the reproduction of the code of a LAS network as explained in the paper linked above. This will contain a short description of the network itself, the learning that came from it, and the things that were not implemented here.

## Components of the network

### Listener

This is the classic encoder in an encoder/decoder architecture.
It gets in the audio, divided in multiple 10ms frames, encoded with `log-mel`. Then, the input is compressed. The compression happens with the fact that layer N+1 has as the input both the output vectors of layer N of 2 `LSTM`. So if layer N has M `LSTM`, layer N + 1 has M / 2 `LSTM`. Every layer the number of outputs halves.

This is what is referred as a pyramidal `LSTM` (`pLSTM`).

To give more context, each layer of `LSTM` has two outputs, one with the frame played forward, the other with it played backward. This means that there is more context of the waves, and that the information flow doubles.

Moreover, this design is called bi-directional `LSTM` (`BLSTM`).

The overall structure is them a `pBLSTM`.

### Attention and Speller

The attention layer is a classic layer that applies attention for previously seen characters.

The speller is a two layer `LSTM` that then feeds into a `MLP` to give out the output character. This is appended to the string result, until it generates the `<EOS>` character.

### Thoughts

The architecture is interesting, and follows a classic encoder/decoder paradigm. In this case, the decoder is much simpler. It only has 2 layers and do not require additional information. The only complication is the attention layer added. But conceptually, the latter part of the network has been much easier to understand and follow. The encoder was more complicated and there were more smart and interesting choices made in the process of creating it.

Attention is interesting, and works pretty much the same as in transformer based architectures. Once grasped is quite straight forward.

## Learning

The first obvious learning has been the shape and logic that was used in the network itself. Moreover, this gives an idea of the possible practical issues that there are in passing in information in a network, and balancing the desire of giving as much context as possible, and the computational limits that hardware and/or time required for training would mean. An example is the compression of the input in the Listener with the pyramid network.

Another, lower level learning, has been how network retain information from the past. From RNN and LSTM and their mechanisms, to the higher level concept of attention, and why keeping a single stream of information that has no semantic bindings with the past is not enough to represent and produce text.

## Limitations and difference with paper

In the paper, the final MLP is not specified what shape has, so in this example were decided to use a one layer MLP with 256 hidden states.

The original paper at inference time uses `beam search` rather than more greedily keeping only a single sequence. This is a simple change that can be implemented as an extension when desired.
