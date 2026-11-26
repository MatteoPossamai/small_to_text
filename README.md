# Small To Text

A learning experiment meant at implementing ML algorithms and neural networks from scratch.

## Introduction

AI and ML, in the past years, have changed the world. This technology is turning out to be incredibly powerful in the things it can already do, and even more in its incredible promises of a better future.
This can well be the defining technology of our generation, if not of our specie.

This made me want to understand more of the basics and underlying concepts and mechanisms that are behind the modern LLMs, using a first principle approach and getting my hands dirty. Not only I wanted to understand the latest and greatest, but I also wanted to have some history of this field, to see more of a progression rather than just the finished pieces.

This work is my journey, in which little practical was accomplished, but lots of learning and fun happened. Code and artifacts can be found [here](https://github.com/MatteoPossamai/small_to_text). Let me guide through it. I hope this can be inspiring and give out some interesting little learning pieces.
Note: I will start with learnings and leave the journey later, since the latter is the majority of this article, but might be less interesting for many, so I will try to give
you some value in exchange for your time, since none of us is earning anything from doing this. I sincerely hope you enjoy, and apologise if you do not.

## Goal

The overall goal was to build some ML based system that could to speach to text, in a quite agile fashion. Initially I though about a model that could run on my smartwatch (a Google Pixel Watch 3),
but later I had to change my mind, given the hardware at hand was not enough to match demand for a complex enough model.

Also, since in the past months I have been learning C++ with the goal of efficient and performant code, I wanted to implement it all from scratch, using only the `stl`,
while still trying to be as efficient as possible, for a CPU (note, not a GPU) training and inference. And this has funny outcomes, more on that later.

Of course, this should have been done with minimal input from AI for code generation, only to help when blocked and for mondane tasks, like tests, and helping me understand the stuff better.

## Learnings

My initial mental model of AI was more or less:

> ... it is like dark magic, you try seemingly random stuff, on expensive hardware, until something works...

Doing this changed a bit my view (that was a understanding and ignorance issue to begin with anyway).

I got the point that it is more a matter of information flow. What I mean is, we are putting into black box systems
some components such that we can give them, encoded with number, information that is useful for the task, for example the
position in the phrase of a word or how a word pays attention to another and so on. We want to be able to give as much information
as possible to the network.

But this is an engineering problem, and as in every case, there are constraints. In this case, that is that putting more and more
information into these more and more complex systems, increases latency, times for training and inferencing, resulting in terrible
final user experience, or scary-high bills for high end hardware and electricity.

Hence, Machine Learning is like a logistical problem, in which you want to pass as much useful information as possible for the task at hand,
without compressing too much of them, but without becoming broke in the process or without taking ages. Researchers have pulled off very
smart ideas, mostly in the information compression and passing as much as I can tell, but I did not find as much when it comes to optimizations.

This might be due to the fact that the hardware that most of these workflow runs on (GPUs) are intellectual property of Nvidia and few others
that have generally inferior hardware (at the time of writing). Hence, there is a wall of how much software can actually get more efficient for
such demanding workflows in such hardware. This on a low level. At a higher level, there have been different attempts at making attention slighly
less expensive (since it has quadratic cost, hence longer contexts means more cost).

Data is quite paramount in what the outcome is, and now it makes more and more sense how AI companies are working hard on scraping and DoS-ing the
internet to get the most out of it, even infringing copyright or similar rules. I am aware that making some derivative data from the original
(like shifting a bit a registered sound or hiding some frequencies to simulate noise) is a very normal thing to do, and I did it myself to get
the best out of my data, and I was interested in finding out how much of the training data nowadays are sythetic. I rememember this being a topic
in the first hour of the ChatGPT explosion back in 2023/2024, where there were plenty of companies starting to do that, and now I did not hear about any
of that anymore. Probably since the labs do it already by themselves, so there is no big market to be attacked? I don't know. I would be intersted in hearing
people's opinion.

## Chapter 1: Classification

First of all, what I wanted was to have a small classificator of 1 second audios (in the form of `.wav` files) that was able to recognize them and give out the correct answer at inference time. The code can be found under `src/classificator` in the repository.

As a training and validation dataset I used the [Google Speach Commands v2 Dataset](https://huggingface.co/datasets/google/speech_commands).
This proved to be quite wide. I tried with a number of experiments, starting from a vanilla neural network, then trying to do smarter things,
like shifting the audio input a bit aheard or before, trying to make it generalize more. Then I changed the input to be a bit more compressed, with `log-mel` waves.
This gave me an accuracy that did not go over 80%.

Then I moved into the territory of Convolutional Neural Networks. This has been a very interesting segway, since it helped me understand the concept of convolution well,
and which fields it applies the most. Was interesting to see how this was much more used when dealing with images rather than audios, and this led me to implement and test
out separately also a small convolution on images, to detect edges, generate grayscale and "compressing" image quality. All of this was fun.
I really encourage anyone reading to have a look into the world of convolution, since it might be more interesting than you think, and could have useful use cases in your
daily life and daily workflows.

In the end, the best accuracy I was able to achieve was 85%, available at the tag `0.4.0` on the Github repository. That was quite fun, and engaging.
I was sending networks to train on my laptop and seeing one CPU pinned at 100% from `btop`, and that was really exciting. Even though a bit slow.
Some real world test told me that:

1. This model was not correct enough (still so many mistakes, especially when with high noise audios)
2. Was impractical (if I spoke more than a word or more than one second was useless, and guys is hard to speak for exactly one second lol)
3. Was platform specific: I had optimized the code for my `x86-64` laptop, that would have not worked well with my smartwatch. Here is where I trashed that original idea

Then I though, OK, time to level up, and see what people did next, and this leads us to...

## Chapter 2: LSTMs and Attention

The sequence to sequence nature of speach to text meant that was time to learn aboud RNN and LSTM. Here some great resources:

- [Karpathy on RNN](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [LSTM visually explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

With these basis, then I read the [LAS](https://arxiv.org/abs/1508.01211) paper. This is a pre-transformer architecture, based on putting together a bunch of
LSTM in an encoder and a decoder, and putting an attention layer in between. In the original paper, the generator's tokens are
characters and not words as with modern LLMs.

This was very intersting to see, because for a stranger to this world meant understanding the physics and the mental models that drove
innovations and breakthroughs in the AI world.

In a nutshell, is all a matter of passing on information to the network, trying to balance as much as possible the information that you are passing in
based on how important that bits are, like relative attention and so on, to the resource constraint.

Hardware and power are not infinite, and in my case very scarse. This meant that in the end, I gave up training this network after spending
4 hours having it hammering my CPU and finishing not even 2 training epochs. For this I used a larger pair of datasets, but the loss was not improving
fast enough, and my estimation is that with current hardware, without spending some money, I would have to wait 341 days before having a model with an accuracy
that was decent over the training set, without knowing if that would perform actually better.

Plus, current speach to text are transformer based architectures, so I am already aware that this cannot be the best design, and this made me prefer
not to increase my electricity bill and consume my laptop CPU for pretty much no return at all.
The code is being tested to work when training and inferencing, with terrible results in terms of generation, given the scarse
quality and time spent training it.

## What is next

After making some experience with generic classificator networks, that are likely to generalize OK, and some more specific machinery
for a given task, the probable next step is to experimet with the architecture I have been venting about: transformers.
This time would be cool to try to write it using CUDA directly, so there is a change I can write and run my own GPT or something,
following the example of karpathy or similar. That would be interesting both from the ML and the C++/performance side.

Alternatively, I could turn more into the original idea, and start to think and push more and more the idea of edge AI, for devices like
my phone or my smartwatch, and see what I can achieve and if there are any improvements that I can have in my life there, of course without
resorting to any cloud provider like Claude or OpenAI, since that is not that novelty and can already been done easily.

Time will tell, for now I will find some more interesting papers to read and experiments that excite me.

## Conclusion

The world of current AI is fascinating, confusing, unintuitive and also extremely challenging. But is so fun.
I had a great time learnign all of these stuff on my time, and I can't wait for more, hoping to get some more interesting insights,
once we approach the edge of human knowledge and understanding of these systems.

Hang around if you are interested, or get in contact if you'd like to exchange some ideas. Many thanks
