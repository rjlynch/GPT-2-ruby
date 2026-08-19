# Ruby GPT2

GPT 2 implemented in Ruby.
Loads the public weights for the small version of GPT 2 into our model and
provides a script for poking around with GPT 2.

Implementation is based on the Python in Build a Large Language Model From
Scratch, hand ported to Ruby.

<img width="591" height="320" alt="Screenshot 2026-08-19 at 16 08 33" src="https://github.com/user-attachments/assets/b0ff0331-3c8b-46e0-bda9-e2857630eee8" />

## Install

Clone this repo

Download libtorch

```
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.13.0.zip > libtorch.zip
unzip -q libtorch.zip
```

Point bundler at it

```
bundle config set build.torch-rb --with-torch-dir=$(pwd)/libtorch
```

Bundle

```
bundle install
```

Download the GPT2 weights

```
curl -fL \
  -o weights/gpt2-small-124M.safetensors \
  https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.safetensors
```

Run the script

```
bundle exec ruby main.rb
```
