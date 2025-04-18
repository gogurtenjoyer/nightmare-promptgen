# nightmare-promptgen
An InvokeAI node that supports text generation models (gpt-neo, llama, etc) to create Stable Diffusion prompts - by default, it works with both a small and large gpt-neo model I made, trained on InvokeAI/Compel-formatted prompts.

It runs the text generation model on your chosen InvokeAI GPU/hardware, and offers configurable sampling, as well as word replacement. It also provides the ability to set a limit on the amount of tokens generated, so you can adhere to Stable Diffusion's token limit, or increase it and split the token using the [Prompt Tools](https://github.com/skunkworxdark/prompt-tools) nodes' auto `and()` node.

![a screenshot of the Nightmare Promptgen node interface](/nightmare-promptgen-screenshot.png?raw=true)


## Installation
Navigate to your nodes folder (in your InvokeAI 3.4+ root folder), and run
```
git clone https://github.com/gogurtenjoyer/nightmare-promptgen
```
Alternately, you can download a zip of this repo with the green 'Code' button in the upper-right here (then 'Download ZIP'), and unzip it in your nodes folder.

### Updating
To update, just `cd` into the `nightmare-promptgen` folder in your InvokeAI `nodes` folder, and run `git pull`. Otherwise, you can just download `nightmare.py` from this page and replace the older one.

## Usage
Add the node as you would any other, then connect its output to a Compel positive prompt node, or similar.

Note: the first time you use this node in your workflow, it'll download a text generation model.
The default model is the `Nightmare Promptgen 3` model (1.8gb or so), but if you'd like to save memory and download time, feel free to try the original: `cactusfriend/nightmare-invokeai-prompts` (550mb) or the `XL` version (1.5gb). These models were specifically trained with InvokeAI's Compel prompt format in mind, and NP3 is particularly smart, but feel free to try other text generation models!

## The `nightmare.yaml` config file
You can duplicate or rename `nightmare.default.yaml` to `nightmare.yaml`, and the node will load this file, instead of the default. It'll fall back to the default file if there's a problem with your customized `nightmare.yaml`, as well.

You can add Huggingface repo IDs (or local file paths) for textgen models to the `Models` section, and words and a list of their replacements to the `Replace` section.

### New (old) replace behavior
The default word replacement behavior checks that the word is separated from others by whitespace or punctuation; for example:

- replacing 'ape' in "my ape friend" would work, but in "I love grapes", it wouldn't.

If you'd like to override the 'word check' behavior, you can surround your word in `^` signs. In this case, `^ape^` would replace the text in both examples above. There's an example of this in `nightmare.default.yaml`.

## Tips and Tricks

Adjusting the min and max new tokens generated can alter generation time dramatically. Because of this, there's also a max time setting that'll let you set (in seconds) the amount of allowed generation time. It'll overrule the other settings - think of it as a 'failsafe'.
