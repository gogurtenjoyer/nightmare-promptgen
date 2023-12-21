# nightmare-promptgen
An InvokeAI node that supports text generation models (gpt-neo, llama, etc) to create Stable Diffusion prompts

## Installation
Navigate to your nodes folder (in your InvokeAI 3.4+ root folder), and run
```
git clone https://github.com/gogurtenjoyer/nightmare-promptgen
```
Alternately, you can download a zip of this repo with the green button in the upper-right here, and unzip it in your nodes folder.

## Usage
Add the node as you would any other, then connect its output to a Compel positive prompt node, or similar.

Note: the first time you use this node in your workflow, it'll download a text generation model.
The default model is the Nightmare Promptgen XL model `cactusfriend/nightmare-promptgen-XL` (1.5gb), but if you'd like to save memory and download time, feel free to try the original: `cactusfriend/nightmare-invokeai-prompts` (550mb). These models were specifically trained with InvokeAI's Compel prompt format in mind, but feel free to try other text generation models!

## The `nightmare.yaml` config file
You can duplicate or rename `nightmare.default.yaml` to `nightmare.yaml`, and the node will load this file, instead of the default. It'll fall back to the default file if there's a problem with your customized `nightmare.yaml`, as well.

You can add Huggingface repo IDs (or local file paths) for textgen models to the `Models` section, and words and a list of their replacements to the `Replace` section.

### New (old) replace behavior
The default word replacement behavior checks that the word is separated from others by whitespace or punctuation; for example:

- replacing 'ape' in "my ape friend" would work, but in "I love grapes", it wouldn't.

If you'd like to override the 'word check' behavior, you can surround your word in `^` signs. In this case, `^ape^` would replace the text in both examples above. There's an example of this in `nightmare.default.yaml`.
