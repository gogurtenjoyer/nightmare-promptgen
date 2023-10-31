# nightmare-promptgen
An InvokeAI node that supports text generation models (gpt-neo, llama, etc) to create Stable Diffusion prompts

## Installation
Navigate to your nodes folder (in your InvokeAI 3.4 root folder), and run
```
git clone https://github.com/gogurtenjoyer/nightmare-promptgen
```
## Usage
Add the node as you would any other, then connect its output to a Compel positive prompt node, or similar.

Note: the first time you use this node in your workflow, it'll download a text generation model.
The default model is the Nightmare Promptgen XL model `cactusfriend/nightmare-promptgen-XL` (1.5gb), but if you'd like to save memory and download time, feel free to try the original: `cactusfriend/nightmare-invokeai-prompts` (550mb). These models were specifically trained with InvokeAI's Compel prompt format in mind, but feel free to try other text generation models!

## TODO
Support external files for the find/replace and model selection features
