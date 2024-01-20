import os
from typing import Literal, Union
import random as r
import re

from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from transformers import pipeline

from invokeai.backend.util.devices import choose_torch_device, choose_precision

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Input,
    InputField,
    InvocationContext,
    invocation,
    invocation_output,
    OutputField,
    UIComponent
)

# load the config file - thanks to SkunkWorxDark for the update here
CONF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightmare.yaml")
DEFAULT_CONF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightmare.default.yaml")
try:
    conf = OmegaConf.load(CONF_PATH)
except:
    print("nightmare.yaml not found or not parsed correctly - loading nightmare.default.yaml instead...")
    conf = OmegaConf.load(DEFAULT_CONF_PATH)
MODEL_REPOS = Literal[tuple(conf['Nightmare']['Models'])]
REPLACE = conf['Nightmare']['Replace']
MEM_CACHE = True

dev = choose_torch_device()
precis = choose_precision(dev)
# print("***NIGHTMARE PROMPTGEN DEBUG***")
# print(f"device: {dev}, precision: {precis}")
# print("*"*30)

@invocation_output("nightmare_str_output")
class NightmareOutput(BaseInvocationOutput):
    """Nightmare prompt string output"""
    prompt: str = OutputField(description="The generated nightmare prompt string")


@invocation("nightmare_promptgen", title="Nightmare Promptgen", tags=["nightmare", "prompt"],
            category="prompt", version="1.1.0", use_cache=False)
class NightmareInvocation(BaseInvocation):
    """makes new friends"""

    # Inputs
    prompt: str = InputField(default="", description="prompt for the nightmare")
    temp: float = InputField(default=1.8, ge=0.5, le=4.0, description="Temperature")
    top_p: float = InputField(default=0.9, ge=0.2, le=0.98, description="Top P sampling")
    top_k: int = InputField(default=20, ge=5, le=80, description="Top K sampling")
    repo_id: MODEL_REPOS = InputField(default='cactusfriend/nightmare-promptgen-XL', input=Input.Direct)


    def loadGenerator(self, repo_id: str):
        """loads the tokenizer, model, etc for the generator"""     
        generator = pipeline(model=repo_id, tokenizer=repo_id, task="text-generation", use_fast=False, device=dev)
        return generator


    def censor(self, phrase: str):
        """simple sanitization for sanity"""
        for k, v in REPLACE.items():
            if k.startswith("^") and k.endswith("^"):
                kclean = k.replace("^", "")
                phrase = re.sub(kclean, r.choice(REPLACE[k]), phrase, flags=re.I | re.M)
            else:
                phrase = re.sub(r"\b{}\b".format(k), r.choice(REPLACE[k]), phrase, flags=re.I | re.M)
        return phrase


    # def splitPrompt(text):
    #     """ If the prompt is too long, let's split it up for blend/and """
    #     puncts = ['.', ',', ';', '--']
    #     splitted = []
    #     while len(text) > 300:
    #         cut_where, cut_why = max((text.rfind(punc, 0, 293), punc) for punc in puncts)
    #         if cut_where <= 0:
    #             cut_where = text.rfind(' ', 0, 293)
    #             cut_why = ' '
    #         cut_where += len(cut_why)
    #         splitted.append(text[:cut_where].rstrip())
    #         text = text[cut_where:].lstrip()
    #     splitted.append(text)
    #     return splitted


    def makePrompts(self, prompt: str, temp: float, p: float, k: int):
        """loads textgen model, generates a (str) prompt, unloads model"""
        generator = self.loadGenerator(self.repo_id)
        output = generator(prompt, max_new_tokens=300, temperature=temp,
                            do_sample=True, top_p=p, top_k=k,
                            num_return_sequences=1,
                            pad_token_id=generator.tokenizer.eos_token_id)

        del generator
        return self.censor(output[0]['generated_text'].strip())


    def invoke(self, context: InvocationContext) -> NightmareOutput:
        """ does the thing """
        unescaped = self.makePrompts(self.prompt.strip(), self.temp, self.top_p, self.top_k)
        generated = unescaped.replace('"', r'\"').strip()
        nl, bl, nr = "\n", "\033[1m", "\033[0m"
        context.services.logger.info(f"{nl}{nl}*** YOUR {bl}NIGHTMARE{nr} IS BELOW ***{nl}{unescaped}{nl}")
        return NightmareOutput(prompt=generated)
