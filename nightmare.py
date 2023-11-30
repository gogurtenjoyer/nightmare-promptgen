import os
from typing import Literal, Union
import random as r
import re

#import torch

from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from transformers import pipeline

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
conf = OmegaConf.load(CONF_PATH)
MODEL_REPOS = Literal[tuple(conf['Nightmare']['Models'])]
REPLACE = conf['Nightmare']['Replace']


@invocation_output("nightmare_str_output")
class NightmareOutput(BaseInvocationOutput):
    """Nightmare prompt string output"""
    prompt: str = OutputField(description="The generated nightmare prompt string")


@invocation("nightmare_promptgen", title="Nightmare Promptgen", tags=["nightmare", "prompt"],
            category="prompt", version="1.0.1", use_cache=False)
class NightmareInvocation(BaseInvocation):
    """makes new friends"""

    # Inputs
    prompt: str = InputField(default="", description="prompt for the nightmare")
    temp: float = InputField(default=1.8, ge=0.7, le=3.0, description="Temperature")
    top_p: float = InputField(default=0.9, ge=0.2, le=0.95, description="Top P sampling")
    top_k: int = InputField(default=40, ge=5, le=60, description="Top K sampling")
    repo_id: MODEL_REPOS = InputField(default='cactusfriend/nightmare-promptgen-XL', input=Input.Direct)
    

    def loadGenerator(self, repo_id: str):
        """loads the tokenizer, model, etc for the generator"""
        
        # if you'd like to try hardware acceleration, add the 'device' arg below, like
        # 'device=0' or'torch.device("cuda")' etc - import torch above if the latter  - example:
        
        # generator = pipeline(model=repo_id, tokenizer=repo_id, task="text-generation", device=torch.device("mps"))
        generator = pipeline(model=repo_id, tokenizer=repo_id, task="text-generation")
        return generator


    def censor(self, phrase: str):
        """simple sanitization for sanity"""
        for k, v in REPLACE.items():
            if k.startswith("^") and k.endswith("^"):
                kclean = k.replace("^", "")
                print(kclean)
                phrase = re.sub(kclean, r.choice(REPLACE[k]), phrase, flags=re.IGNORECASE)
            else:
                phrase = re.sub(r"\b{}\b".format(k), r.choice(REPLACE[k]), phrase, flags=re.I | re.M)
        return phrase


    def makePrompts(self, prompt: str, temp: float, p: float, k: int):
        """loads textgen model, generates a (str) prompt, unloads model"""
        generator = self.loadGenerator(self.repo_id)
        output = generator(prompt, max_new_tokens=300, temperature=temp,
                            do_sample=True, top_p=p, top_k=k,
                            num_return_sequences=1)
        del generator
        return self.censor(output[0]['generated_text'])


    def invoke(self, context: InvocationContext) -> NightmareOutput:
        """ does the thing """
        generated = self.makePrompts(self.prompt, self.temp, self.top_p, self.top_k)
        nl, bl, nr = "\n", "\033[1m", "\033[0m"
        context.services.logger.info(f"{nl}{nl}*** YOUR {bl}NIGHTMARE{nr} IS BELOW ***{nl}{generated}{nl}")
        return NightmareOutput(prompt=generated)
