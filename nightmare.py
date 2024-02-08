import os
from typing import Literal
import random as r
import re

from omegaconf import OmegaConf
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
except OSError as e:
    print("nightmare.yaml not found or not parsed correctly - loading nightmare.default.yaml instead...")
    print(e)
    conf = OmegaConf.load(DEFAULT_CONF_PATH)
MODEL_REPOS = Literal[tuple(conf['Nightmare']['Models'])]
REPLACE = conf['Nightmare']['Replace']
MEM_CACHE = True

dev = choose_torch_device()
precis = choose_precision(dev)


@invocation_output("nightmare_str_output")
class NightmareOutput(BaseInvocationOutput):
    """Nightmare prompt string output"""
    prompt: str = OutputField(description="The generated nightmare prompt string")

@invocation_output("escaped_quotes_output")
class EscapedOutput(BaseInvocationOutput):
    """The input str with double quotes escaped"""
    prompt: str = OutputField(description="The input str with double quotes escaped")


@invocation("quote_escaper", title="Quote Escaper", tags=["quote", "escaper"],
            category="prompt", version="1.0.0", use_cache=False)
class EscaperInvocation(BaseInvocation):
    """Escapes double quotes from input strings"""
    prompt: str = InputField(default="", description="the string to escape quotes from")

    def invoke(self, context: InvocationContext) -> EscapedOutput:
        return EscapedOutput(prompt=self.prompt.replace('"',r'\"'))

@invocation("nightmare_promptgen", title="Nightmare Promptgen", tags=["nightmare", "prompt"],
            category="prompt", version="1.3.2", use_cache=False)
class NightmareInvocation(BaseInvocation):
    """makes new friends"""

    # Inputs
    prompt: str =               InputField(default="", 
                                           description="starting point for the generated prompt", ui_component=UIComponent.Textarea)
    split_prompt: bool =        InputField(default=False, description="If the prompt is too long, will split it with .and()")
    max_new_tokens: int =       InputField(default=300, ge=3, le=1200, 
                                           description="the maximum allowed amount of new tokens to generate")
    min_new_tokens: int =       InputField(default=30, ge=0, le=800, 
                                           description="the minimum new tokens - NOTE, this can increase generation time")
    max_time: float =           InputField(default=10.0, ge=5.0, le=120.0, 
                                           description="Overrules min tokens; the max amount of time allowed to generate")
    temp: float =               InputField(default=1.8, ge=0.5, le=4.0, description="Temperature")
    typical_p: float =          InputField(default=1.0, ge=0.1, le=4.0, description="Lower than 1.0 seems crazier, higher is more consistent.")
    top_p: float =              InputField(default=0.9, ge=0.2, le=0.98, description="Top P sampling")
    top_k: int =                InputField(default=20, ge=5, le=80, description="Top K sampling")
    repetition_penalty: float = InputField(default=1.0, ge=0.5, le=3.0, description="Higher than 1.0 will try to prevent repetition.")
    repo_id: MODEL_REPOS =      InputField(default='cactusfriend/nightmare-promptgen-XL', input=Input.Direct)


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


    def splitPrompt(self, text):
        """ If the prompt is too long, let's split it up for .and() """
        puncts = ['.', ',', ';', '--']
        splitted = []
        while len(text) > 200:
            cut_where, cut_why = max((text.rfind(punc, 0, 193), punc) for punc in puncts)
            if cut_where <= 0:
                cut_where = text.rfind(' ', 0, 193)
                cut_why = ' '
            cut_where += len(cut_why)
            splitted.append(text[:cut_where].rstrip())
            text = text[cut_where:].lstrip()
        splitted.append(text)
        return splitted


    def makePrompts(self, prompt: str, temp: float, 
                    p: float, k: int, mnt: int, mnnt: int, time: float,
                    reppen: float, typical: float):
        """loads textgen model, generates a (str) prompt, unloads model"""
        generator = self.loadGenerator(self.repo_id)
        output = generator(prompt, max_new_tokens=mnt, min_new_tokens=mnnt, 
                            temperature=temp, max_time=time,
                            do_sample=True, top_p=p, top_k=k,
                            typical_p=typical,
                            repetition_penalty=reppen,
                            num_return_sequences=1,
                            return_full_text=False,
                            pad_token_id=generator.tokenizer.eos_token_id)

        del generator
        return self.censor(output[0]['generated_text'].rstrip())


    def invoke(self, context: InvocationContext) -> NightmareOutput:
        """ does the thing """
        if len(self.prompt) > 1:
            endsWithSpace = (self.prompt[-1] == " ")
        else:
            endsWithSpace = False
        prompt = self.prompt.strip()
        unescaped = self.makePrompts(prompt, self.temp, self.top_p, self.top_k, 
                    self.max_new_tokens, self.min_new_tokens, self.max_time,
                    self.repetition_penalty, self.typical_p)
        generated = unescaped.replace('"', r'\"').rstrip()
        prompt = f"{self.censor(prompt)}{endsWithSpace and ' ' or ''}"
        generated = f"{prompt}{generated}"
        if len(generated) > 200 and self.split_prompt:
            context.services.logger.info("[nightmare promptgen] I AM GONNA SPLIT!!!")
            start = '("'
            end = '").and()'
            split_prompt = self.splitPrompt(generated)
            together = '","'.join([i for i in split_prompt])
            generated = f"{start}{together}{end}"
            generated = re.sub('\s+',' ', generated)
        nl, bl, nr = "\n", "\033[1m", "\033[0m"
        context.services.logger.info(f"{nl}{nl}*** YOUR {bl}NIGHTMARE{nr} IS BELOW ***{nl}{generated}")
        return NightmareOutput(prompt=generated)
