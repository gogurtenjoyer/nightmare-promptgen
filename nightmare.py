import os
from typing import Literal
import random as r
import re
from unicodedata import category
import torch
import yaml
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from invokeai.backend.util.devices import TorchDevice
from invokeai.invocation_api import (
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
CONF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightmare-new.yaml")
DEFAULT_CONF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightmare-new.default.yaml")
try:
    #conf = OmegaConf.load(CONF_PATH)
    with open(CONF_PATH, 'r') as yamldata:
        conf = yaml.safe_load(yamldata)
except OSError as e:
    print("nightmare-new.yaml not found or not parsed correctly - loading nightmare-new.default.yaml instead...")
    print(e)
    #conf = OmegaConf.load(DEFAULT_CONF_PATH)
    with open(DEFAULT_CONF_PATH) as yamldata:
        conf = yaml.safe_load(yamldata)
MODEL_REPOS = Literal[tuple(conf['Nightmare']['Models'])]
INSTRUCT_REPOS = Literal[tuple(conf['Nightmare']['InstructModels'])]
REPLACE = conf['Nightmare']['Replace']
MEM_CACHE = True
SYSTEM_PROMPT = "You are a visual prompt generator. Output only the prompt, adding description and visual detail, nothing else. Only use quotation marks when displaying literal text from signs, t-shirts, or similar."

dev = TorchDevice.choose_torch_device()
precis = TorchDevice.choose_torch_dtype()


@invocation_output("instruction_str_output")
class NightmareInstructOutput(BaseInvocationOutput):
    """Nightmare instruct string output"""
    instruction: str = OutputField(description="The original instruction for the prompt")
    prompt: str = OutputField(description="The generated nightmare prompt string")

@invocation_output("nightmare_str_output")
class NightmareOutput(BaseInvocationOutput):
    """Nightmare prompt string output"""
    prompt: str = OutputField(description="The generated nightmare prompt string")


@invocation("nightmare_promptgen_instruct", title="Nightmare Promptgen - Instruct",
            tags=["nightmare", "prompt", "promptgen", "instruct"],
            category="prompt", version="1.0.0", use_cache=False)
class NightmareInstructInvocation(BaseInvocation):
    """Instruct a LLM to slop up prompts for you."""

    # Inputs
    instruction: str =          InputField(default="",
                                           description="Instruction to request a generated prompt: 'Make me a...', 'A painting of...'", ui_component=UIComponent.Textarea)
    system_prompt: str =        InputField(default=SYSTEM_PROMPT,
                                           description="Be careful changing this, as model was trained on this default.", ui_component=UIComponent.Textarea)
    temp: float =               InputField(default=1.2, ge=0.3, le=4.0, description="Temperature")
    top_p: float =              InputField(default=0.8, ge=0.2, le=0.95, description="Top P sampling")
    top_k: int =                InputField(default=15, ge=5, le=80, description="Top K sampling")
    min_p: float =              InputField(default=0.2, ge=0.0, le=0.5, description="Min P sampling")
    model: INSTRUCT_REPOS =     InputField(default='nightmare-promptqwen3-instruct', input=Input.Direct)


    def loadGenerator(self, model_choice: str):
        """loads the tokenizer, model, etc for the generator"""

        model_conf = conf["Nightmare"]["InstructModels"][model_choice]
        if model_conf["type"] == "lora":
            tokenizer = AutoTokenizer.from_pretrained(model_conf["repo"])
            model = AutoPeftModelForCausalLM.from_pretrained(model_conf["repo"], torch_dtype=precis).to(dev)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_conf["repo"], use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_conf["repo"], torch_dtype = precis).to(dev)
        return tokenizer, model


    def censor(self, phrase: str):
        """simple sanitization for sanity"""
        for k, v in REPLACE.items():
            if k.startswith("^") and k.endswith("^"):
                kclean = k.replace("^", "")
                phrase = re.sub(kclean, r.choice(REPLACE[k]), phrase, flags=re.I | re.M)
            else:
                phrase = re.sub(r"\b{}\b".format(k), r.choice(REPLACE[k]), phrase, flags=re.I | re.M)
        return phrase


    @torch.inference_mode
    def instructPrompt(self, instruction: str, system_prompt: str, temp: float,
                    p: float, k: int, min_p: float):
        """loads instruct model, generates a (str) prompt, unloads model"""
        tokenizer, model = self.loadGenerator(self.model)
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": self.instruction},
        ]
        prompt = tokenizer.apply_chat_template(
                        msgs, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False
                        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=350,
                             do_sample=True, temperature=temp,
                             top_p=p, top_k=k, min_p=min_p)

        result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        del inputs, out, tokenizer, model
        return self.censor(result.rstrip())


    def invoke(self, context: InvocationContext) -> NightmareInstructOutput:
        """ does the thing """
        generated = self.instructPrompt(self.instruction, self.system_prompt, self.temp, self.top_p, self.top_k, self.min_p)

        nl, bl, nr = "\n", "\033[1m", "\033[0m"
        context.logger.info(f"{nl}{nl}*** THE RESULT OF YOUR {bl}INSTRUCTION{nr} IS BELOW ***{nl}{generated}")
        return NightmareInstructOutput(instruction=self.instruction, prompt=generated)


@invocation("nightmare_promptgen", title="Nightmare Promptgen", 
            tags=["nightmare", "prompt", "promptgen"],
            category="prompt", version="2.0.0", use_cache=False)
class NightmareInvocation(BaseInvocation):
    """makes new friends"""

    # Inputs
    prompt: str =               InputField(default="", 
                                           description="starting point for the generated prompt", ui_component=UIComponent.Textarea)
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
    include_starter: bool =     InputField(default=True, description="Include your prompt starter with the output or not.")
    model: MODEL_REPOS =        InputField(default='nightmare-promptqwen3', input=Input.Direct)


    def loadGenerator(self, model_choice: str):
        """loads the tokenizer, model, etc for the generator"""     
        model_conf = conf["Nightmare"]["Models"][model_choice]
        if model_conf["type"] == "lora":
            model = AutoPeftModelForCausalLM.from_pretrained(model_conf["repo"], torch_dtype=precis).to(dev)
            tokenizer = AutoTokenizer.from_pretrained(model.peft_config["default"].base_model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_conf["repo"], use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_conf["repo"], torch_dtype = precis).to(dev)
        return tokenizer, model


    def censor(self, phrase: str):
        """simple sanitization for sanity"""
        for k, v in REPLACE.items():
            if k.startswith("^") and k.endswith("^"):
                kclean = k.replace("^", "")
                phrase = re.sub(kclean, r.choice(REPLACE[k]), phrase, flags=re.I | re.M)
            else:
                phrase = re.sub(r"\b{}\b".format(k), r.choice(REPLACE[k]), phrase, flags=re.I | re.M)
        return phrase


    @torch.inference_mode
    def makePrompts(self, prompt: str, temp: float,
                    p: float, k: int, mnt: int, mnnt: int, time: float,
                    reppen: float, typical: float):
        """loads textgen model, generates a (str) prompt, unloads model"""
        tokenizer, model = self.loadGenerator(self.model)
        begin = tokenizer.bos_token or ""
        input = tokenizer(f"{begin}{prompt}", return_tensors="pt", padding=True).to(dev)
        eos_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
        output = model.generate(input.input_ids, attention_mask=input.attention_mask,
                                max_new_tokens=mnt, min_new_tokens=mnnt, 
                                temperature=temp, max_time=time,
                                do_sample=True, top_p=p, top_k=k,
                                typical_p=typical,
                                repetition_penalty=reppen,
                                num_return_sequences=1,
                                eos_token_id=eos_ids,
                                pad_token_id=tokenizer.pad_token_id)

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        del input, output, tokenizer, model
        return self.censor(result.rstrip())


    def invoke(self, context: InvocationContext) -> NightmareOutput:
        """ does the thing """
        prompt_censored = self.censor(self.prompt)
        generated = self.makePrompts(prompt_censored, self.temp, self.top_p, self.top_k,
                    self.max_new_tokens, self.min_new_tokens, self.max_time,
                    self.repetition_penalty, self.typical_p)
        if not self.include_starter:
            generated = generated.replace(prompt_censored, "")

        generated = re.sub(r'\s+',' ', generated)
        generated = "".join(ch for ch in generated if category(ch)[0]!="C") #further clean up weird control characters
        nl, bl, nr = "\n", "\033[1m", "\033[0m"
        context.logger.info(f"{nl}{nl}*** YOUR {bl}NIGHTMARE{nr} IS BELOW ***{nl}{generated}")
        return NightmareOutput(prompt=generated)
