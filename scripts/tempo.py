import os
import time
import json
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import easy_llama as ez
from easy_llama import SamplerParams

os.environ['LIBLLAMA'] = "/Users/victorcarre/Code/Projects/llm-assistant/resources/llama.cpp/build/bin/libllama.dylib"

'''
model = Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf")
'''

config_path = Path("/Users/victorcarre/Code/Projects/llm-assistant/resources/config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

DB_PATH = config["data"]["db_path"]

model_cfg = config["models"]["llm"]
model_path = Path(model_cfg["model_path"])

llm = ez.Llama(str(model_path), n_ctx=2048, n_threads=os.cpu_count() or 4, chat_format="chatml", verbose=False)
thread = ez.Thread(llm)

sp = model_cfg["sampler_params"]
sampler_params = SamplerParams(
    llama=llm,
    seed=sp["seed"],
    top_k=sp["top_k"],
    top_p=sp["top_p"],
    min_p=sp["min_p"],
    temp=sp["temp"],
    penalty_last_n=sp["penalty_last_n"],
    penalty_repeat=sp["penalty_repeat"],
    mirostat=sp["mirostat"]
)

system_prompt = model_cfg["system_prompt"]
stop_tokens = [32000]

user_prompt_1 = """<|im_start|>user
Qu'est-ce que la photochimie, et en quoi peut-elle être une solution d'avenir ?  /no_think<|im_end|>
"""

user_prompt_2 = """<|im_start|>user
Qu'est-ce que la photochimie ?<|im_end|>
<|im_start|>assistant
Photochimie étude réaction chimique provoquer lumière photocatalyseur dégrader polluant air eau produire de l'énergie photochimie processus d'élaboration papier partir bois processus<|im_end|>
<|im_start|>user
Quels sont les défis de l'intégration de la photochimie dans les procédés industriels durables ? Peut-on concevoir des prodrugs photoactivables pour un relargage contrôlé du principe actif dans des zones précises du corps ?<|im_end|>
<|im_start|>assistant
Photochimique étude réaction chimique provoquer lumière photolyse lumière brise lien molécule libérer énergie photodégradation photosynthèse photochimie industrie production énergie solaire traitement<|im_end|>
<|im_start|>user
Quel est l’impact du design des canaux de réacteurs en flux dans les réactions photochimiques ?<|im_end|>
<|im_start|>assistant
Conduction thermique assurer contrôle précis température optimiser réaction photochimique canal influencer transfert chaleur matière temps résidence répartition flux élimination impureté caractéristique affecter la réaction Photochimique améliorer la séparation<|im_end|>
<|im_start|>user
Qu'est-ce que la photochimie, et en quoi peut-elle être une solution d'avenir ? <|im_end|> 
"""

user_prompt_3 = """<|im_start|>system
Utilise les échanges précédents pour répondre à la dernière question posée de manière claire, rigoureuse et scientifique. Ne répète pas les informations déjà mentionnées."<|im_end|>
<|im_start|>user
Qu'est-ce que la photochimie ?<|im_end|>
<|im_start|>assistant
Photochimie étude réaction chimique provoquer lumière photocatalyseur dégrader polluant air eau produire de l'énergie photochimie processus d'élaboration papier partir bois processus<|im_end|>
<|im_start|>user
Quels sont les défis de l'intégration de la photochimie dans les procédés industriels durables ? Peut-on concevoir des prodrugs photoactivables pour un relargage contrôlé du principe actif dans des zones précises du corps ?<|im_end|>
<|im_start|>assistant
Photochimique étude réaction chimique provoquer lumière photolyse lumière brise lien molécule libérer énergie photodégradation photosynthèse photochimie industrie production énergie solaire traitement<|im_end|>
<|im_start|>user
Quel est l’impact du design des canaux de réacteurs en flux dans les réactions photochimiques ?<|im_end|>
<|im_start|>assistant
Conduction thermique assurer contrôle précis température optimiser réaction photochimique canal influencer transfert chaleur matière temps résidence répartition flux élimination impureté caractéristique affecter la réaction Photochimique améliorer la séparation<|im_end|>
<|im_start|>user
Qu'est-ce que la photochimie, et en quoi peut-elle être une solution d'avenir ?<|im_end|> /no_think
"""

full_prompt = user_prompt_3

def text_formatting(raw_response: str) -> str:
    cleaned_text = re.sub(r"<\|im_start\|>", "", raw_response)
    cleaned_text = re.sub(r"<\|im_end\|>", "", cleaned_text)
    cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+(?=\n)", "", cleaned_text)
    cleaned_text = re.sub(r"\n[ \t]+", "\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    return cleaned_text.strip()

def remove_think_blocks(text: str) -> str:
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text

input_tokens = llm.tokenize(full_prompt.encode('utf-8'), add_special=True, parse_special=True)
output_tokens = llm.generate(
    input_tokens,
    n_predict=2000,
    sampler_preset=sampler_params,
)

response = llm.detokenize(output_tokens, special=True).strip()

formatted_response = text_formatting(response)
nothink_response = remove_think_blocks(formatted_response)

print(f"(Nombre de tokens dans le prompt : {len(input_tokens)})")
print(f"(Nombre de tokens générés : {len(output_tokens)})")
print("\n=== Réponse initiale ===\n", response)
print("\n=== Réponse générée ===\n", nothink_response)