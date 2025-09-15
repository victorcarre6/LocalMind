import os
import time
import json
from pathlib import Path
import easy_llama as ez
from easy_llama import SamplerParams
from langdetect import detect
import re

from main import insert_conversation_if_new, init_db_connection

# === CONFIG & INITIALISATION ===

os.environ['LIBLLAMA'] = "/Users/victorcarre/Code/Projects/llm-assistant/resources/llama.cpp/build/bin/libllama.dylib"
config_path = Path(__file__).parent.parent / "resources" / "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

db_path = config["data"]["db_path"]

model_cfg = config["models"]["llm"]
model_path = Path(model_cfg["model_path"])

llm = ez.Llama(str(model_path), n_ctx=33280, n_threads=os.cpu_count() or 4, chat_format="chatml", verbose=False)
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
stop_tokens = [151645]

# === FONCTIONS UTILITAIRES ===

def text_formatting(raw_response: str) -> str:
    cleaned_text = re.sub(r"<\|im_start\|>", "", raw_response)
    cleaned_text = re.sub(r"<\|im_end\|>", "", cleaned_text)
    cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+(?=\n)", "", cleaned_text)
    cleaned_text = re.sub(r"\n[ \t]+", "\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)

    # --- Supprimer un ':' au début et tout retour à la ligne initial ---
    cleaned_text = re.sub(r"^\s*:\s*", "", cleaned_text)

    return cleaned_text.strip()

def remove_think_blocks(text: str) -> str:
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text

def add_no_think(prompt: str) -> str:
    pattern = r"(<\|im_start\|>user.*?)(<\|im_end\|>)"
    matches = list(re.finditer(pattern, prompt, flags=re.DOTALL))
    if matches:
        last_match = matches[-1]
        start, end = last_match.span(2)  # position de <|im_end|> du dernier bloc
        prompt = prompt[:start] + " /no_think" + prompt[start:]
    return prompt

# === FONCTION PRINCIPALE ===

def generate_response(user_input: str, input_text: str, enable_thinking: bool, show_thinking: bool, ephemeral_mode: bool) -> str:
    if not enable_thinking and "/no_think" not in input_text:
        input_text = add_no_think(input_text)
    start_generate = time.time()    
    input_tokens = llm.tokenize(input_text.encode('utf-8'), add_special=True, parse_special=True)
    output_tokens = llm.generate(
        input_tokens,
        n_predict=20000,
        sampler_preset=sampler_params,
        stop_tokens=stop_tokens
    )
    if 32000 in output_tokens:
        output_tokens = output_tokens[:output_tokens.index(32000)]
    raw_response = llm.detokenize(output_tokens, special=True).strip()
    formatted_response = text_formatting(raw_response)
    clean_response = remove_think_blocks(formatted_response)
    end_generate = time.time()
    print(f"Durée de génération de la réponse : {end_generate - start_generate:.1f} s")
    # Insert directement dans la base SQL
    start_sync = time.time()
    init_db_connection(db_path)
    if not ephemeral_mode:
        insert_conversation_if_new(user_input, clean_response, model_path.name)
    end_sync = time.time()
    print(f"Échange ajouté à la base de donnée en {end_sync - start_sync:.1f} s.")
    if not show_thinking:
        return clean_response
    else:
        return formatted_response

# === BLOC PRINCIPAL ===

if __name__ == "__main__":
    prompt = input("Tape ta question : ")
    print(generate_response(prompt))