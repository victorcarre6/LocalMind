import heapq
import json
import logging
import os
import re
import sqlite3
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import hashlib
import faiss
import numpy as np
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import T5ForConditionalGeneration, T5Tokenizer, logging as transformers_logging, pipeline
from keybert import KeyBERT

# === INITIALISATION ===

# --- Configuration projet & chargement du fichier config ---
PROJECT_ROOT = Path(__file__).parent.parent
config_path = PROJECT_ROOT / "resources" / "config.json"

def expand_path(value):
    """Convertit un chemin relatif en chemin absolu"""
    # Si c'est déjà un Path, on le retourne directement
    if isinstance(value, Path):
        return str(value.resolve())
    
    # Expansion du ~ et conversion en Path
    expanded = Path(os.path.expanduser(value))
    
    # Si le chemin est relatif, on le combine avec PROJECT_ROOT
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    
    return str(expanded.resolve())

def load_config(config_path):
    """Charge la configuration avec gestion robuste des chemins"""
    config_path = expand_path(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    
    # Configuration de base
    config = {
        "environment": {
            "venv_activate_path": expand_path(raw_config["environment"]["venv_activate_path"]),
            "conversations_path": expand_path(raw_config["environment"]["conversations_path"])
        },
        "scripts": {
            "sync_script_path": expand_path(raw_config["scripts"]["sync_script_path"]),
            "llm_script_path": expand_path(raw_config["scripts"]["llm_script_path"])
        },
        "data": {
            "db_path": expand_path(raw_config["data"]["db_path"]),
            "stopwords_file_path": expand_path(raw_config["data"]["stopwords_file_path"])
        },
        "models": raw_config["models"],  # On conserve la structure originale
        "memory_parameters": raw_config["memory_parameters"]
    }
    
    # Traitement spécial pour le model_path
    config["models"]["llm"]["model_path"] = expand_path(raw_config["models"]["llm"]["model_path"])
    
    return config

config = load_config(config_path)

# --- Logging & suppression des avertissements ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# --- Chargement des stopwords ---
stopwords_path = config["data"]["stopwords_file_path"]
with open(stopwords_path, "r", encoding="utf-8") as f:
    french_stop_words = set(json.load(f))

combined_stopwords = list(ENGLISH_STOP_WORDS.union(french_stop_words))

# --- Connexion à la base SQLite ---
db_path = config["data"]["db_path"]
if not os.path.exists(db_path):
    default_db_path = os.path.join(os.path.dirname(db_path), "conversations_example.db")
    if os.path.exists(default_db_path):
        config["db_path"] = default_db_path
    else:
        raise FileNotFoundError(f"Neither {db_path} nor {default_db_path} exist.")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# --- Initialisation de l'index vectoriel ---
VECTOR_DIM = 384
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# --- Initialisation des modèles avec fallback modèle local ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = config.get("summarizing_model", "plguillou/t5-base-fr-sum-cnndm")
local_model_dir = PROJECT_ROOT / "resources" / "models" / "t5-base-fr-sum-cnndm"
try:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    summarizing_model = T5ForConditionalGeneration.from_pretrained(model_name)
except Exception as e:
    print("Modèle introuvable sur Hugging Face, chargement local en cours...")
    if not os.path.exists(local_model_dir):
        raise FileNotFoundError(f"Le modèle local {local_model_dir} est introuvable.")
    tokenizer = T5Tokenizer.from_pretrained(local_model_dir)
    summarizing_model = T5ForConditionalGeneration.from_pretrained(local_model_dir)
summarizing_pipeline = pipeline(task="summarization", model=summarizing_model, tokenizer=tokenizer, framework="pt")

system_prompt = config["models"]["llm"]["system_prompt"]

# === STRUCTURES DE DONNÉES ===

@dataclass
class KeywordsData:
    kw_lemma: str
    weight: float
    freq: int
    score: float

@dataclass
class ContextData:
    user_input: str
    llm_output: str
    score_kw: float
    llm_model: str
    timestamp: str
    convo_id: str
    score_rerank: float

    @property
    def combined_score(self):
        return self.score_kw * self.score_rerank

# === PIPELINE LINGUISTIQUE ===

class LanguagePipeline:
    def __init__(self, text):
        self.text = text
        self.lang = self.detect_language(text)
        self._nlp = None
        self._kw_model = None

    def detect_language(self, text):
        try:
            lang = detect(text)
        except LangDetectException:
            lang = "fr"
        return lang

    @property
    def nlp(self):
        if self._nlp is None:
            if self.lang.startswith("en"):
                self._nlp = spacy.load("en_core_web_lg")
            else:
                self._nlp = spacy.load("fr_core_news_lg")
        return self._nlp

    @property
    def kw_model(self):
        if self._kw_model is None:
            if self.lang.startswith("en"):
                self._kw_model = KeyBERT(model=SentenceTransformer("allenai/scibert_scivocab_uncased"))
            else:
                self._kw_model = KeyBERT(model=SentenceTransformer("camembert-base"))
        return self._kw_model

    def lemmatize(self, word):
        return self.nlp(word)[0].lemma_.lower()

    def extract_keywords(self, keyphrase_ngram_range=(1, 1), stopwords=None, top_n=5):
        return self.kw_model.extract_keywords(
            self.text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stopwords,
            top_n=top_n
        )

# === FONCTIONS PRINCIPALES ===

def extract_keywords(text, top_n=5):
    pipeline = LanguagePipeline(text)

    # Extraction brute
    raw_keywords = pipeline.extract_keywords(
        keyphrase_ngram_range=(1, 1),
        stopwords=combined_stopwords,
        top_n=top_n * config.get("keyword_multiplier", 2)
    )

    # Lemmatisation + fréquence
    tokens = re.findall(r'\b[a-zA-Z\-]{3,}\b', text.lower())
    lemmatized_tokens = [pipeline.lemmatize(tok) for tok in tokens if tok not in combined_stopwords]
    token_freq = Counter(lemmatized_tokens)

    def is_valid_kw(kw):
        return (
            kw not in combined_stopwords and
            len(kw) > 2 and
            (kw.isalpha() or '-' in kw)
        )

    # Filtrage : unicité, lemmatisation, score pondéré
    filtered_raw, seen = [], set()
    for kw, weight in raw_keywords:
        kw_clean = kw.lower().strip()
        if is_valid_kw(kw_clean):
            kw_lemma = pipeline.lemmatize(kw_clean)
            if kw_lemma.endswith('s') and kw_lemma[:-1] in seen:
                continue
            if kw_lemma in seen:
                continue
            freq = token_freq.get(kw_lemma, 0)
            score = freq * weight
            filtered_raw.append((score, freq, kw_lemma, weight))
            seen.add(kw_lemma)

    top_filtered = heapq.nlargest(top_n, filtered_raw, key=lambda x: x[0])

    # Création objets KeywordsData
    filtered_keywords, seen = [], set()
    for score, freq, kw_lemma, weight in top_filtered:
        if kw_lemma not in seen:
            seen.add(kw_lemma)
            filtered_keywords.append(
                KeywordsData(
                    kw_lemma=kw_lemma,
                    weight=round(weight, 2),
                    freq=freq,
                    score=round(score, 2)
                )
            )

    return filtered_keywords

def format_cleaner(contenu):
    lignes = contenu.split('\n')
    lignes_nettoyees = []
    for ligne in lignes:
        ligne = ligne.strip()
        ligne_modifiee = re.sub(r"^\s*[-*\d]+[\.\)]?\s*[^:]*:\s*", "", ligne)
        lignes_nettoyees.append(ligne_modifiee)
    return "\n".join(lignes_nettoyees)

def get_relevant_context(user_question, limit=3):
    # Étape 1 : extraction des mots-clés et encodage
    keywords = extract_keywords(user_question)
    keyword_strings = [kw.kw_lemma for kw in keywords]
    query_kw_vectors = np.array(embedding_model.encode(keyword_strings, convert_to_tensor=False)).astype('float32')

    # Étape 2 : extraction des vecteurs stockés depuis SQLite
    try:
        cur.execute("SELECT conversation_id, keyword, vector FROM vectors")
        vector_rows = cur.fetchall()
    except Exception:
        return []

    if not vector_rows:
        return []

    convo_ids, stored_kw_vectors = [], []
    for conv_id, keyword, vector in vector_rows:
        try:
            vec = np.frombuffer(vector, dtype='float32')
            if vec.shape == (384,):
                stored_kw_vectors.append(vec)
                convo_ids.append(conv_id)
        except Exception:
            continue

    if not stored_kw_vectors:
        return []

    stored_kw_vectors = np.vstack(stored_kw_vectors).astype('float32')

    # Vérifications
    if any(np.isnan(arr).any() or np.isinf(arr).any() for arr in [stored_kw_vectors, query_kw_vectors]):
        return []

    # Normalisation
    faiss.normalize_L2(stored_kw_vectors)
    faiss.normalize_L2(query_kw_vectors)

    query_kw_vectors /= np.linalg.norm(query_kw_vectors, axis=1, keepdims=True)
    stored_kw_vectors /= np.linalg.norm(stored_kw_vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(query_kw_vectors, stored_kw_vectors.T)

    # Recherche top-k
    k = min(limit, stored_kw_vectors.shape[0])
    topk_indices = np.argsort(similarity_matrix, axis=1)[:, -k:][:, ::-1]
    topk_scores = np.take_along_axis(similarity_matrix, topk_indices, axis=1)

    matched_convo_ids = {
        convo_ids[idx]
        for scores_row, indices_row in zip(topk_scores, topk_indices)
        for score, idx in zip(scores_row, indices_row) if score >= config.get("similarity_threshold", 0.2)
    }

    if not matched_convo_ids:
        return []

    # Calcul score par conversation
    convo_sim_scores = {cid: [] for cid in matched_convo_ids}
    for scores_row, indices_row in zip(topk_scores, topk_indices):
        for score, idx in zip(scores_row, indices_row):
            cid = convo_ids[idx]
            if cid in convo_sim_scores and score >= config.get("similarity_threshold", 0.2):
                convo_sim_scores[cid].append(score)

    final_sim_scores = {cid: max(scores) for cid, scores in convo_sim_scores.items()}

    # Étape 3 : récupération des lignes de contexte
    placeholders_ids = ','.join(['?'] * len(matched_convo_ids))
    cur.execute(f'''
        SELECT user_input, llm_output, llm_model, timestamp, id
        FROM conversations
        WHERE id IN ({placeholders_ids})
    ''', list(matched_convo_ids))
    context_rows = cur.fetchall()

    if not context_rows:
        return []

    # Étape 4 : rerank par similarité directe avec la user_question
    user_inputs = [row[0] for row in context_rows]
    user_question_vec = embedding_model.encode([user_question], convert_to_tensor=False)
    user_question_vec = np.array(user_question_vec[0]).astype('float32')
    user_question_vec /= np.linalg.norm(user_question_vec)

    input_vectors = embedding_model.encode(user_inputs, convert_to_tensor=False)
    input_vectors = np.array(input_vectors).astype('float32')
    input_vectors /= np.linalg.norm(input_vectors, axis=1, keepdims=True)

    rerank_scores = np.dot(input_vectors, user_question_vec)

    # Construction des objets de contexte
    filtered_context = []
    for i, (user_input, llm_output, llm_model, timestamp, convo_id) in enumerate(context_rows):
        score_kw = final_sim_scores.get(convo_id, 0)
        score_rerank = rerank_scores[i]
        filtered_context.append(ContextData(
            user_input=user_input,
            llm_output=llm_output,
            score_kw=score_kw,
            llm_model=llm_model,
            timestamp=timestamp,
            convo_id=convo_id,
            score_rerank=score_rerank
        ))

    filtered_context.sort(key=lambda x: x.combined_score, reverse=True)
    return filtered_context

def summarize(text, focus_terms=None, max_length=50):
    try:
        if focus_terms:
            sentences = [
                s.strip() for s in text.split('.') 
                if any(term.lower() in s.lower() for term in focus_terms)
            ]
            filtered_text = '. '.join(sentences)
            text = filtered_text if filtered_text else text
        text = text[:2000]

        result = summarizing_pipeline(
            text,
            max_new_tokens=max_length,
            min_length=15,
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True
        )
        return result[0]['summary_text']
    except Exception as e:
        print(f"[ERREUR] summarization : {type(e).__name__} - {e}")
        return text[:max_length] + "... [résumé tronqué]"

def compress_text(text):
    pipeline = LanguagePipeline(text)
    doc = pipeline.nlp(text)

    compressed_tokens = []

    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_.lower() not in combined_stopwords
        ):
            lemma = token.lemma_.lower()
            # On filtre les tokens trop courts ou non alphanumériques
            if len(lemma) > 2 and lemma.isalpha():
                compressed_tokens.append(lemma)

    return " ".join(compressed_tokens)

def generate_prompt_paragraph(context, question, keywords=None, lang=None):
    global context_count
    if not context:
        return question

    processed_items = []
    limit = context_count_var.get() if 'context_count_var' in globals() else 5
    context_count = 0

    for idx, item in enumerate(context[:limit], start=1):
        try:
            user_input = str(item.user_input)[:300]
            llm_output = str(item.llm_output)
            cleaned_output = format_cleaner(llm_output)
            compressed_output = compress_text(cleaned_output)
            summary = summarize(compressed_output)
            processed_items.append((idx, user_input, summary))
        except Exception as e:
            print(f"Erreur traitement item : {e}")
            continue

    context_count = len(processed_items)

    if not processed_items:
        return question

    parts = []
    parts.append('<|im_start|>system')
    parts.append("Utilise les échanges précédents pour répondre à la dernière question posée de manière claire, rigoureuse et scientifique. Ne répète pas les informations déjà mentionnées. <|im_end|>" + system_prompt)
    
    for idx, q, a in processed_items:
        parts.append(f"<|im_start|>user\n{q}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")
    
    parts.append(f"<|im_start|>user\n{question} <|im_end|>")
    parts.append('<|im_start|>assistant')
    #KBogus parts.append('"""')

    return "\n".join(parts)

keyword_count_var = None
context_count_var = None

def set_gui_vars(keyword_var, context_var):
    global keyword_count_var, context_count_var
    keyword_count_var = keyword_var
    context_count_var = context_var

def on_ask(user_input, context_limit=3, keyword_count=5):
    """Fonction principale à appeler depuis un autre script
    
    Args:
        user_input (str): La question de l'utilisateur
        context_limit (int): Nombre maximum de contextes à utiliser
        keyword_count (int): Nombre de mots-clés à extraire
        
    Returns:
        str: Le prompt final généré
    """
    pipeline = LanguagePipeline(user_input)
    lang = pipeline.lang
    context = get_relevant_context(user_input, limit=context_limit)
    prompt = generate_prompt_paragraph(context, user_input, lang=lang)
    return prompt

def insert_conversation_if_new(user_input, llm_output, llm_model, keyword_count=5):
    """Insère une nouvelle conversation dans la base de données si elle n'existe pas déjà
    
    Args:
        user_input (str): La question de l'utilisateur
        llm_output (str): La réponse du modèle
        llm_model (str): Le nom du modèle utilisé
        
    Returns:
        bool: True si la conversation a été insérée, False si elle existait déjà
    """
    if cur is None:
        raise ValueError("La base de données n'est pas initialisée.")
    
    combined = user_input + llm_output
    hash_digest = hashlib.md5(combined.encode('utf-8')).hexdigest()
    cur.execute("SELECT 1 FROM hash_index WHERE hash = ?", (hash_digest,))
    if cur.fetchone():
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO conversations (user_input, llm_output, llm_model, timestamp) VALUES (?, ?, ?, ?)",
                (user_input, llm_output, llm_model, now))
    conversation_id = cur.lastrowid
    
    keywords = extract_keywords(combined, top_n=keyword_count)
    for kw in keywords:
        vector = embedding_model.encode([kw.kw_lemma], convert_to_tensor=False)[0].astype('float32')
        cur.execute("INSERT INTO vectors (conversation_id, keyword, vector) VALUES (?, ?, ?)",
                    (conversation_id, kw.kw_lemma, vector.tobytes()))
    
    cur.execute("INSERT INTO hash_index (hash) VALUES (?)", (hash_digest,))
    conn.commit()
    return True

def init_db_connection(db_path: str):
    global conn, cur
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS hash_index (hash TEXT PRIMARY KEY)''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            llm_model TEXT NOT NULL,
            llm_output TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            keyword TEXT,
            vector BLOB,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    ''')
    conn.commit()