import threading
import heapq
import json
import logging
import os
import re
import sqlite3
import warnings
import tkinter as tk
from tkinter import ttk
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
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer, logging as transformers_logging, pipeline
from keybert import KeyBERT
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# === INITIALISATION ===

# --- Configuration projet & chargement du fichier config ---
PROJECT_ROOT = Path(__file__).parent.parent
config_path = PROJECT_ROOT / "resources" / "config.json"

def expand_path(value):
    """Convertit un chemin relatif en chemin absolu"""
    # Si c'est déjà un Path, on le retourne directement
    if isinstance(value, Path):
        return str(value.resolve())
    
    expanded = Path(os.path.expanduser(value))

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

db_path = config["data"]["db_path"]
if not os.path.exists(db_path):
    default_db_path = os.path.join(os.path.dirname(db_path), "conversations_example.db")
    if os.path.exists(default_db_path):
        config["db_path"] = default_db_path
    else:
        raise FileNotFoundError(f"Neither {db_path} nor {default_db_path} exist.")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# === PROFILS ===

active_profile_name = "Default"
local_profiles = ["Default", "All"]

# === INDEX VECTORIEL ET MODÈLES ===

VECTOR_DIM = 384
FAISS_INDEX_PATH = PROJECT_ROOT / "resources" / "faiss.index"

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        index = faiss.IndexHNSWFlat(VECTOR_DIM, 32)  # 32 = nb voisins connectés
        index.hnsw.efSearch = 64  # contrôle précision/vitesse
    return index

def save_faiss_index(index):
    faiss.write_index(index, str(FAISS_INDEX_PATH))

def load_models_in_background():
    global embedding_model, summarizing_pipeline, _SPACY_MODELS, _KEYBERT_MODELS, faiss_index
    try:
        # Charger le modèle d'embedding
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Charger le pipeline de résumé HuggingFace
        model_name = "Falconsai/text_summarization"
        summarizing_pipeline = pipeline(
            task="summarization",
            model=model_name,
            tokenizer=model_name,
            framework="pt",
            device=0  # mettre -1 si uniquement CPU
        )

        # Charger l'index FAISS
        faiss_index = load_faiss_index()
    except Exception as e:
        print(f"[ERREUR] lors du préchargement des modèles : {type(e).__name__} - {e}")

faiss_index = load_faiss_index()

embedding_model = None
summarizing_pipeline = None
_SPACY_MODELS = {}
_KEYBERT_MODELS = {}
faiss_index = None

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
    llm_output_summary: str
    score_kw: float
    llm_model: str
    timestamp: str
    convo_id: str
    score_rerank: float

    @property
    def combined_score(self):
        return self.score_kw * self.score_rerank

# === PIPELINE ET FONCTIONS SECONDAIRES ===

_LEMMATIZE_CACHE = {}

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
        global _SPACY_MODELS
        if self._nlp is not None:
            return self._nlp
        lang_code = "en" if self.lang.startswith("en") else "fr"
        if lang_code not in _SPACY_MODELS:
            if lang_code == "en":
                _SPACY_MODELS["en"] = spacy.load("en_core_web_lg")
            else:
                _SPACY_MODELS["fr"] = spacy.load("fr_core_news_lg")
        self._nlp = _SPACY_MODELS[lang_code]
        return self._nlp

    @property
    def kw_model(self):
        global _KEYBERT_MODELS
        if self._kw_model is not None:
            return self._kw_model
        lang_code = "en" if self.lang.startswith("en") else "fr"
        if lang_code not in _KEYBERT_MODELS:
            import time
            from sentence_transformers import SentenceTransformer
            t0 = time.time()
            if lang_code == "en":
                _KEYBERT_MODELS["en"] = KeyBERT(model=SentenceTransformer("allenai/scibert_scivocab_uncased"))
            else:
                _KEYBERT_MODELS["fr"] = KeyBERT(model=SentenceTransformer("camembert-base"))
        self._kw_model = _KEYBERT_MODELS[lang_code]
        return self._kw_model

    def lemmatize(self, word):
        key = (self.lang, word)
        if key in _LEMMATIZE_CACHE:
            return _LEMMATIZE_CACHE[key]
        lemma = self.nlp(word)[0].lemma_.lower()
        _LEMMATIZE_CACHE[key] = lemma
        return lemma

    def extract_keywords(self, keyphrase_ngram_range=(1, 1), stopwords=None, top_n=5):
        return self.kw_model.extract_keywords(
            self.text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stopwords,
            top_n=top_n
        )

def format_cleaner(contenu):
    lignes = contenu.split('\n')
    lignes_nettoyees = []
    for ligne in lignes:
        ligne = ligne.strip()
        ligne_modifiee = re.sub(r"^\s*[-*\d]+[\.\)]?\s*[^:]*:\s*", "", ligne)
        lignes_nettoyees.append(ligne_modifiee)
    return "\n".join(lignes_nettoyees)

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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            llm_output_summary TEXT,
            profile_name TEXT DEFAULT 'Default'
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
    cur.execute('''
        CREATE TABLE IF NOT EXISTS conversation_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            vector BLOB,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    ''')
    # Suppression de la création de la table profiles
    conn.commit()

def set_gui_vars(keyword_var, context_var):
    global keyword_count_var, context_count_var
    keyword_count_var = keyword_var
    context_count_var = context_var

# === FONCTIONS PRINCIPALES ===

keyword_count_var = None
context_count_var = None
keywords = None
context = None

def on_ask(user_input, context_limit=3, keyword_count=5, recall=True, history_limit=0, instant_memory=True, similarity_threshold=0.6, system_prompt=None):
    global context, prompt
    pipeline = LanguagePipeline(user_input)
    lang = pipeline.lang

    context = get_relevant_context(
        user_input,
        limit=context_limit,
        recall=recall,
        pipeline=pipeline,
        similarity_threshold=similarity_threshold
    )
    prompt = generate_prompt_paragraph(
        context,
        user_input,
        lang=lang,
        history_limit=history_limit,
        instant_memory=instant_memory,
        system_prompt=system_prompt
    )
    return prompt

def extract_keywords(text, top_n=5, pipeline=None):
    # Use existing pipeline if provided, else create one (and reuse preloaded models)
    if pipeline is None:
        pipeline = LanguagePipeline(text)

    # Extraction brute
    raw_keywords = pipeline.extract_keywords(
        keyphrase_ngram_range=(1, 1),
        stopwords=combined_stopwords,
        top_n=top_n * 2
    )

    # Lemmatisation + fréquence (avec cache et set)
    tokens = set(re.findall(r'\b[a-zA-Z\-]{3,}\b', text.lower()))
    lemmatized_tokens = [pipeline.lemmatize(tok) for tok in tokens if tok not in combined_stopwords]
    token_freq = Counter(lemmatized_tokens)

    def is_valid_kw(kw):
        return (
            kw not in combined_stopwords and
            len(kw) > 2 and
            (kw.isalpha() or '-' in kw)
        )

    # Filtrage et calcul des scores (score = freq * weight)
    seen = set()
    candidates = []
    for kw, weight in raw_keywords:
        kw_clean = kw.lower().strip()
        if not is_valid_kw(kw_clean):
            continue
        kw_lemma = pipeline.lemmatize(kw_clean)
        if kw_lemma in seen:
            continue
        freq = token_freq.get(kw_lemma, 0)
        score = freq * weight
        candidates.append((score, freq, kw_lemma, weight))
        seen.add(kw_lemma)

    # Prendre les top N en une seule passe
    top_filtered = heapq.nlargest(top_n, candidates, key=lambda x: x[0])
    return [
        KeywordsData(
            kw_lemma=kw_lemma,
            weight=round(weight, 2),
            freq=freq,
            score=round(score, 2)
        )
        for score, freq, kw_lemma, weight in top_filtered
    ]

def get_relevant_context(user_question, limit=3, recall=True, pipeline=None, similarity_threshold=0.6):
    global embedding_model, faiss_index, keyword_count_var, keywords, active_profile_name
    if not recall:
        return []

    # === 1. Embedding de la question utilisateur ===
    query_vec = embedding_model.encode([user_question], convert_to_tensor=False, batch_size=16)
    query_vec = np.array(query_vec).astype('float32')
    faiss.normalize_L2(query_vec)

    # === 2. Récupération des vecteurs stockés ===
    try:
        cur.execute("SELECT conversation_id, vector FROM conversation_vectors")
        vector_rows = cur.fetchall()
    except Exception:
        return []
    if not vector_rows:
        return []
    convo_ids, stored_vectors = [], []
    for conv_id, vector in vector_rows:
        try:
            vec = np.frombuffer(vector, dtype='float32')
            if vec.shape == (VECTOR_DIM,):
                stored_vectors.append(vec)
                convo_ids.append(conv_id)
        except Exception:
            continue
    if not stored_vectors:
        return []
    stored_vectors = np.vstack(stored_vectors).astype('float32')

    # === 3. Synchronisation & recherche FAISS ===
    if faiss_index.ntotal != len(stored_vectors):
        faiss_index.reset()
        faiss.normalize_L2(stored_vectors)
        faiss_index.add(stored_vectors)
        save_faiss_index(faiss_index)
    # Contrôle du nombre de contextes retournés par FAISS
    k = min(limit * 2, faiss_index.ntotal)
    if k == 0:
        return []
    D, I = faiss_index.search(query_vec, k)
    matched_convo_ids = {convo_ids[idx] for idx in I[0]}
    if not matched_convo_ids:
        return []
    # Scores de similarité initiale
    final_sim_scores = {convo_ids[idx]: float(score) for score, idx in zip(D[0], I[0])}

    # === 4. Extraction des mots-clés de la question ===
    keyword_count = keyword_count_var.get() if 'keyword_count_var' in globals() and keyword_count_var is not None else 5
    keywords = extract_keywords(user_question, top_n=keyword_count)
    keyword_lemmas = set(kw.kw_lemma for kw in keywords)
    kw_score_map = {kw.kw_lemma: kw.score for kw in keywords}

    # === 5. Récupération des conversations candidates ===
    placeholders_ids = ','.join(['?'] * len(matched_convo_ids))
    # Filtrer selon le profil actif sauf si "All"
    if active_profile_name and active_profile_name != "All":
        cur.execute(f'''
            SELECT user_input, llm_output, llm_model, timestamp, id, llm_output_summary
            FROM conversations
            WHERE id IN ({placeholders_ids}) AND profile_name = ?
        ''', list(matched_convo_ids) + [active_profile_name])
    else:
        cur.execute(f'''
            SELECT user_input, llm_output, llm_model, timestamp, id, llm_output_summary
            FROM conversations
            WHERE id IN ({placeholders_ids})
        ''', list(matched_convo_ids))
    context_rows = cur.fetchall()
    if not context_rows:
        return []

    # === 6. Reranking des contextes ===
    filtered_context = []
    for (user_input, llm_output, llm_model, timestamp, convo_id, llm_output_summary) in context_rows:
        context_text = (user_input or "") + " " + (llm_output or "")
        context_tokens = set(re.findall(r'\b[a-zA-Z\-]{3,}\b', context_text.lower()))

        # Utiliser le pipeline fourni
        if pipeline is None:
            pipeline = LanguagePipeline(user_question)
        context_lemmas = set(pipeline.lemmatize(tok) for tok in context_tokens)

        match_lemmas = keyword_lemmas.intersection(context_lemmas)
        kw_match_score = sum(kw_score_map[lem] for lem in match_lemmas if lem in kw_score_map)
        sim_score = final_sim_scores.get(convo_id, 0.0)

        score_kw = sim_score * 0.7 + kw_match_score * 0.3
        score_rerank = 1.0 + (len(match_lemmas) / (len(keyword_lemmas) + 1e-6))

        filtered_context.append(ContextData(
            user_input=user_input,
            llm_output=llm_output,
            llm_output_summary=llm_output_summary,
            score_kw=score_kw,
            llm_model=llm_model,
            timestamp=timestamp,
            convo_id=convo_id,
            score_rerank=score_rerank
        ))

    # === 7. Tri final par score combiné ===
    filtered_context.sort(key=lambda x: x.combined_score, reverse=True)
    filtered_context = [ctx for ctx in filtered_context if ctx.combined_score >= similarity_threshold]
    return filtered_context[:limit]

def get_last_conversations_with_summary(limit=3):
    cur.execute("""
        SELECT user_input, llm_output_summary
        FROM conversations
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    return rows[::-1]

def summarize(text, focus_terms=None, max_length=50):
    global summarizing_pipeline
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
        summary = result[0]['summary_text']
        return summary
    except Exception as e:
        print(f"[ERREUR] summarization : {type(e).__name__} - {e}")
        summary = text[:max_length] + "... [résumé tronqué]"
        return summary

def generate_prompt_paragraph(context, question, keywords=None, lang=None, history_limit=0, instant_memory=True, system_prompt=None):
    global context_count

    limit = context_count_var.get() if 'context_count_var' in globals() and context_count_var is not None else 5
    # Récupération des contextes long-term (anciennes conversations)
    processed_items = []
    if context:
        for idx, item in enumerate(context[:limit], start=1):
            try:
                user_input = str(item.user_input)[:300]
                summary = getattr(item, "llm_output_summary", None)
                if summary is None:
                    summary = ""
                processed_items.append((idx, user_input, summary))
            except Exception as e:
                print(f"Erreur traitement item : {e}")
                continue
    context_count = len(processed_items)

    # Récupération des derniers échanges (short-term memory) uniquement si non-éphémère
    processed_last_convos = []
    if instant_memory:
        last_convos = get_last_conversations_with_summary(limit=history_limit)
        for idx, (user_input, llm_output_summary) in enumerate(last_convos, start=1):
            try:
                shortened_user_input = str(user_input)[:300]
                summary = llm_output_summary if llm_output_summary is not None else ""
                processed_last_convos.append((idx, shortened_user_input, summary))
            except Exception as e:
                print(f"Erreur traitement item : {e}")
                continue

    has_long = bool(processed_items)
    has_short = bool(processed_last_convos)
    parts = []

    if not has_long and not has_short:
        # 1. Ni short-term ni long-term
        sys_text = (
            f"{system_prompt}\n"
            "Very important: answer in the same language as used in the MAIN QUESTION."
        )
        parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>")
    elif has_long and not has_short:
        # 2. Long-term seulement
        sys_text = (
            "You are a scientific assistant. Use the provided PAST CONVERSATIONS as secondary context only if they are relevant to the MAIN QUESTION. "
            "Do not repeat information already mentioned. Write clear, concise, and complete answers.\n"
            f"{system_prompt}\n"
            "Remember: The provided PAST CONVERSATIONS are only for secondary context or examples. Prioritize answering the MAIN QUESTION."
        )
        parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>")
    elif not has_long and has_short:
        # 3. Short-term seulement
        sys_text = (
            "You are a scientific assistant. Use the provided RECENT EXCHANGES (chronological order, most recent last) as your main context. "
            "Do not repeat information already present in the RECENT EXCHANGES. Write clear, concise, and complete answers.\n"
            f"{system_prompt}\n"
            "Remember: Focus on the RECENT EXCHANGES for context. Answer the MAIN QUESTION accordingly."
        )
        parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>")
    else:
        # 4. Les deux présents
        sys_text = (
            "You are a scientific assistant. Use the RECENT EXCHANGES (main context, most recent last) as your primary source, "
            "and use PAST CONVERSATIONS only for secondary context or examples IF they are relevant to the MAIN QUESTION. "
            "Do not repeat information already present in the RECENT EXCHANGES. Write clear, concise, and complete answers.\n"
            f"{system_prompt}\n"
            "Remember: Prioritize the RECENT EXCHANGES, and only use PAST CONVERSATIONS for additional, relevant context."
        )
        parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>")

    if has_long:
        parts.append("\n### PAST CONVERSATIONS (secondary context) ###")
        for idx, q, a in processed_items:
            parts.append(f"<|im_start|>user\n{q}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")
    if has_short:
        parts.append("\n### RECENT EXCHANGES (main context, chronological order) ###")
        for idx, user_input_short, last_output_summary in processed_last_convos:
            parts.append(f"<|im_start|>user\n{user_input_short}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{last_output_summary}<|im_end|>")
    # Always add the main question at the end
    parts.append("\n### MAIN QUESTION ###")
    parts.append(f"<|im_start|>user\n{question} <|im_end|>")
    parts.append('<|im_start|>assistant')
    return "\n".join(parts)

def insert_conversation_if_new(user_input, llm_output, llm_model, keyword_count=5, conversation_id=None):
    global embedding_model, faiss_index, active_profile_name
    if cur is None:
        raise ValueError("La base de données n'est pas initialisée.")

    combined = user_input + llm_output
    hash_digest = hashlib.md5(combined.encode('utf-8')).hexdigest()
    cur.execute("SELECT 1 FROM hash_index WHERE hash = ?", (hash_digest,))
    if cur.fetchone():
        return False

    cleaned_output = format_cleaner(llm_output)
    compressed_output = compress_text(cleaned_output)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Embedding du user_input
    vector = embedding_model.encode([user_input], convert_to_tensor=False)[0].astype('float32')
    faiss.normalize_L2(vector.reshape(1, -1))

    # Utiliser "Default" si le profil actif est "All", None ou vide
    profile_for_insert = active_profile_name
    if not profile_for_insert or profile_for_insert == "All":
        profile_for_insert = "Default"
    # Insertion de la conversation (avec profil)
    cur.execute(
        "INSERT INTO conversations (user_input, llm_output, llm_model, timestamp, llm_output_summary, profile_name) VALUES (?, ?, ?, ?, ?, ?)",
        (user_input, llm_output, llm_model, now, None, profile_for_insert)
    )
    conversation_id = cur.lastrowid

    # Insertion du hash
    cur.execute("INSERT INTO hash_index (hash) VALUES (?)", (hash_digest,))

    # Insertion du vecteur principal dans conversation_vectors (batch possible si plusieurs)
    cur.execute("INSERT INTO conversation_vectors (conversation_id, vector) VALUES (?, ?)", (conversation_id, vector.tobytes()))

    # Ajout du user_input dans l’index FAISS (principal)
    faiss_index.add(vector.reshape(1, -1))
    save_faiss_index(faiss_index)

    # Gestion des mots-clés et embeddings associés en thread séparé
    def process_keywords_and_vectors(conversation_id_arg):
        # Nouvelle connexion SQLite locale pour ce thread
        thread_conn = sqlite3.connect(db_path)
        thread_cur = thread_conn.cursor()
        try:
            keywords = extract_keywords(combined, top_n=keyword_count)
            kw_lemmas = [kw.kw_lemma for kw in keywords]
            if not kw_lemmas:
                thread_conn.close()
                return
            kw_vectors = embedding_model.encode(kw_lemmas, convert_to_tensor=False)
            kw_vectors = np.asarray(kw_vectors, dtype='float32')
            faiss.normalize_L2(kw_vectors)
            kw_sql_data = [(conversation_id_arg, kw.kw_lemma, vec.tobytes()) for kw, vec in zip(keywords, kw_vectors)]
            thread_cur.executemany("INSERT INTO vectors (conversation_id, keyword, vector) VALUES (?, ?, ?)", kw_sql_data)
            thread_conn.commit()
            # Ajout au FAISS index principal (thread-safe pour l'instant, mais attention si multi-thread)
            if kw_vectors.shape[0] > 0:
                faiss_index.add(kw_vectors)
                save_faiss_index(faiss_index)
        finally:
            thread_conn.close()

    kw_thread = threading.Thread(target=process_keywords_and_vectors, args=(conversation_id,))
    kw_thread.start()

    # Résumé en thread séparé (ThreadPoolExecutor)
    output_summary = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_summary = executor.submit(summarize, compressed_output) if compressed_output else None
        if future_summary:
            output_summary = future_summary.result()
        else:
            output_summary = ""

    cur.execute("UPDATE conversations SET llm_output_summary = ? WHERE id = ?", (output_summary, conversation_id))
    conn.commit()

    kw_thread.join()

    return True


# === PROFILS : Gestion via base de données ===

def get_all_profiles():
    """Retourne la liste fusionnée des profils locaux et de ceux présents dans la base."""
    global local_profiles
    cur.execute("SELECT DISTINCT profile_name FROM conversations")
    db_profiles = [row[0] for row in cur.fetchall() if row[0]]

    all_profiles = set(local_profiles) | set(db_profiles)

    # Toujours forcer l’ordre avec Default et All en tête
    ordered = ["Default", "All"] + sorted(p for p in all_profiles if p not in ("Default", "All"))
    return ordered

def add_profile(name: str):
    """Ajoute un nouveau profil dans la liste locale s’il n’existe pas déjà."""
    global local_profiles
    name = name.strip()
    if not name or name in ("Default", "All"):
        return
    if name not in local_profiles:
        local_profiles.append(name)

def edit_profile(old_name: str, new_name: str):
    """Renomme un profil dans la liste locale et dans la base de données."""
    global local_profiles
    if old_name in ("Default", "All") or old_name not in get_all_profiles():
        raise ValueError("Profil invalide.")
    if new_name in ("Default", "All") or new_name in get_all_profiles():
        raise ValueError("Nom de profil déjà utilisé.")

    # Mettre à jour la liste locale
    if old_name in local_profiles:
        local_profiles[local_profiles.index(old_name)] = new_name

    # Mettre à jour la base
    cur.execute("UPDATE conversations SET profile_name = ? WHERE profile_name = ?", (new_name, old_name))
    conn.commit()

def delete_profile(name: str):
    """
    Supprime un profil et toutes les conversations associées dans la base.
    Si le profil est "Default" ou "All", ne supprime que les conversations associées,
    mais laisse le profil dans la liste locale.
    """
    global local_profiles
    all_profiles = get_all_profiles()
    if name not in all_profiles:
        raise ValueError("Profil inexistant.")

    # Supprime seulement les conversations, pas le profil de la liste
    if name in ("Default", "All"):
        cur.execute("SELECT id FROM conversations WHERE profile_name = ?", (name,))
        convo_ids = [row[0] for row in cur.fetchall()]
        if convo_ids:
            placeholders = ",".join("?" for _ in convo_ids)
            cur.execute(f"DELETE FROM vectors WHERE conversation_id IN ({placeholders})", convo_ids)
            cur.execute(f"DELETE FROM conversation_vectors WHERE conversation_id IN ({placeholders})", convo_ids)
            cur.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", convo_ids)
        conn.commit()
        # Ne retire pas le profil de local_profiles
        return

    # Sinon, suppression normale (conversations + profil local)
    if name in local_profiles:
        local_profiles.remove(name)

    cur.execute("SELECT id FROM conversations WHERE profile_name = ?", (name,))
    convo_ids = [row[0] for row in cur.fetchall()]
    if convo_ids:
        placeholders = ",".join("?" for _ in convo_ids)
        cur.execute(f"DELETE FROM vectors WHERE conversation_id IN ({placeholders})", convo_ids)
        cur.execute(f"DELETE FROM conversation_vectors WHERE conversation_id IN ({placeholders})", convo_ids)
        cur.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", convo_ids)
    conn.commit()

# === FONCTIONS TERTIAIRES ===

def show_infos(keywords_list=None, context_list=None):
    global keywords, context, prompt
    if keywords_list is None:
        keywords_list = keywords
    if context_list is None:
        context_list = context

    info_window = tk.Toplevel()
    info_window.title("Information about the generated prompt and the database")
    info_window.geometry("800x750")
    info_window.configure(bg="#323232")
    info_window.transient(None)
    info_window.grab_set()

    # --- Extract prompt_keywords ONCE for reuse ---
    try:
        prompt_keywords = extract_keywords(prompt, top_n=15)
    except Exception:
        prompt_keywords = []

    notebook = ttk.Notebook(info_window, style="TNotebook")
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # --- Keywords Tab ---
    tab_keywords = ttk.Frame(notebook, style="TFrame")
    notebook.add(tab_keywords, text="Keywords")

    def plot_keywords_bar(ax, kw_list, title):
        sorted_kw = sorted(kw_list, key=lambda kw: kw.score, reverse=True)
        kw_lemmas = [kw.kw_lemma for kw in sorted_kw]
        freqs = [kw.freq for kw in sorted_kw]
        weights = [kw.weight for kw in sorted_kw]
        scores = [kw.score for kw in sorted_kw]
        bar_width = 0.25
        x = list(range(len(kw_lemmas)))
        ax.bar([i-bar_width for i in x], freqs, width=bar_width, label="Frequency", color="#599258")
        ax.bar(x, weights, width=bar_width, label="Weight", color="#a2d149")
        ax.bar([i+bar_width for i in x], scores, width=bar_width, label="Score", color="#e08c26")
        ax.set_facecolor("#323232")
        ax.set_xticks(x)
        ax.set_xticklabels(kw_lemmas, rotation=45, ha='right', color="white", fontsize=9)
        ax.tick_params(axis='y', colors="white", labelsize=10)
        ax.set_title(title, color="white", fontsize=9, fontweight='bold')
        ax.legend(facecolor="#323232", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_color('white')
        return ax

    if keywords_list and isinstance(keywords_list[0], KeywordsData):
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100, facecolor="#323232")
        plot_keywords_bar(ax, keywords_list, "Frequency, weight, and score from initial prompt keywords")
        fig.tight_layout()
        canvas1 = FigureCanvasTkAgg(fig, master=tab_keywords)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))
        plt.close(fig)
        if prompt_keywords and isinstance(prompt_keywords[0], KeywordsData):
            fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=100, facecolor="#323232")
            plot_keywords_bar(ax2, prompt_keywords, "Frequency, weight, and score from generated prompt keywords")
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=tab_keywords)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(20, 0))
            plt.close(fig2)
    else:
        tk.Label(tab_keywords, text="No keywords available", fg="white", bg="#323232").pack(pady=20)

    # --- Contexts Tab ---
    tab_contexts = ttk.Frame(notebook, style="TFrame")
    # tab_contexts.configure(bg="#323232")  # Removed: ttk.Frame does not support 'bg'
    notebook.add(tab_contexts, text="Contexts")
    # Use a single scrollable Text widget for the entire contexts tab
    text_frame = tk.Frame(tab_contexts, bg="#323232")
    text_frame.pack(fill="both", expand=True)
    scrollb = tk.Scrollbar(text_frame)
    scrollb.pack(side="right", fill="y")
    text_widget = tk.Text(
        text_frame,
        width=90,
        height=38,
        wrap="word",
        bg="#323232",
        fg="white",
        font=("Segoe UI", 11),
        bd=0,
        padx=4,
        pady=2,
        yscrollcommand=scrollb.set
    )
    scrollb.config(command=text_widget.yview)
    # Configure tags for styling
    text_widget.tag_configure("user_label", foreground="#599258", font=("Segoe UI", 11, "bold"))
    text_widget.tag_configure("user_input", foreground="#599258", font=("Segoe UI", 11, "bold"))
    text_widget.tag_configure("assistant_label", foreground="#CECABF", font=("Segoe UI", 11, "bold"))
    text_widget.tag_configure("assistant_output", foreground="#CECABF", font=("Segoe UI", 11))
    text_widget.tag_configure("score", foreground="#599258", font=("Segoe UI", 10))
    if context_list:
        for idx, ctx in enumerate(context_list, 1):
            user_input = getattr(ctx, "user_input", "")
            llm_output = getattr(ctx, "llm_output", "")
            combined_score = getattr(ctx, "combined_score", 0)
            text_widget.insert(tk.END, f"{idx}. ", ("score",))
            text_widget.insert(tk.END, "User: ", ("user_label",))
            text_widget.insert(tk.END, user_input.strip() + "\n", ("user_input",))
            text_widget.insert(tk.END, "Assistant: ", ("assistant_label",))
            text_widget.insert(tk.END, llm_output.strip() + "\n", ("assistant_output",))
            text_widget.insert(tk.END, f"Score: {combined_score:.2f}\n\n", ("score",))
    else:
        text_widget.insert(tk.END, "No contexts available", ("score",))
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(side="left", fill="both", expand=True)

    # --- Heatmap Tab ---
    heatmap_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(heatmap_tab, text="Heatmap Correlation")
    if prompt_keywords and isinstance(prompt_keywords[0], KeywordsData):
        kw_lemmas = [kw.kw_lemma for kw in prompt_keywords]
        embeddings = embedding_model.encode(kw_lemmas)
        sim_matrix = cosine_similarity(embeddings)
        fig_hm, ax_hm = plt.subplots(figsize=(10, 10), dpi=100)
        heatmap = sns.heatmap(
            sim_matrix, xticklabels=kw_lemmas, yticklabels=kw_lemmas,
            cmap="coolwarm", annot=False, ax=ax_hm, vmax=0.6, color="white",
            cbar_kws={'label': 'Similarity', 'shrink': 0.65, 'aspect': 20}, square=True
        )
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Similarity', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')
        ax_hm.set_title("Keywords semantic similarity", color="white", fontsize=10, fontweight='bold')
        ax_hm.tick_params(axis='x', colors="white", labelsize=8, pad=20)
        ax_hm.tick_params(axis='y', colors="white", labelsize=8)
        fig_hm.patch.set_facecolor("#323232")
        ax_hm.set_facecolor("#323232")
        plt.tight_layout(pad=3)
        fig_hm.subplots_adjust(left=0.25, right=0.95, bottom=0.25, top=1)
        container = tk.Frame(heatmap_tab, bg="#323232")
        container.pack(fill='both', expand=True)
        top_right_frame = tk.Frame(container, bg="#323232")
        top_right_frame.pack(anchor='ne', expand=True, padx=20, pady=20)
        canvas_hm = FigureCanvasTkAgg(fig_hm, master=top_right_frame)
        canvas_hm.draw()
        canvas_hm.get_tk_widget().pack()
        plt.close(fig_hm)
    else:
        tk.Label(heatmap_tab, text="No valid data to display heatmap.", fg="white", bg="#323232").pack(pady=20)
    plt.close('all')

    # --- Database Tab ---
    stats_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(stats_tab, text="Database")
    frame_stats = tk.Frame(stats_tab, bg="#323232")
    frame_stats.pack(fill="both", expand=True, padx=20, pady=20)
    conn = sqlite3.connect(config["data"]["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM conversations")
    nb_conversations = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT keyword) FROM vectors")
    nb_mots_clefs_uniques = cur.fetchone()[0]
    cur.execute("""
        SELECT keyword, COUNT(*) as freq
        FROM vectors
        GROUP BY keyword
        ORDER BY freq DESC
        LIMIT 20
    """)
    top_keywords = cur.fetchall()
    cur.execute("SELECT llm_model, COUNT(*) FROM conversations GROUP BY llm_model")
    model_counts = cur.fetchall()
    conn.close()
    tk.Label(frame_stats, text="Overall database statistics", fg="white", bg="#323232", font=("Segoe UI", 12, "bold")).pack(pady=(0, 20))
    tk.Label(frame_stats, text=f"Number of conversations : {nb_conversations}", fg="white", bg="#323232", font=("Segoe UI", 12)).pack(anchor="w", pady=2)
    tk.Label(frame_stats, text=f"Number of keywords : {nb_mots_clefs_uniques}", fg="white", bg="#323232", font=("Segoe UI", 12)).pack(anchor="w", pady=2)
    two_col_frame = tk.Frame(frame_stats, bg="#323232")
    two_col_frame.pack(expand=True, fill="both", pady=20)
    titles_frame = tk.Frame(two_col_frame, bg="#323232")
    titles_frame.pack(fill="x", padx=(0, 30), pady=(0, 5))
    header_font = ("Segoe UI", 10, "bold")
    tk.Label(titles_frame, text="Most frequent keywords", font=header_font, fg="white", bg="#323232", anchor="w").pack(side="left", padx=(0,180))
    tk.Label(titles_frame, text="Conversations by LLM models", font=header_font, fg="white", bg="#323232", anchor="w").pack(side="left", expand=True, pady=0)
    table_frame = tk.Frame(two_col_frame, bg="#323232")
    table_frame.pack(side="left", fill="y", padx=(0, 30))
    tk.Label(table_frame, text="Keywords", fg="white", bg="#323232", font=header_font, width=20, anchor="w").grid(row=0, column=0, sticky="w", padx=3, pady=2)
    tk.Label(table_frame, text="Frequency", fg="white", bg="#323232", font=header_font, width=10, anchor="w").grid(row=0, column=1, sticky="w", padx=3, pady=2)
    data_font = ("Segoe UI", 10)
    for i, (keyword, freq) in enumerate(top_keywords, start=1):
        tk.Label(table_frame, text=keyword, fg="white", bg="#323232", font=data_font, anchor="w", width=20).grid(row=i, column=0, sticky="w", padx=3, pady=1)
        tk.Label(table_frame, text=str(freq), fg="white", bg="#323232", font=data_font, anchor="w", width=10).grid(row=i, column=1, sticky="w", padx=3, pady=1)
    graph_frame = tk.Frame(two_col_frame, bg="#323232")
    graph_frame.pack(side="left", expand=True, fill="both", padx=(0, 10))
    models = [row[0] for row in model_counts]
    counts = [row[1] for row in model_counts]
    fig_model, ax_model = plt.subplots(figsize=(5, 4.5), dpi=100, facecolor='none')
    bars = ax_model.bar(models, counts, color="#599258", width=0.5)
    ax_model.set_facecolor("#323232")
    for spine in ['bottom', 'left', 'top', 'right']:
        ax_model.spines[spine].set_visible(True)
        ax_model.spines[spine].set_color('white')
    ax_model.set_xticks([])
    ax_model.tick_params(axis='y', which='both', colors='white', labelsize=8)
    ax_model.set_ylabel("Count", color="white", fontsize=9)
    ax_model.set_xlabel("LLM Models", color="white", fontsize=9)
    plt.tight_layout()
    fig_model.patch.set_facecolor("#323232")
    for bar, model in zip(bars, models):
        height = bar.get_height()
        ax_model.text(bar.get_x() + bar.get_width()/2, height/2, model, rotation=90, ha='center', va='bottom', color='white', fontsize=8)
    canvas_model = FigureCanvasTkAgg(fig_model, master=graph_frame)
    canvas_model.draw()
    canvas_model.get_tk_widget().pack(expand=True, fill="both", pady=(0, 0))
    plt.close(fig_model)