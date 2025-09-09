import threading
import heapq
import json
import logging
import os
import re
import sqlite3
import warnings
import time
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

# --- FAISS persistant ---
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
    import time
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

# === FONCTIONS SECONDAIRES ===

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            llm_output_summary TEXT
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
    conn.commit()

def set_gui_vars(keyword_var, context_var):
    global keyword_count_var, context_count_var
    keyword_count_var = keyword_var
    context_count_var = context_var

# === FONCTIONS PRINCIPALES ===

keyword_count_var = None
context_count_var = None

def on_ask(user_input, context_limit=3, keyword_count=5, recall=True, history_limit=0):

    pipeline = LanguagePipeline(user_input)
    lang = pipeline.lang

    context = get_relevant_context(user_input, limit=context_limit, recall=recall, pipeline=pipeline)
    prompt = generate_prompt_paragraph(context, user_input, lang=lang, history_limit=history_limit)
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

def get_relevant_context(user_question, limit=3, recall=True, pipeline=None):
    global embedding_model, faiss_index, keyword_count_var
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
    return filtered_context[:limit]

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

def generate_prompt_paragraph(context, question, keywords=None, lang=None, history_limit=0):
    global context_count
    if not context:
        return f"<|im_start|>user\n{question} <|im_end|>\n<|im_start|>assistant"

    limit = context_count_var.get() if 'context_count_var' in globals() else 5
    # Préparation des données pour ANCIENNES CONVERSATIONS
    processed_items = []
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
    # Préparation des données pour DERNIERS ECHANGES
    def get_last_conversations_with_summary(limit=3):
        cur.execute("""
            SELECT user_input, llm_output_summary
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        return rows[::-1]

    last_convos = get_last_conversations_with_summary(limit=history_limit)
    processed_last_convos = []
    for idx, (user_input, llm_output_summary) in enumerate(last_convos, start=1):
        try:
            shortened_user_input = str(user_input)[:300]
            summary = llm_output_summary if llm_output_summary is not None else ""
            processed_last_convos.append((idx, shortened_user_input, summary))
        except Exception as e:
            print(f"Erreur traitement item : {e}")
            continue

    if not processed_items:
        return f"<|im_start|>user\n{question} <|im_end|>\n<|im_start|>assistant"

    parts = []
    parts.append('<|im_start|>system')
    parts.append("Réponds scientifiquement à QUESTION PRINCIPALE en utilisant le contexte fournit : priorise les DERNIERS ECHANGES, utilise les ANCIENNES CONVERSATIONS uniquement pour exemples ou contexte secondaire SI ILS ONT UN RAPPORT AVEC QUESTION PRINCIPALE. Ne répète pas les informations déjà mentionnées, surtout celles présentes dans DERNIERS ECHANGES. Rédige des réponses complètes, claires et concises. <|im_end|>" + system_prompt)
    
    # Section ANCIENNES CONVERSATIONS
    if processed_items:
        parts.append("\n### ANCIENNES CONVERSATIONS (contexte secondaire) ###")
        for idx, q, a in processed_items:
            parts.append(f"<|im_start|>user\n{q}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")

    # Section DERNIERS ECHANGES (prioritaires)
    if processed_last_convos:
        parts.append("\n### DERNIERS ECHANGES (prioritaires, par ordre chronologique) ###")
        for idx, user_input_short, last_output_summary in processed_last_convos:
            parts.append(f"<|im_start|>user\n{user_input_short}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{last_output_summary}<|im_end|>")
    parts.append("\n### QUESTION PRINCIPALE ###")
    parts.append(f"<|im_start|>user\n{question} <|im_end|>")
    parts.append('<|im_start|>assistant')

    return "\n".join(parts)

def insert_conversation_if_new(user_input, llm_output, llm_model, keyword_count=5, conversation_id=None):
    global embedding_model, faiss_index
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

    # Insertion de la conversation (pour obtenir l'ID tout de suite)
    cur.execute(
        "INSERT INTO conversations (user_input, llm_output, llm_model, timestamp, llm_output_summary) VALUES (?, ?, ?, ?, ?)",
        (user_input, llm_output, llm_model, now, None)
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