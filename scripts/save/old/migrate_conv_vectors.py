import os
import sqlite3
import logging
import numpy as np
import faiss
from memorization import embedding_model, faiss_index, save_faiss_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = "datas/conversations.db"
FAISS_INDEX_PATH = "resources/faiss.index"

def main():
    logging.info("Connexion à la base de données SQLite...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    logging.info("Création de la table conversation_vectors si elle n'existe pas...")
    cur.execute('''
        CREATE TABLE IF NOT EXISTS conversation_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            vector BLOB,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    ''')
    conn.commit()

    logging.info("Récupération des conversations...")
    cur.execute("SELECT id, user_input FROM conversations")
    rows = cur.fetchall()

    vectors_to_add = []
    for convo_id, user_input in rows:
        vector = embedding_model.encode([user_input], convert_to_tensor=False)[0].astype('float32')
        cur.execute("INSERT INTO conversation_vectors (conversation_id, vector) VALUES (?, ?)",
                    (convo_id, vector.tobytes()))
        vectors_to_add.append(vector)

    conn.commit()
    logging.info(f"{len(vectors_to_add)} vecteurs insérés dans la base.")

    if vectors_to_add:
        arr = np.stack(vectors_to_add).astype('float32')
        faiss.normalize_L2(arr)
        faiss_index.add(arr)
        save_faiss_index(faiss_index)
        logging.info(f"Index FAISS mis à jour avec {len(vectors_to_add)} vecteurs.")

    conn.close()
    logging.info("Migration terminée et connexion fermée.")

if __name__ == "__main__":
    main()