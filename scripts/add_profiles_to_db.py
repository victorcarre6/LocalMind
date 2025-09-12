import sqlite3
import os
import main  # Ajout de l'import de main.py

# --- CONFIGURATION ---
# Utilisation du chemin de la base défini dans main.py
db_path = main.db_path

def apply_default_profile(db_path):
    """
    Applique le profil 'Default' à toutes les conversations existantes.
    """
    if not os.path.exists(db_path):
        print(f"[ERREUR] La base de données n'existe pas : {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Vérifier que la colonne profile_name existe
    cur.execute("PRAGMA table_info(conversations)")
    columns = [col[1] for col in cur.fetchall()]
    if "profile_name" not in columns:
        print("[INFO] La colonne 'profile_name' n'existe pas. Création de la colonne.")
        cur.execute("ALTER TABLE conversations ADD COLUMN profile_name TEXT DEFAULT 'Default'")
        conn.commit()

    # Mettre à jour toutes les conversations
    cur.execute("UPDATE conversations SET profile_name = 'Default'")
    conn.commit()

    # Retour d'information
    cur.execute("SELECT COUNT(*) FROM conversations")
    total = cur.fetchone()[0]
    print(f"[INFO] Profil 'Default' appliqué à {total} conversation(s).")

    conn.close()

if __name__ == "__main__":
    apply_default_profile(db_path)