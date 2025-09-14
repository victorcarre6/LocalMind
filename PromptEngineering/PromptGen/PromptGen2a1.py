import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import json
import os
import pyperclip
import sv_ttk  # pip install sv-ttk
from datetime import datetime

# === Fichier de sauvegarde des modèles personnalisés ===
MODELS_FILE = "user_prompt_models.json"
HISTORY_FILE = "prompt_history.json"

DEFAULT_MODELS = {
    "Personnalisé": {},
    "📧 Rédaction d'email professionnel": {
        "role": "assistant",
        "tone": "formel",
        "format": "paragraphe",
        "length": "moyen",
        "avoid": "",
        "chain_of_thought": False,
        "few_shot": False,
        "elic": False,
        "tldr": False,
        "jargonize": False,
        "humanize": False,
        "feynman": False,
        "socratic": False,
        "rewrite_person": "",
        "reverse_prompt": False,
        "self_critic": False,
        "temperature": 0.7,
        "input": "Rédige un email professionnel pour [préciser le but : demande, relance, remerciement, etc.]. Destiné à [nom ou fonction du destinataire].",
        "context": "Objet : [insérer objet]. Inclure une formule de politesse adaptée."
    },
    "💻 Génération de code": {
        "role": "expert",
        "tone": "technique",
        "format": "code",
        "length": "détaillé",
        "avoid": "fonction obsolète, code non sécurisé",
        "chain_of_thought": True,
        "few_shot": False,
        "elic": False,
        "tldr": False,
        "jargonize": True,
        "humanize": False,
        "feynman": False,
        "socratic": False,
        "rewrite_person": "",
        "reverse_prompt": False,
        "self_critic": True,
        "temperature": 0.3,
        "input": "Écris un programme en [langage] qui [décris la fonctionnalité].",
        "context": "Précise les bibliothèques, commentaires, et gestion des erreurs."
    },
    "💡 Brainstorming d'idées": {
        "role": "créatif",
        "tone": "amical",
        "format": "liste à puces",
        "length": "détaillé",
        "avoid": "",
        "chain_of_thought": False,
        "few_shot": True,
        "elic": False,
        "tldr": False,
        "jargonize": False,
        "humanize": True,
        "feynman": False,
        "socratic": True,
        "rewrite_person": "",
        "reverse_prompt": False,
        "self_critic": False,
        "temperature": 0.9,
        "input": "Propose des idées originales pour [sujet : événement, nom de produit, campagne marketing, etc.].",
        "context": "Public cible : [préciser]. Contraintes : [budget, durée, etc.]"
    },
    "📝 Résumé de texte": {
        "role": "résumeur",
        "tone": "neutre",
        "format": "liste à puces",
        "length": "court",
        "avoid": "",
        "chain_of_thought": False,
        "few_shot": False,
        "elic": False,
        "tldr": True,
        "jargonize": False,
        "humanize": False,
        "feynman": False,
        "socratic": False,
        "rewrite_person": "",
        "reverse_prompt": False,
        "self_critic": False,
        "temperature": 0.5,
        "input": "Résume le texte suivant en gardant les points clés.",
        "context": "Limite : 5 points maximum. Style concis."
    },
    "🎨 Contenu créatif": {
        "role": "créatif",
        "tone": "humoristique",
        "format": "paragraphe",
        "length": "détaillé",
        "avoid": "",
        "chain_of_thought": False,
        "few_shot": True,
        "elic": False,
        "tldr": False,
        "jargonize": False,
        "humanize": True,
        "feynman": False,
        "socratic": False,
        "rewrite_person": "",
        "reverse_prompt": False,
        "self_critic": False,
        "temperature": 0.8,
        "input": "Rédige un contenu engageant pour [type : post LinkedIn, accroche publicitaire, histoire courte, etc.].",
        "context": "Ton : humoristique ou inspirant. Public : [préciser]"
    },
    "🔍 Analyse critique": {
        "role": "critique",
        "tone": "professionnel",
        "format": "liste numérotée",
        "length": "très détaillé",
        "avoid": "",
        "chain_of_thought": True,
        "few_shot": False,
        "elic": False,
        "tldr": False,
        "jargonize": False,
        "humanize": False,
        "feynman": False,
        "socratic": True,
        "rewrite_person": "",
        "reverse_prompt": False,
        "self_critic": True,
        "temperature": 0.6,
        "input": "Analyse les forces et faiblesses de [produit, idée, stratégie, texte].",
        "context": "Objectiv : amélioration. Critères : innovation, faisabilité, impact."
    }
}

# Personnalités disponibles pour la réécriture
PERSONALITIES = [
    "",
    "Steve Jobs",
    "Albert Einstein",
    "William Shakespeare",
    "Ernest Hemingway",
    "Jane Austen",
    "Martin Luther King",
    "Winston Churchill",
    "Marie Curie",
    "Neil deGrasse Tyson",
    "David Attenborough",
    "Elon Musk",
    "Sherlock Holmes",
    "Yoda",
    "Dumbledore"
]


class PromptOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ Générateur de Prompt Optimisé ++")
        self.root.geometry("900x700")  # Taille réduite pour s'adapter aux petits écrans
        self.root.resizable(True, True)

        # Créer un cadre principal avec barres de défilement
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Créer un canvas et des barres de défilement
        self.canvas = tk.Canvas(self.main_container)
        self.v_scrollbar = ttk.Scrollbar(self.main_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.main_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Placement des éléments
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Configurer le défilement avec la molette de la souris
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # Charger les modèles personnalisés
        self.user_models = self.load_user_models()
        self.all_models = {**DEFAULT_MODELS, **self.user_models}
        
        # Charger l'historique
        self.history = self.load_history()

        # Variables
        self.template_var = tk.StringVar(value="Personnalisé")
        self.role_var = tk.StringVar(value="assistant")
        self.tone_var = tk.StringVar(value="neutre")
        self.format_var = tk.StringVar(value="paragraphe")
        self.length_var = tk.StringVar(value="moyen")
        self.avoid_var = tk.StringVar()
        self.chain_of_thought_var = tk.BooleanVar()
        self.few_shot_var = tk.BooleanVar()
        self.context_var = tk.StringVar()
        
        # Nouvelles variables
        self.elic_var = tk.BooleanVar()
        self.tldr_var = tk.BooleanVar()
        self.jargonize_var = tk.BooleanVar()
        self.humanize_var = tk.BooleanVar()
        self.feynman_var = tk.BooleanVar()
        self.socratic_var = tk.BooleanVar()
        self.rewrite_person_var = tk.StringVar(value="")
        self.reverse_prompt_var = tk.BooleanVar()
        self.self_critic_var = tk.BooleanVar()
        self.temperature_var = tk.DoubleVar(value=0.7)
        
        # Variable pour la prévisualisation en temps réel
        self.preview_update_id = None

        self.setup_ui()
        self.populate_template_combo()
        
        # Démarrer la prévisualisation
        self.schedule_preview_update()

    def _bind_mousewheel(self, event):
        """Lier la molette de la souris au défilement"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _unbind_mousewheel(self, event):
        """Délier la molette de la souris"""
        self.canvas.unbind_all("<MouseWheel>")
        
    def _on_mousewheel(self, event):
        """Gérer le défilement avec la molette de la souris"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_user_models(self):
        """Charge les modèles personnalisés depuis le fichier JSON."""
        if os.path.exists(MODELS_FILE):
            try:
                with open(MODELS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Garder uniquement les modèles utilisateur (pas les prédéfinis)
                    return {k: v for k, v in data.items() if k not in DEFAULT_MODELS}
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger les modèles personnalisés : {e}")
        return {}
    
    def load_history(self):
        """Charge l'historique des prompts depuis le fichier JSON."""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger l'historique : {e}")
        return []

    def save_user_models(self):
        """Sauvegarde tous les modèles (prédéfinis + utilisateur) dans le fichier, mais ne garde que les user."""
        try:
            # Fusionner les modèles par défaut + user (mais on ne sauve que les user)
            combined = {**DEFAULT_MODELS, **self.user_models}
            with open(MODELS_FILE, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder : {e}")
    
    def save_history(self):
        """Sauvegarde l'historique des prompts."""
        try:
            # Garder seulement les 50 derniers prompts
            if len(self.history) > 50:
                self.history = self.history[-50:]
                
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder l'historique : {e}")

    def populate_template_combo(self):
        """Met à jour la liste déroulante avec tous les modèles."""
        values = ["Personnalisé"] + sorted([k for k in self.all_models.keys() if k != "Personnalisé"])
        self.template_combo['values'] = values
        self.template_var.set("Personnalisé")

    def setup_ui(self):
        # Notebook (onglets)
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Onglet principal
        main_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(main_frame, text="Génération")
        
        # Onglet historique
        history_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(history_frame, text="Historique")
        self.setup_history_tab(history_frame)

        # === Titre et thème ===
        title = ttk.Label(main_frame, text="✨ Générateur de Prompt Optimisé ++", font=("Helvetica", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        # Bouton thème
        self.theme_btn = ttk.Button(main_frame, text="🌙 Thème Sombre", command=self.toggle_theme)
        self.theme_btn.grid(row=0, column=3, padx=10)

        # === Sélection modèle ===
        ttk.Label(main_frame, text="Modèle :").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.template_var = tk.StringVar(value="Personnalisé")
        self.template_combo = ttk.Combobox(main_frame, textvariable=self.template_var, state="readonly", width=35)
        self.template_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.template_combo.bind("<<ComboboxSelected>>", self.apply_template)

        # Boutons gestion modèles
        ttk.Button(main_frame, text="💾 Sauvegarder comme modèle", command=self.save_current_as_model).grid(row=1, column=2, padx=5)
        ttk.Button(main_frame, text="🗑️ Supprimer modèle", command=self.delete_current_model).grid(row=1, column=3, padx=5)

        # === Zone de saisie ===
        ttk.Label(main_frame, text="Requête ou idée :").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.input_text = scrolledtext.ScrolledText(main_frame, height=6, width=85)
        self.input_text.grid(row=3, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))
        self.input_text.bind("<KeyRelease>", self.schedule_preview_update)

        # === Contexte ===
        ttk.Label(main_frame, text="Contexte supplémentaire :").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.context_entry = ttk.Entry(main_frame, textvariable=self.context_var, width=85)
        self.context_entry.grid(row=5, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))
        self.context_entry.bind("<KeyRelease>", self.schedule_preview_update)

        # === Options avec Notebook interne ===
        options_notebook = ttk.Notebook(main_frame)
        options_notebook.grid(row=6, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        # Onglet Options de base
        basic_options_frame = ttk.Frame(options_notebook, padding="10")
        options_notebook.add(basic_options_frame, text="Options de base")
        self.setup_basic_options(basic_options_frame)
        
        # Onglet Techniques avancées
        advanced_options_frame = ttk.Frame(options_notebook, padding="10")
        options_notebook.add(advanced_options_frame, text="Techniques avancées")
        self.setup_advanced_options(advanced_options_frame)

        # === Contrôle de température ===
        temp_frame = ttk.Frame(main_frame)
        temp_frame.grid(row=7, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(temp_frame, text="Température (créativité) :").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(temp_frame, from_=0, to=1, variable=self.temperature_var, 
                 orient=tk.HORIZONTAL, command=self.on_temperature_change).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=10)
        self.temp_value_label = ttk.Label(temp_frame, text="0.7")
        self.temp_value_label.grid(row=0, column=2, padx=5)
        
        temp_frame.columnconfigure(1, weight=1)

        # === Boutons d'action ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=4, pady=15)

        ttk.Button(button_frame, text="✨ Générer le prompt", command=self.generate_prompt).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="📋 Copier", command=self.copy_to_clipboard).grid(row=0, column=1, padx=10)
        ttk.Button(button_frame, text="🗑️ Réinitialiser", command=self.reset_form).grid(row=0, column=2, padx=10)
        ttk.Button(button_frame, text="💾 Sauvegarder", command=self.save_to_history).grid(row=0, column=3, padx=10)

        # === Résultat ===
        ttk.Label(main_frame, text="Prompt optimisé :").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.output_text = scrolledtext.ScrolledText(main_frame, height=12, width=85, wrap=tk.WORD)
        self.output_text.grid(row=10, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Redimensionnement
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(10, weight=1)

        # Appliquer thème sombre par défaut
        sv_ttk.set_theme("dark")
    
    def setup_basic_options(self, frame):
        """Configure les options de base"""
        ttk.Label(frame, text="Rôle :").grid(row=0, column=0, sticky=tk.W, padx=5)
        role_combo = ttk.Combobox(frame, textvariable=self.role_var, state="readonly", width=18)
        role_combo['values'] = ("assistant", "expert", "tuteur", "critique", "créatif", "analyste", "résumeur", "traducteur")
        role_combo.grid(row=0, column=1, padx=5, pady=2)
        role_combo.bind("<<ComboboxSelected>>", self.schedule_preview_update)

        ttk.Label(frame, text="Tonalité :").grid(row=0, column=2, sticky=tk.W, padx=5)
        tone_combo = ttk.Combobox(frame, textvariable=self.tone_var, state="readonly", width=18)
        tone_combo['values'] = ("neutre", "formel", "amical", "professionnel", "humoristique", "technique", "sobre")
        tone_combo.grid(row=0, column=3, padx=5, pady=2)
        tone_combo.bind("<<ComboboxSelected>>", self.schedule_preview_update)

        ttk.Label(frame, text="Format :").grid(row=1, column=0, sticky=tk.W, padx=5)
        format_combo = ttk.Combobox(frame, textvariable=self.format_var, state="readonly", width=18)
        format_combo['values'] = ("paragraphe", "liste à puces", "liste numérotée", "tableau", "code", "JSON", "titre et résumé")
        format_combo.grid(row=1, column=1, padx=5, pady=2)
        format_combo.bind("<<ComboboxSelected>>", self.schedule_preview_update)

        ttk.Label(frame, text="Longueur :").grid(row=1, column=2, sticky=tk.W, padx=5)
        length_combo = ttk.Combobox(frame, textvariable=self.length_var, state="readonly", width=18)
        length_combo['values'] = ("court", "moyen", "détaillé", "très détaillé")
        length_combo.grid(row=1, column=3, padx=5, pady=2)
        length_combo.bind("<<ComboboxSelected>>", self.schedule_preview_update)

        ttk.Label(frame, text="Éviter :").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.avoid_entry = ttk.Entry(frame, textvariable=self.avoid_var, width=50)
        self.avoid_entry.grid(row=2, column=1, columnspan=3, padx=5, pady=2, sticky=tk.W)
        self.avoid_entry.bind("<KeyRelease>", self.schedule_preview_update)

        ttk.Checkbutton(frame, text="Chaîne de pensée (raisonne étape par étape)", 
                       variable=self.chain_of_thought_var, command=self.schedule_preview_update).grid(
            row=3, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame, text="Ajouter un exemple de réponse (few-shot)", 
                       variable=self.few_shot_var, command=self.schedule_preview_update).grid(
            row=4, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)
    
    def setup_advanced_options(self, frame):
        """Configure les techniques avancées"""
        ttk.Checkbutton(frame, text="ELIC (Explain Like I'm a Child - Explique comme si j'étais un enfant)", 
                       variable=self.elic_var, command=self.schedule_preview_update).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame, text="TLDR (Too Long; Didn't Read - Résumé concis)", 
                       variable=self.tldr_var, command=self.schedule_preview_update).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame, text="Jargonize (Utiliser un langage technique/spécialisé)", 
                       variable=self.jargonize_var, command=self.schedule_preview_update).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame, text="Humanize (Rendre plus humain et naturel)", 
                       variable=self.humanize_var, command=self.schedule_preview_update).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame, text="Feynman Technique (Expliquer avec des analogies simples)", 
                       variable=self.feynman_var, command=self.schedule_preview_update).grid(
            row=0, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame, text="Socratic Method (Poser des questions pour approfondir)", 
                       variable=self.socratic_var, command=self.schedule_preview_update).grid(
            row=1, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame, text="Reverse Prompt (Inverser la perspective de la requête)", 
                       variable=self.reverse_prompt_var, command=self.schedule_preview_update).grid(
            row=2, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame, text="Self-Critic (Auto-critique et amélioration)", 
                       variable=self.self_critic_var, command=self.schedule_preview_update).grid(
            row=3, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(frame, text="Réécrire comme :").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        rewrite_combo = ttk.Combobox(frame, textvariable=self.rewrite_person_var, values=PERSONALITIES, width=20)
        rewrite_combo.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        rewrite_combo.bind("<<ComboboxSelected>>", self.schedule_preview_update)
    
    def setup_history_tab(self, frame):
        """Configure l'onglet historique"""
        # Liste des prompts sauvegardés
        self.history_listbox = tk.Listbox(frame, width=100, height=20)
        self.history_listbox.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Barre de défilement
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.history_listbox.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.history_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Boutons pour l'historique
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="Charger la sélection", command=self.load_from_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Supprimer la sélection", command=self.delete_from_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Effacer tout l'historique", command=self.clear_history).pack(side=tk.LEFT, padx=5)
        
        # Configuration du redimensionnement
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # Charger l'historique
        self.refresh_history_list()

    def refresh_history_list(self):
        """Met à jour la liste de l'historique"""
        self.history_listbox.delete(0, tk.END)
        for i, item in enumerate(self.history):
            timestamp = item.get('timestamp', '')
            preview = item.get('preview', '')[:50] + '...' if len(item.get('preview', '')) > 50 else item.get('preview', '')
            self.history_listbox.insert(tk.END, f"{timestamp} - {preview}")

    def load_from_history(self):
        """Charge un prompt depuis l'historique"""
        selection = self.history_listbox.curselection()
        if not selection:
            messagebox.showwarning("Avertissement", "Veuillez sélectionner un élément de l'historique.")
            return
        
        index = selection[0]
        if 0 <= index < len(self.history):
            prompt_data = self.history[index].get('data', {})
            self.apply_prompt_data(prompt_data)
            self.notebook.select(0)  # Revenir à l'onglet de génération

    def delete_from_history(self):
        """Supprime un élément de l'historique"""
        selection = self.history_listbox.curselection()
        if not selection:
            messagebox.showwarning("Avertissement", "Veuillez sélectionner un élément à supprimer.")
            return
        
        index = selection[0]
        if 0 <= index < len(self.history):
            del self.history[index]
            self.save_history()
            self.refresh_history_list()

    def clear_history(self):
        """Efface tout l'historique"""
        if messagebox.askyesno("Confirmer", "Voulez-vous vraiment effacer tout l'historique ?"):
            self.history = []
            self.save_history()
            self.refresh_history_list()

    def apply_prompt_data(self, data):
        """Applique les données d'un prompt à l'interface"""
        self.reset_form(except_template=True)
        
        self.role_var.set(data.get("role", "assistant"))
        self.tone_var.set(data.get("tone", "neutre"))
        self.format_var.set(data.get("format", "paragraphe"))
        self.length_var.set(data.get("length", "moyen"))
        self.avoid_var.set(data.get("avoid", ""))
        self.chain_of_thought_var.set(data.get("chain_of_thought", False))
        self.few_shot_var.set(data.get("few_shot", False))
        self.elic_var.set(data.get("elic", False))
        self.tldr_var.set(data.get("tldr", False))
        self.jargonize_var.set(data.get("jargonize", False))
        self.humanize_var.set(data.get("humanize", False))
        self.feynman_var.set(data.get("feynman", False))
        self.socratic_var.set(data.get("socratic", False))
        self.rewrite_person_var.set(data.get("rewrite_person", ""))
        self.reverse_prompt_var.set(data.get("reverse_prompt", False))
        self.self_critic_var.set(data.get("self_critic", False))
        self.temperature_var.set(data.get("temperature", 0.7))
        self.temp_value_label.config(text=f"{self.temperature_var.get():.1f}")
        
        self.context_var.set(data.get("context", ""))
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, data.get("input", ""))

    def on_temperature_change(self, event=None):
        """Met à jour l'affichage de la valeur de température"""
        self.temp_value_label.config(text=f"{self.temperature_var.get():.1f}")
        self.schedule_preview_update()

    def schedule_preview_update(self, event=None):
        """Planifie la mise à jour de la prévisualisation"""
        if self.preview_update_id:
            self.root.after_cancel(self.preview_update_id)
        self.preview_update_id = self.root.after(500, self.update_preview)

    def update_preview(self):
        """Met à jour la prévisualisation du prompt"""
        base_text = self.input_text.get("1.0", tk.END).strip()
        if not base_text:
            self.output_text.delete("1.0", tk.END)
            return
        
        # Générer un aperçu (sans sauvegarde dans l'historique)
        preview_prompt = self.generate_prompt_string()
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, preview_prompt)

    def toggle_theme(self):
        current = sv_ttk.get_theme()
        if current == "dark":
            sv_ttk.use_light_theme()
            self.theme_btn.configure(text="🌙 Thème Sombre")
        else:
            sv_ttk.use_dark_theme()
            self.theme_btn.configure(text="☀️ Thème Clair")

    def apply_template(self, event=None):
        key = self.template_var.get()
        if key == "Personnalisé" or not key:
            return

        model = self.all_models.get(key, {})
        if not model:
            return

        # Appliquer les valeurs
        self.reset_form(except_template=True)

        self.role_var.set(model.get("role", "assistant"))
        self.tone_var.set(model.get("tone", "neutre"))
        self.format_var.set(model.get("format", "paragraphe"))
        self.length_var.set(model.get("length", "moyen"))
        self.avoid_var.set(model.get("avoid", ""))
        self.chain_of_thought_var.set(model.get("chain_of_thought", False))
        self.few_shot_var.set(model.get("few_shot", False))
        self.elic_var.set(model.get("elic", False))
        self.tldr_var.set(model.get("tldr", False))
        self.jargonize_var.set(model.get("jargonize", False))
        self.humanize_var.set(model.get("humanize", False))
        self.feynman_var.set(model.get("feynman", False))
        self.socratic_var.set(model.get("socratic", False))
        self.rewrite_person_var.set(model.get("rewrite_person", ""))
        self.reverse_prompt_var.set(model.get("reverse_prompt", False))
        self.self_critic_var.set(model.get("self_critic", False))
        self.temperature_var.set(model.get("temperature", 0.7))
        self.temp_value_label.config(text=f"{self.temperature_var.get():.1f}")
        
        self.context_var.set(model.get("context", ""))

        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, model.get("input", ""))
        
        # Mettre à jour la prévisualisation
        self.schedule_preview_update()

    def save_current_as_model(self):
        name = simpledialog.askstring("Sauvegarder", "Nom du modèle personnalisé :")
        if not name or name.strip() == "":
            return
        name = name.strip()

        if name in DEFAULT_MODELS:
            messagebox.showwarning("Nom réservé", "Ce nom est utilisé par un modèle par défaut.")
            return

        # Récupérer les valeurs actuelles
        model_data = {
            "role": self.role_var.get(),
            "tone": self.tone_var.get(),
            "format": self.format_var.get(),
            "length": self.length_var.get(),
            "avoid": self.avoid_var.get(),
            "chain_of_thought": self.chain_of_thought_var.get(),
            "few_shot": self.few_shot_var.get(),
            "elic": self.elic_var.get(),
            "tldr": self.tldr_var.get(),
            "jargonize": self.jargonize_var.get(),
            "humanize": self.humanize_var.get(),
            "feynman": self.feynman_var.get(),
            "socratic": self.socratic_var.get(),
            "rewrite_person": self.rewrite_person_var.get(),
            "reverse_prompt": self.reverse_prompt_var.get(),
            "self_critic": self.self_critic_var.get(),
            "temperature": self.temperature_var.get(),
            "input": self.input_text.get("1.0", tk.END).strip(),
            "context": self.context_var.get()
        }

        # Ajouter au dictionnaire utilisateur
        self.user_models[name] = model_data
        self.all_models[name] = model_data

        # Mettre à jour l'interface
        self.populate_template_combo()
        self.template_var.set(name)

        # Sauvegarder
        self.save_user_models()
        messagebox.showinfo("Sauvegardé", f"Modèle '{name}' sauvegardé avec succès !")

    def delete_current_model(self):
        name = self.template_var.get()
        if name in DEFAULT_MODELS or name == "Personnalisé":
            messagebox.showwarning("Impossible", "Vous ne pouvez pas supprimer un modèle prédéfini.")
            return

        if messagebox.askyesno("Confirmer", f"Supprimer le modèle personnalisé '{name}' ?"):
            if name in self.user_models:
                del self.user_models[name]
                del self.all_models[name]
                self.save_user_models()
                self.populate_template_combo()
                self.template_var.set("Personnalisé")
                messagebox.showinfo("Supprimé", f"Modèle '{name}' supprimé.")

    def generate_prompt_string(self):
        """Génère la chaîne de prompt sans l'afficher ni la sauvegarder (pour la prévisualisation)"""
        base_text = self.input_text.get("1.0", tk.END).strip()
        if not base_text:
            return ""

        role = self.role_var.get()
        tone = self.tone_var.get()
        format_ = self.format_var.get()
        length = self.length_var.get()
        avoid = self.avoid_var.get()
        context = self.context_var.get()

        # Construction du prompt
        prompt_parts = []

        # Ajouter le rôle
        role_map = {
            "assistant": "un assistant IA utile",
            "expert": f"un expert en {base_text.split()[0] if base_text.split() else 'ce domaine'}",
            "tuteur": "un tuteur pédagogique",
            "critique": "un critique constructif",
            "créatif": "un créatif innovant",
            "analyste": "un analyste rigoureux",
            "résumeur": "un spécialiste de la synthèse",
            "traducteur": "un traducteur expert"
        }
        role_text = role_map.get(role, role_map["assistant"])
        prompt_parts.append(f"Agis comme {role_text}.")

        # Ajouter la tonalité
        if tone != "neutre":
            prompt_parts.append(f"Utilise un ton {tone}.")

        
        # Ajouter les techniques avancées
        if self.elic_var.get():
            prompt_parts.append("Explique comme si je n'avais aucune connaissance préalable du sujet, avec des termes simples et des analogies accessibles.")
        
        if self.tldr_var.get():
            prompt_parts.append("Fournis un résumé très concis (TLDR) en plus de ta réponse complète.")
        
        if self.jargonize_var.get():
            prompt_parts.append("Utilise un langage technique et spécialisé approprié au domaine.")
        
        if self.humanize_var.get():
            prompt_parts.append("Rends ta réponse plus humaine, naturelle et conversationnelle.")
        
        if self.feynman_var.get():
            prompt_parts.append("Applique la technique Feynman : explique avec des analogies simples et des exemples concrets.")
        
        if self.socratic_var.get():
            prompt_parts.append("Applique la méthode socratique : pose des questions pertinentes pour approfondir la réflexion.")
        
        if self.rewrite_person_var.get():
            person = self.rewrite_person_var.get()
            if person:
                prompt_parts.append(f"Rédige ta réponse comme si tu étais {person}, en imitant son style et sa manière de s'exprimer.")
        
        if self.reverse_prompt_var.get():
            prompt_parts.append("Inverse la perspective : aborde le problème sous un angle opposé ou différent.")
        
        if self.self_critic_var.get():
            prompt_parts.append("Après ta réponse principale, ajoute une auto-critique pour identifier les limites ou améliorations possibles.")

        # Ajouter la structure
        format_map = {
            "paragraphe": "en paragraphes structurés",
            "liste à puces": "sous forme de liste à puces",
            "liste numérotée": "sous forme de liste numérotée",
            "tableau": "sous forme de tableau organisé",
            "code": "en fournissant du code bien commenté",
            "JSON": "en format JSON valide",
            "titre et résumé": "avec un titre clair et un résumé"
        }
        format_text = format_map.get(format_, format_map["paragraphe"])
        prompt_parts.append(f"Structure ta réponse {format_text}.")

        # Ajouter la longueur
        length_map = {
            "court": "Sois concis et va droit au but.",
            "moyen": "Fournis une réponse équilibrée avec les détails essentiels.",
            "détaillé": "Développe ta réponse avec des explications complètes.",
            "très détaillé": "Sois exhaustif et couvre tous les aspects importants."
        }
        prompt_parts.append(length_map.get(length, length_map["moyen"]))

        # Éléments à éviter
        if avoid:
            prompt_parts.append(f"Évite : {avoid}.")

        # Chaîne de pensée
        if self.chain_of_thought_var.get():
            prompt_parts.append("Montre ton raisonnement étape par étape avant de donner ta réponse finale.")

        # Few-shot learning
        if self.few_shot_var.get():
            prompt_parts.append("Inclus un exemple concret pour illustrer ta réponse.")

        # Contexte supplémentaire
        if context:
            prompt_parts.append(f"Contexte : {context}")

        # Ajouter la requête de base
        prompt_parts.append(f"Ma requête : {base_text}")

        # Ajouter la température
        temp = self.temperature_var.get()
        if temp < 0.3:
            prompt_parts.append("Sois très factuel et précis dans ta réponse.")
        elif temp > 0.7:
            prompt_parts.append("N'hésite pas à être créatif et exploratoire dans ta réponse.")

        return "\n\n".join(prompt_parts)

    def generate_prompt(self):
        prompt = self.generate_prompt_string()
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, prompt)

    def copy_to_clipboard(self):
        prompt = self.output_text.get("1.0", tk.END).strip()
        if prompt:
            pyperclip.copy(prompt)
            messagebox.showinfo("Copié", "Prompt copié dans le presse-papiers !")
        else:
            messagebox.showwarning("Vide", "Aucun prompt à copier.")

    def save_to_history(self):
        prompt = self.output_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Vide", "Aucun prompt à sauvegarder.")
            return
        
        # Récupérer les données actuelles
        prompt_data = {
            "role": self.role_var.get(),
            "tone": self.tone_var.get(),
            "format": self.format_var.get(),
            "length": self.length_var.get(),
            "avoid": self.avoid_var.get(),
            "chain_of_thought": self.chain_of_thought_var.get(),
            "few_shot": self.few_shot_var.get(),
            "elic": self.elic_var.get(),
            "tldr": self.tldr_var.get(),
            "jargonize": self.jargonize_var.get(),
            "humanize": self.humanize_var.get(),
            "feynman": self.feynman_var.get(),
            "socratic": self.socratic_var.get(),
            "rewrite_person": self.rewrite_person_var.get(),
            "reverse_prompt": self.reverse_prompt_var.get(),
            "self_critic": self.self_critic_var.get(),
            "temperature": self.temperature_var.get(),
            "input": self.input_text.get("1.0", tk.END).strip(),
            "context": self.context_var.get(),
            "prompt": prompt,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "preview": prompt[:100]  # Prévisualisation pour l'historique
        }
        
        # Ajouter à l'historique
        self.history.append(prompt_data)
        self.save_history()
        self.refresh_history_list()
        
        messagebox.showinfo("Sauvegardé", "Prompt sauvegardé dans l'historique !")

    def reset_form(self, except_template=False):
        if not except_template:
            self.template_var.set("Personnalisé")
        
        self.role_var.set("assistant")
        self.tone_var.set("neutre")
        self.format_var.set("paragraphe")
        self.length_var.set("moyen")
        self.avoid_var.set("")
        self.chain_of_thought_var.set(False)
        self.few_shot_var.set(False)
        self.elic_var.set(False)
        self.tldr_var.set(False)
        self.jargonize_var.set(False)
        self.humanize_var.set(False)
        self.feynman_var.set(False)
        self.socratic_var.set(False)
        self.rewrite_person_var.set("")
        self.reverse_prompt_var.set(False)
        self.self_critic_var.set(False)
        self.temperature_var.set(0.7)
        self.temp_value_label.config(text="0.7")
        self.context_var.set("")
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)


def main():
    root = tk.Tk()
    app = PromptOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
