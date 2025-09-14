import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
import json
import os
from datetime import datetime

class GGCodeGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Générateur de Prompt pour Code Source - Pro++")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)
        
        # Variables pour les nouvelles fonctionnalités
        self.prompt_templates = {}
        self.profiles = {}
        self.tags = set()
        self.history = []
        self.max_history = 10
        
        # Chargement des données externes
        self.load_external_data()
        
        self.setup_ui()
        
    def load_external_data(self):
        """Charge les templates, profils et tags depuis des fichiers JSON externes"""
        # Chargement des templates
        try:
            if os.path.exists("templates.json"):
                with open("templates.json", "r", encoding="utf-8") as f:
                    self.prompt_templates = json.load(f)
            else:
                # Templates par défaut
                self.prompt_templates = {
                    "Python Basique": {
                        "langage": "Python",
                        "fonctionnalites": "Un programme simple qui affiche 'Hello World'",
                        "contraintes": "Utiliser Python 3.x",
                        "bibliotheques": "",
                        "niveau": "basique",
                        "paradigmes": [],
                        "comments": True,
                        "tests": False,
                        "docs": False,
                        "style": "Standard",
                        "tags": ["python", "débutant"]
                    }
                }
        except Exception as e:
            print(f"Erreur lors du chargement des templates: {e}")
            self.prompt_templates = {}
        
        # Chargement des profils
        try:
            if os.path.exists("profiles.json"):
                with open("profiles.json", "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
            else:
                # Profils par défaut
                self.profiles = {
                    "ChatGPT": {
                        "prefix": "En tant qu'expert en développement, ",
                        "suffix": " Assure-toi que le code est propre, efficace et bien documenté.",
                        "max_tokens": 1500,
                        "temperature": 0.7
                    },
                    "Claude": {
                        "prefix": "En tant que développeur expérimenté, ",
                        "suffix": " Le code doit être concis mais complet, avec une bonne structure.",
                        "max_tokens": 2000,
                        "temperature": 0.5
                    },
                    "Copilot": {
                        "prefix": "",
                        "suffix": " Fournis uniquement le code sans explications supplémentaires.",
                        "max_tokens": 1000,
                        "temperature": 0.3
                    }
                }
        except Exception as e:
            print(f"Erreur lors du chargement des profils: {e}")
            self.profiles = {
                "ChatGPT": {
                    "prefix": "En tant qu'expert en développement, ",
                    "suffix": " Assure-toi que le code est propre, efficace et bien documenté.",
                    "max_tokens": 1500,
                    "temperature": 0.7
                }
            }
        
        # Extraction des tags depuis les templates
        self.extract_tags_from_templates()
    
    def extract_tags_from_templates(self):
        """Extrait tous les tags des templates pour les proposer dans l'interface"""
        self.tags = set()
        for template in self.prompt_templates.values():
            if "tags" in template:
                self.tags.update(template["tags"])
    
    def save_external_data(self):
        """Sauvegarde les templates, profils et tags dans des fichiers JSON externes"""
        # Sauvegarde des templates
        try:
            with open("templates.json", "w", encoding="utf-8") as f:
                json.dump(self.prompt_templates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder les templates: {str(e)}")
        
        # Sauvegarde des profils
        try:
            with open("profiles.json", "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder les profils: {str(e)}")
    
    def setup_ui(self):
        # Notebook (onglets)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Onglet principal - Génération de prompt
        self.main_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.main_frame, text="Génération")
        
        # Onglet historique
        self.history_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.history_frame, text="Historique")
        
        # Onglet templates
        self.templates_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.templates_frame, text="Templates")
        
        # Onglet profils
        self.profiles_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.profiles_frame, text="Profils")
        
        # Configuration de l'onglet principal
        self.setup_main_tab()
        
        # Configuration de l'onglet historique
        self.setup_history_tab()
        
        # Configuration de l'onglet templates
        self.setup_templates_tab()
        
        # Configuration de l'onglet profils
        self.setup_profiles_tab()
    
    def setup_main_tab(self):
        # Canvas et Scrollbar pour l'onglet principal
        canvas = tk.Canvas(self.main_frame)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Permettre le défilement avec la molette
        self.bind_mousewheel(canvas)
        
        # Titre
        label_titre = tk.Label(
            scrollable_frame,
            text="Générateur de Prompt pour Code Source - Pro++",
            font=("Helvetica", 16, "bold")
        )
        label_titre.pack(pady=15)
        
        # Frame pour les sélections rapides (template, profil, tags)
        quick_select_frame = tk.Frame(scrollable_frame)
        quick_select_frame.pack(pady=5, fill="x", padx=20)
        
        # Template selection
        frame_template = tk.Frame(quick_select_frame)
        frame_template.pack(pady=5, fill="x")
        tk.Label(frame_template, text="Template de prompt :").pack(anchor="w")
        
        template_frame = tk.Frame(frame_template)
        template_frame.pack(fill="x")
        
        self.template_var = tk.StringVar()
        self.template_combo = ttk.Combobox(
            template_frame,
            textvariable=self.template_var,
            values=list(self.prompt_templates.keys()),
            state="readonly"
        )
        self.template_combo.pack(side="left", fill="x", expand=True)
        self.template_combo.bind("<<ComboboxSelected>>", self.apply_template)
        
        tk.Button(
            template_frame, 
            text="Sauvegarder comme template", 
            command=self.save_as_template
        ).pack(side="right", padx=(5, 0))
        
        # Profil selection
        frame_profil = tk.Frame(quick_select_frame)
        frame_profil.pack(pady=5, fill="x")
        tk.Label(frame_profil, text="Profil de sortie :").pack(anchor="w")
        
        self.profile_var = tk.StringVar(value="ChatGPT")
        profile_combo = ttk.Combobox(
            frame_profil,
            textvariable=self.profile_var,
            values=list(self.profiles.keys()),
            state="readonly"
        )
        profile_combo.pack(fill="x")
        
        # Tags selection
        frame_tags = tk.Frame(quick_select_frame)
        frame_tags.pack(pady=5, fill="x")
        tk.Label(frame_tags, text="Tags (optionnel) :").pack(anchor="w")
        
        self.tags_var = tk.StringVar()
        self.tags_entry = ttk.Combobox(
            frame_tags,
            textvariable=self.tags_var,
            values=list(self.tags),
            state="normal"
        )
        self.tags_entry.pack(fill="x")
        self.tags_entry.bind("<KeyRelease>", self.update_tags_suggestions)
        
        # Langage
        frame_langage = tk.Frame(scrollable_frame)
        frame_langage.pack(pady=5, fill="x", padx=20)
        tk.Label(frame_langage, text="Langage de programmation cible :").pack(anchor="w")
        self.combo_langage = ttk.Combobox(
            frame_langage,
            values=[
                "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "TypeScript",
                "PHP", "Swift", "Kotlin", "Ruby", "Dart", "SQL", "HTML/CSS", "Autre"
            ],
            state="readonly"
        )
        self.combo_langage.pack(fill="x")
        self.combo_langage.set("Python")
        
        # Paradigmes de programmation
        frame_paradigmes = tk.Frame(scrollable_frame)
        frame_paradigmes.pack(pady=5, fill="x", padx=20)
        tk.Label(frame_paradigmes, text="Paradigmes de programmation (optionnel) :").pack(anchor="w")
        
        self.paradigme_vars = {
            "OOP": tk.BooleanVar(),
            "Fonctionnel": tk.BooleanVar(),
            "Impératif": tk.BooleanVar(),
            "Réactif": tk.BooleanVar()
        }
        
        paradigmes_frame = tk.Frame(frame_paradigmes)
        paradigmes_frame.pack(fill="x")
        
        for i, (text, var) in enumerate(self.paradigme_vars.items()):
            cb = tk.Checkbutton(paradigmes_frame, text=text, variable=var)
            cb.pack(side="left", padx=(0, 10))
        
        # Fonctionnalités
        frame_fonc = tk.Frame(scrollable_frame)
        frame_fonc.pack(pady=10, fill="both", expand=True, padx=20)
        tk.Label(frame_fonc, text="Fonctionnalités souhaitées :").pack(anchor="w")
        self.texte_fonctionnalites = scrolledtext.ScrolledText(frame_fonc, height=6, wrap=tk.WORD)
        self.texte_fonctionnalites.pack(fill="both", expand=True)
        
        # Bibliothèques
        frame_bib = tk.Frame(scrollable_frame)
        frame_bib.pack(pady=10, fill="x", padx=20)
        tk.Label(frame_bib, text="Bibliothèques ou frameworks à utiliser (optionnel) :").pack(anchor="w")
        self.entry_bibliotheques = tk.Entry(frame_bib)
        self.entry_bibliotheques.pack(fill="x")
        
        # Contraintes
        frame_contr = tk.Frame(scrollable_frame)
        frame_contr.pack(pady=10, fill="both", expand=True, padx=20)
        tk.Label(frame_contr, text="Contraintes ou préférences :").pack(anchor="w")
        self.texte_contraintes = scrolledtext.ScrolledText(frame_contr, height=4, wrap=tk.WORD)
        self.texte_contraintes.pack(fill="both", expand=True)
        
        # Niveau
        frame_niveau = tk.Frame(scrollable_frame)
        frame_niveau.pack(pady=10, fill="x", padx=20)
        tk.Label(frame_niveau, text="Niveau de détail du code :").pack(anchor="w")
        self.var_niveau = tk.StringVar(value="intermédiaire")
        for text, value in [("Basique", "basique"), ("Intermédiaire", "intermédiaire"), ("Avancé", "avancé")]:
            tk.Radiobutton(frame_niveau, text=text, variable=self.var_niveau, value=value).pack(anchor="w")
        
        # Options supplémentaires
        frame_options = tk.Frame(scrollable_frame)
        frame_options.pack(pady=10, fill="x", padx=20)
        tk.Label(frame_options, text="Options supplémentaires :").pack(anchor="w")
        
        options_frame = tk.Frame(frame_options)
        options_frame.pack(fill="x")
        
        self.var_comments = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Inclure des commentaires", variable=self.var_comments).pack(side="left", padx=(0, 10))
        
        self.var_tests = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Inclure des tests unitaires", variable=self.var_tests).pack(side="left", padx=(0, 10))
        
        self.var_docs = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Inclure la documentation", variable=self.var_docs).pack(side="left")
        
        # Style de code
        frame_style = tk.Frame(scrollable_frame)
        frame_style.pack(pady=5, fill="x", padx=20)
        tk.Label(frame_style, text="Style de code (optionnel) :").pack(anchor="w")
        self.combo_style = ttk.Combobox(
            frame_style,
            values=["Standard", "PEP8 (Python)", "Airbnb (JavaScript)", "Google", "Autre"],
            state="readonly"
        )
        self.combo_style.pack(fill="x")
        self.combo_style.set("Standard")
        
        # Boutons d'action
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(pady=15, fill="x", padx=20)
        
        self.btn_generer = tk.Button(
            button_frame,
            text="Générer le Prompt",
            command=self.generer_prompt,
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 10, "bold")
        )
        self.btn_generer.pack(side="left", padx=(0, 10))
        
        self.btn_copier = tk.Button(
            button_frame,
            text="Copier le Prompt",
            command=self.copier_prompt,
            bg="#2196F3",
            fg="white"
        )
        self.btn_copier.pack(side="left", padx=(0, 10))
        
        self.btn_save = tk.Button(
            button_frame,
            text="Sauvegarder",
            command=self.save_prompt,
            bg="#FF9800",
            fg="white"
        )
        self.btn_save.pack(side="left")
        
        # Résultat
        frame_resultat = tk.Frame(scrollable_frame)
        frame_resultat.pack(pady=10, fill="both", expand=True, padx=20)
        tk.Label(frame_resultat, text="Prompt généré :").pack(anchor="w")
        self.texte_resultat = scrolledtext.ScrolledText(frame_resultat, height=12, wrap=tk.WORD, font=("Courier", 9))
        self.texte_resultat.pack(fill="both", expand=True)
    
    def setup_history_tab(self):
        # Configuration de l'onglet historique
        tk.Label(self.history_frame, text="Historique des prompts générés", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Liste de l'historique
        self.history_listbox = tk.Listbox(self.history_frame, height=15)
        self.history_listbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_select)
        
        # Boutons pour l'historique
        history_btn_frame = tk.Frame(self.history_frame)
        history_btn_frame.pack(pady=10)
        
        tk.Button(history_btn_frame, text="Charger la sélection", command=self.load_selected_history).pack(side="left", padx=5)
        tk.Button(history_btn_frame, text="Effacer l'historique", command=self.clear_history).pack(side="left", padx=5)
    
    def setup_templates_tab(self):
        # Configuration de l'onglet templates
        tk.Label(self.templates_frame, text="Gestion des templates de prompts", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Frame pour la recherche/filtrage par tags
        filter_frame = tk.Frame(self.templates_frame)
        filter_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filtrer par tag:").pack(side="left")
        self.filter_tag_var = tk.StringVar()
        filter_tag_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_tag_var,
            values=["Tous"] + list(self.tags),
            state="readonly",
            width=15
        )
        filter_tag_combo.pack(side="left", padx=5)
        filter_tag_combo.set("Tous")
        filter_tag_combo.bind("<<ComboboxSelected>>", self.filter_templates_by_tag)
        
        # Liste des templates
        tk.Label(self.templates_frame, text="Templates disponibles :").pack(anchor="w", padx=10, pady=(10, 0))
        self.templates_listbox = tk.Listbox(self.templates_frame, height=10)
        self.templates_listbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.templates_listbox.bind('<<ListboxSelect>>', self.on_template_select)
        
        # Boutons pour les templates
        template_btn_frame = tk.Frame(self.templates_frame)
        template_btn_frame.pack(pady=10)
        
        tk.Button(template_btn_frame, text="Charger le template", command=self.load_selected_template).pack(side="left", padx=5)
        tk.Button(template_btn_frame, text="Supprimer le template", command=self.delete_template).pack(side="left", padx=5)
        
        # Prévisualisation du template
        tk.Label(self.templates_frame, text="Prévisualisation :").pack(anchor="w", padx=10, pady=(10, 0))
        self.template_preview = scrolledtext.ScrolledText(self.templates_frame, height=8, wrap=tk.WORD)
        self.template_preview.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Mise à jour de la liste des templates
        self.update_templates_list()
    
    def setup_profiles_tab(self):
        # Configuration de l'onglet profils
        tk.Label(self.profiles_frame, text="Gestion des profils de sortie", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Liste des profils
        tk.Label(self.profiles_frame, text="Profils disponibles :").pack(anchor="w", padx=10)
        self.profiles_listbox = tk.Listbox(self.profiles_frame, height=10)
        self.profiles_listbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.profiles_listbox.bind('<<ListboxSelect>>', self.on_profile_select)
        
        # Boutons pour les profils
        profile_btn_frame = tk.Frame(self.profiles_frame)
        profile_btn_frame.pack(pady=10)
        
        tk.Button(profile_btn_frame, text="Nouveau profil", command=self.create_new_profile).pack(side="left", padx=5)
        tk.Button(profile_btn_frame, text="Modifier le profil", command=self.edit_profile).pack(side="left", padx=5)
        tk.Button(profile_btn_frame, text="Supprimer le profil", command=self.delete_profile).pack(side="left", padx=5)
        
        # Détails du profil
        tk.Label(self.profiles_frame, text="Détails du profil :").pack(anchor="w", padx=10, pady=(10, 0))
        self.profile_details = scrolledtext.ScrolledText(self.profiles_frame, height=8, wrap=tk.WORD)
        self.profile_details.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Mise à jour de la liste des profils
        self.update_profiles_list()
    
    def bind_mousewheel(self, canvas):
        # Permettre le défilement avec la molette
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
    
    def update_tags_suggestions(self, event):
        """Met à jour les suggestions de tags en fonction de la saisie"""
        current_text = self.tags_var.get().lower()
        if current_text:
            suggestions = [tag for tag in self.tags if current_text in tag.lower()]
            self.tags_entry['values'] = suggestions
        else:
            self.tags_entry['values'] = list(self.tags)
    
    def generer_prompt(self):
        langage = self.combo_langage.get().strip()
        fonctionnalites = self.texte_fonctionnalites.get("1.0", tk.END).strip()
        contraintes = self.texte_contraintes.get("1.0", tk.END).strip()
        bibliotheques = self.entry_bibliotheques.get().strip()
        niveau = self.var_niveau.get()
        tags = self.tags_var.get().strip()
        profile_name = self.profile_var.get()
        
        if not langage:
            messagebox.showwarning("Champ manquant", "Veuillez sélectionner un langage de programmation.")
            return
        if not fonctionnalites:
            messagebox.showwarning("Champ manquant", "Veuillez décrire les fonctionnalités souhaitées.")
            return
        
        # Récupérer le profil sélectionné
        profile = self.profiles.get(profile_name, {})
        
        # Récupérer les paradigmes sélectionnés
        paradigmes = [p for p, var in self.paradigme_vars.items() if var.get()]
        
        # Récupérer les options supplémentaires
        include_comments = self.var_comments.get()
        include_tests = self.var_tests.get()
        include_docs = self.var_docs.get()
        code_style = self.combo_style.get()
        
        # Construire le prompt avec le préfixe du profil
        prompt = profile.get("prefix", "")
        prompt += f"Tu es un développeur expérimenté en {langage}. "
        
        if paradigmes:
            prompt += f"Spécialiste en programmation {', '.join(paradigmes)}. "
        
        prompt += f"Écris un programme en {langage} qui implémente les fonctionnalités suivantes :\n\n"
        prompt += f"{fonctionnalites}\n\n"

        if bibliotheques:
            prompt += f"Utilise les bibliothèques ou frameworks suivants : {bibliotheques}.\n\n"

        if contraintes:
            prompt += f"Contraintes ou exigences spécifiques :\n{contraintes}\n\n"
        
        # Ajouter les exigences de style
        if code_style != "Standard":
            prompt += f"Respecte le style de code {code_style}.\n\n"
        
        # Ajouter les exigences supplémentaires
        additional_requirements = []
        if include_comments:
            additional_requirements.append("commenté de manière appropriée")
        if include_tests:
            additional_requirements.append("incluant des tests unitaires complets")
        if include_docs:
            additional_requirements.append("incluant une documentation complète")
        
        if additional_requirements:
            prompt += f"Le code doit être {', '.join(additional_requirements)}.\n\n"

        prompt += f"Le code doit être bien structuré et adapté à un niveau {niveau}. "
        prompt += "Fournis uniquement le code source complet, avec les imports nécessaires, et ajoute une brève explication si besoin."
        
        # Ajouter le suffixe du profil
        prompt += profile.get("suffix", "")

        self.texte_resultat.delete("1.0", tk.END)
        self.texte_resultat.insert(tk.END, prompt)
        
        # Ajouter à l'historique
        self.add_to_history(prompt)
    
    def copier_prompt(self):
        prompt = self.texte_resultat.get("1.0", tk.END).strip()
        if prompt:
            self.root.clipboard_clear()
            self.root.clipboard_append(prompt)
            messagebox.showinfo("Copié", "Le prompt a été copié dans le presse-papiers.")
        else:
            messagebox.showwarning("Aucun contenu", "Aucun prompt à copier.")
    
    def save_prompt(self):
        prompt = self.texte_resultat.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Aucun contenu", "Aucun prompt à sauvegarder.")
            return
        
        # Demander où sauvegarder
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(prompt)
                messagebox.showinfo("Sauvegardé", "Le prompt a été sauvegardé avec succès.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde : {str(e)}")
    
    def add_to_history(self, prompt):
        # Ajouter le prompt à l'historique avec un timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.insert(0, (timestamp, prompt))
        
        # Garder seulement les 10 derniers éléments
        if len(self.history) > self.max_history:
            self.history = self.history[:self.max_history]
        
        # Mettre à jour la liste d'historique
        self.update_history_list()
    
    def update_history_list(self):
        self.history_listbox.delete(0, tk.END)
        for timestamp, prompt in self.history:
            # Prendre les 50 premiers caractères pour l'affichage
            preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            self.history_listbox.insert(tk.END, f"{timestamp}: {preview}")
    
    def on_history_select(self, event):
        # Afficher le prompt complet dans une preview quand un élément est sélectionné
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            _, prompt = self.history[index]
            self.texte_resultat.delete("1.0", tk.END)
            self.texte_resultat.insert(tk.END, prompt)
    
    def load_selected_history(self):
        # Charger l'élément sélectionné de l'historique dans l'éditeur principal
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            _, prompt = self.history[index]
            self.texte_resultat.delete("1.0", tk.END)
            self.texte_resultat.insert(tk.END, prompt)
            self.notebook.select(0)  # Revenir à l'onglet principal
        else:
            messagebox.showwarning("Aucune sélection", "Veuillez sélectionner un élément de l'historique.")
    
    def clear_history(self):
        if messagebox.askyesno("Confirmer", "Voulez-vous vraiment effacer tout l'historique ?"):
            self.history = []
            self.update_history_list()
    
    def save_as_template(self):
        # Sauvegarder la configuration actuelle comme template
        name = simpledialog.askstring("Nom du template", "Entrez un nom pour ce template:")
        if not name:
            return
        
        # Demander les tags
        tags = simpledialog.askstring("Tags", "Entrez des tags séparés par des virgules (optionnel):")
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # Récupérer les paradigmes sélectionnés
        paradigmes = [p for p, var in self.paradigme_vars.items() if var.get()]
        
        # Créer le template
        self.prompt_templates[name] = {
            "langage": self.combo_langage.get(),
            "fonctionnalites": self.texte_fonctionnalites.get("1.0", tk.END).strip(),
            "contraintes": self.texte_contraintes.get("1.0", tk.END).strip(),
            "bibliotheques": self.entry_bibliotheques.get().strip(),
            "niveau": self.var_niveau.get(),
            "paradigmes": paradigmes,
            "comments": self.var_comments.get(),
            "tests": self.var_tests.get(),
            "docs": self.var_docs.get(),
            "style": self.combo_style.get(),
            "tags": tag_list
        }
        
        # Mettre à jour les tags disponibles
        self.tags.update(tag_list)
        self.tags_entry['values'] = list(self.tags)
        
        # Sauvegarder et mettre à jour l'interface
        self.save_external_data()
        self.update_templates_list()
        self.template_combo["values"] = list(self.prompt_templates.keys())
        messagebox.showinfo("Template sauvegardé", f"Le template '{name}' a été sauvegardé.")
    
    def update_templates_list(self):
        self.templates_listbox.delete(0, tk.END)
        for name in self.prompt_templates.keys():
            self.templates_listbox.insert(tk.END, name)
    
    def filter_templates_by_tag(self, event):
        tag = self.filter_tag_var.get()
        self.templates_listbox.delete(0, tk.END)
        
        if tag == "Tous":
            for name in self.prompt_templates.keys():
                self.templates_listbox.insert(tk.END, name)
        else:
            for name, template in self.prompt_templates.items():
                if "tags" in template and tag in template["tags"]:
                    self.templates_listbox.insert(tk.END, name)
    
    def on_template_select(self, event):
        # Afficher la preview du template sélectionné
        selection = self.templates_listbox.curselection()
        if selection:
            name = self.templates_listbox.get(selection[0])
            template = self.prompt_templates[name]
            
            # Générer une preview du prompt
            preview = f"Langage: {template['langage']}\n"
            preview += f"Niveau: {template['niveau']}\n"
            preview += f"Fonctionnalités: {template['fonctionnalites'][:100]}...\n"
            if template['contraintes']:
                preview += f"Contraintes: {template['contraintes'][:100]}...\n"
            if template['bibliotheques']:
                preview += f"Bibliothèques: {template['bibliotheques']}\n"
            if template['paradigmes']:
                preview += f"Paradigmes: {', '.join(template['paradigmes'])}\n"
            if 'tags' in template and template['tags']:
                preview += f"Tags: {', '.join(template['tags'])}\n"
            
            self.template_preview.delete("1.0", tk.END)
            self.template_preview.insert(tk.END, preview)
    
    def load_selected_template(self):
        # Charger le template sélectionné dans l'interface
        selection = self.templates_listbox.curselection()
        if selection:
            name = self.templates_listbox.get(selection[0])
            self.apply_template_by_name(name)
            self.notebook.select(0)  # Revenir à l'onglet principal
        else:
            messagebox.showwarning("Aucune sélection", "Veuillez sélectionner un template.")
    
    def apply_template(self, event=None):
        # Appliquer le template sélectionné dans la combobox
        name = self.template_var.get()
        if name in self.prompt_templates:
            self.apply_template_by_name(name)
    
    def apply_template_by_name(self, name):
        template = self.prompt_templates[name]
        
        # Appliquer les valeurs du template
        self.combo_langage.set(template['langage'])
        self.texte_fonctionnalites.delete("1.0", tk.END)
        self.texte_fonctionnalites.insert(tk.END, template['fonctionnalites'])
        self.texte_contraintes.delete("1.0", tk.END)
        self.texte_contraintes.insert(tk.END, template['contraintes'])
        self.entry_bibliotheques.delete(0, tk.END)
        self.entry_bibliotheques.insert(0, template['bibliotheques'])
        self.var_niveau.set(template['niveau'])
        self.combo_style.set(template['style'])
        
        # Appliquer les paradigmes
        for paradigme, var in self.paradigme_vars.items():
            var.set(paradigme in template['paradigmes'])
        
        # Appliquer les options
        self.var_comments.set(template['comments'])
        self.var_tests.set(template['tests'])
        self.var_docs.set(template['docs'])
        
        # Appliquer les tags
        if 'tags' in template and template['tags']:
            self.tags_var.set(", ".join(template['tags']))
        
        messagebox.showinfo("Template appliqué", f"Le template '{name}' a été appliqué.")
    
    def delete_template(self):
        # Supprimer le template sélectionné
        selection = self.templates_listbox.curselection()
        if selection:
            name = self.templates_listbox.get(selection[0])
            if messagebox.askyesno("Confirmer", f"Voulez-vous vraiment supprimer le template '{name}' ?"):
                del self.prompt_templates[name]
                self.save_external_data()
                self.update_templates_list()
                self.template_combo["values"] = list(self.prompt_templates.keys())
        else:
            messagebox.showwarning("Aucune sélection", "Veuillez sélectionner un template à supprimer.")
    
    def update_profiles_list(self):
        self.profiles_listbox.delete(0, tk.END)
        for name in self.profiles.keys():
            self.profiles_listbox.insert(tk.END, name)
    
    def on_profile_select(self, event):
        # Afficher les détails du profil sélectionné
        selection = self.profiles_listbox.curselection()
        if selection:
            name = self.profiles_listbox.get(selection[0])
            profile = self.profiles[name]
            
            details = f"Nom: {name}\n\n"
            details += f"Préfixe: {profile.get('prefix', 'Aucun')}\n\n"
            details += f"Suffixe: {profile.get('suffix', 'Aucun')}\n\n"
            details += f"Max tokens: {profile.get('max_tokens', 'Non spécifié')}\n"
            details += f"Température: {profile.get('temperature', 'Non spécifié')}"
            
            self.profile_details.delete("1.0", tk.END)
            self.profile_details.insert(tk.END, details)
    
    def create_new_profile(self):
        # Créer un nouveau profil
        name = simpledialog.askstring("Nom du profil", "Entrez un nom pour ce profil:")
        if not name:
            return
        
        if name in self.profiles:
            messagebox.showwarning("Existe déjà", "Un profil avec ce nom existe déjà.")
            return
        
        # Fenêtre pour configurer le nouveau profil
        self.edit_profile_window(name, {})
    
    def edit_profile(self):
        # Modifier le profil sélectionné
        selection = self.profiles_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aucune sélection", "Veuillez sélectionner un profil à modifier.")
            return
        
        name = self.profiles_listbox.get(selection[0])
        profile = self.profiles[name]
        
        # Fenêtre pour modifier le profil
        self.edit_profile_window(name, profile)
    
    def edit_profile_window(self, name, profile):
        # Créer une fenêtre pour éditer un profil
        window = tk.Toplevel(self.root)
        window.title(f"Édition du profil: {name}")
        window.geometry("500x400")
        window.transient(self.root)
        window.grab_set()
        
        tk.Label(window, text=f"Profil: {name}", font=("Helvetica", 12, "bold")).pack(pady=10)
        
        # Préfixe
        tk.Label(window, text="Préfixe du prompt:").pack(anchor="w", padx=10)
        prefix_text = scrolledtext.ScrolledText(window, height=4, wrap=tk.WORD)
        prefix_text.pack(fill="x", padx=10, pady=5)
        prefix_text.insert("1.0", profile.get("prefix", ""))
        
        # Suffixe
        tk.Label(window, text="Suffixe du prompt:").pack(anchor="w", padx=10, pady=(10, 0))
        suffix_text = scrolledtext.ScrolledText(window, height=4, wrap=tk.WORD)
        suffix_text.pack(fill="x", padx=10, pady=5)
        suffix_text.insert("1.0", profile.get("suffix", ""))
        
        # Paramètres avancés
        advanced_frame = tk.Frame(window)
        advanced_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(advanced_frame, text="Max tokens:").grid(row=0, column=0, sticky="w")
        max_tokens_var = tk.StringVar(value=str(profile.get("max_tokens", 1500)))
        max_tokens_entry = tk.Entry(advanced_frame, textvariable=max_tokens_var, width=10)
        max_tokens_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(advanced_frame, text="Température:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        temp_var = tk.StringVar(value=str(profile.get("temperature", 0.7)))
        temp_entry = tk.Entry(advanced_frame, textvariable=temp_var, width=10)
        temp_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Boutons de sauvegarde/annulation
        btn_frame = tk.Frame(window)
        btn_frame.pack(pady=10)
        
        def save_profile():
            self.profiles[name] = {
                "prefix": prefix_text.get("1.0", tk.END).strip(),
                "suffix": suffix_text.get("1.0", tk.END).strip(),
                "max_tokens": int(max_tokens_var.get()),
                "temperature": float(temp_var.get())
            }
            self.save_external_data()
            self.update_profiles_list()
            self.profile_var.set(name)  # Mettre à jour la sélection dans l'onglet principal
            window.destroy()
            messagebox.showinfo("Profil sauvegardé", f"Le profil '{name}' a été sauvegardé.")
        
        tk.Button(btn_frame, text="Sauvegarder", command=save_profile).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Annuler", command=window.destroy).pack(side="left", padx=5)
    
    def delete_profile(self):
        # Supprimer le profil sélectionné
        selection = self.profiles_listbox.curselection()
        if selection:
            name = self.profiles_listbox.get(selection[0])
            if messagebox.askyesno("Confirmer", f"Voulez-vous vraiment supprimer le profil '{name}' ?"):
                del self.profiles[name]
                self.save_external_data()
                self.update_profiles_list()
                
                # Si le profil supprimé était sélectionné dans l'onglet principal
                if self.profile_var.get() == name:
                    if self.profiles:
                        self.profile_var.set(next(iter(self.profiles.keys())))
                    else:
                        self.profile_var.set("")
        else:
            messagebox.showwarning("Aucune sélection", "Veuillez sélectionner un profil à supprimer.")


# Lancement de l'application
if __name__ == "__main__":
    root = tk.Tk()
    app = GGCodeGenerator(root)
    root.mainloop()
