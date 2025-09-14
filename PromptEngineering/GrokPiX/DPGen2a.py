import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from datetime import datetime
import os

def create_universal_prompt_gui():
    root = tk.Tk()
    root.title("🎨 Générateur de Prompt Artistique (Universel)")
    root.geometry("950x700")
    root.configure(padx=15, pady=15)

    # === Canvas avec Scrollbar vertical ===
    main_canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    scrollable_frame = ttk.Frame(main_canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
    )

    main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    main_canvas.configure(yscrollcommand=scrollbar.set)

    # Positionnement du canvas et de la scrollbar
    main_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Permettre au canvas de s'étendre
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    scrollable_frame.grid_columnconfigure(0, weight=1)

    # === Variables (exemples pédagogiques) ===
    meta_version = tk.StringVar(value="1.0")
    meta_auteur = tk.StringVar(value="Votre nom ou pseudonyme")
    meta_date_creation = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
    meta_projet = tk.StringVar(value="Nom de votre projet artistique (ex: 'Fantasy Illustrée', 'Réalisme urbain')")

    sujet_principal = tk.StringVar(value="Un guerrier solitaire")
    description_sujet = tk.StringVar(value="vêtu d'une armure ancienne couverte de runes lumineuses, tenant une épée brisée, regard déterminé")
    contexte = tk.StringVar(value="debout sur une colline venteuse sous un ciel orageux, avec des ruines médiévales en arrière-plan")

    style_artistique = tk.StringVar(value="style peinture à l'huile numérique, réalisme fantastique, inspiration par Greg Rutkowski et Artgerm")
    mouvement_artistique = tk.StringVar(value="fantasy épique, néo-romantisme numérique, art conceptuel")
    composition = tk.StringVar(value="plan rapproché en contre-plongée, règle des tiers appliquée, arrière-plan flou mais évocateur")
    lumiere = tk.StringVar(value="lumière dramatique venant d'en haut, contrastes forts, reflets sur l'armure, ombres profondes")
    couleurs = tk.StringVar(value="tons froids dominants (bleus, gris acier), touches de rouge sang et d'or ancien, saturation modérée")
    ambiance = tk.StringVar(value="mélancolique, héroïque, solitude face au destin, tension palpable")
    details_visuels = tk.StringVar(value="pluie fine en suspension, déchirures dans le manteau, poussière soulevée par le vent, fissures dans le sol")
    elements_symboliques = tk.StringVar(value="un corbeau perché sur une pierre, une bannière en lambeaux flottant au vent")
    qualite = tk.StringVar(value="ultra-réaliste, 8K, textures détaillées, lumière volumétrique, rendu cinématographique")

    # Interdits
    interdits_modernes = tk.BooleanVar()
    interdits_fond_blanc = tk.BooleanVar()
    interdits_cartoon_simple = tk.BooleanVar()
    interdits_humains = tk.BooleanVar()

    # Paramètres techniques
    ratio_aspect = tk.StringVar(value="16:9")
    style_dalle = tk.StringVar(value="vivid")
    niveau_detail = tk.StringVar(value="maximum")

    langue_prompt = tk.StringVar(value="français")
    variation = tk.IntVar(value=1)
    mots_cles = tk.StringVar(value="fantasy, guerrier, ruines, tempête, réalisme, épique, courage")

    # === Fonction pour charger les templates ===
    def charger_templates():
        template_file = "DPTemplate.json"
        templates = {}
        
        if os.path.exists(template_file):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    templates_data = json.load(f)
                    for template in templates_data.get("templates", []):
                        templates[template["nom"]] = template
                return templates
            except Exception as e:
                messagebox.showerror("❌ Erreur", f"Erreur lors du chargement des templates :\n{str(e)}")
                return {}
        else:
            # Créer un fichier de templates par défaut s'il n'existe pas
            templates_par_defaut = {
                "templates": [
                    {
                        "nom": "Médiéval Fantastique",
                        "sujet_principal": "Un chevalier en armure étincelante",
                        "description_sujet": "tenant une épée légendaire, regard déterminé",
                        "contexte": "dans un château en ruine au sommet d'une montagne brumeuse",
                        "style_artistique": "style peinture à l'huile numérique, réalisme fantastique",
                        "mouvement_artistique": "fantasy épique, art médiéval revisité",
                        "composition": "plan large, perspective atmosphérique",
                        "lumiere": "lumière dorée du coucher de soleil, rayons crépusculaires",
                        "couleurs": "tons terreux, ors et argentés, touches de bleu royal",
                        "ambiance": "épique, mystérieuse, empreinte de légende",
                        "details_visuels": "bannières flottantes, pierres anciennes, lichen sur les murs",
                        "elements_symboliques": "un dragon sculpté sur l'armure, un blason familial",
                        "qualite": "ultra-détaillé, 8K, textures réalistes",
                        "interdits": ["pas d'éléments modernes"],
                        "mots_cles": "médiéval, fantastique, chevalier, château, légende"
                    },
                    {
                        "nom": "Science-Fiction",
                        "sujet_principal": "Un explorateur spatial en combinaison high-tech",
                        "description_sujet": "équipé d'un casque à visière holographique et d'outils futuristes",
                        "contexte": "sur une planète étrangère aux couleurs vibrantes, avec deux lunes à l'horizon",
                        "style_artistique": "digital painting, concept art, style cinématographique",
                        "mouvement_artistique": "futurisme, cyberpunk, rétrofuturisme",
                        "composition": "plan américain, angle dynamique",
                        "lumiere": "éclairage néon, lueurs d'hologrammes, contrastes marqués",
                        "couleurs": "violet électrique, bleu néon, orange fluorescent sur fond noir profond",
                        "ambiance": "mystérieuse, avant-gardiste, sensation de découverte",
                        "details_visuels": "projections holographiques, interfaces numériques, particules d'énergie",
                        "elements_symboliques": "un artefact alien énigmatique, un robot compagnon",
                        "qualite": "détaillé, 4K, effets de lumière complexes",
                        "interdits": ["pas d'éléments médiévaux"],
                        "mots_cles": "science-fiction, espace, futuriste, alien, technologie"
                    },
                    {
                        "nom": "Cyberpunk",
                        "sujet_principal": "Un hacker augmenté dans les ruelles de Neo-Tokyo",
                        "description_sujet": "avec des implants cybernétiques visibles, vêtements synthétiques",
                        "contexte": "dans une ruelle sombre inondée de néons, sous une pluie fine",
                        "style_artistique": "style Blade Runner, inspiration Syd Mead",
                        "mouvement_artistique": "cyberpunk, post-modernisme",
                        "composition": "cadrage serré, forte perspective",
                        "lumiere": "néons colorés, reflets sur sol mouillé, zones d'ombre profondes",
                        "couleurs": "magenta, cyan, jaune électrique sur fond noir et gris métallique",
                        "ambiance": "oppressante, mélancolique mais vibrante d'énergie",
                        "details_visuels": "enseignes holographiques, câbles électriques, fumée s'échappant des grilles",
                        "elements_symboliques": "un terminal de données antique, un graffiti numérique",
                        "qualite": "high contrast, sharp focus, cinematic lighting",
                        "interdits": ["pas d'éléments naturels", "pas de style vintage"],
                        "mots_cles": "cyberpunk, néon, futuriste, technologie, nuit"
                    },
                    {
                        "nom": "Steampunk",
                        "sujet_principal": "Un inventeur victorien en tenue d'explorateur",
                        "description_sujet": "avec des lunettes à volets multiples et des outils mécaniques complexes",
                        "contexte": "dans son laboratoire rempli de machines à vapeur et d'inventions bizarres",
                        "style_artistique": "illustration vintage, précision mécanique",
                        "mouvement_artistique": "steampunk, néo-victorien",
                        "composition": "plan moyen, perspective symétrique",
                        "lumiere": "lumière chaude de lampes à huile, reflets cuivrés, ombres douces",
                        "couleurs": "bruns sépia, cuivre, laiton, touches de vert bouteille",
                        "ambiance": "curiosité inventive, mystère technologique, nostalgie rétro-futuriste",
                        "details_visuels": "engrenages visibles, vapeur s'échappant, détails en laiton",
                        "elements_symboliques": "une carte au trésor mécanique, un oiseau automaton",
                        "qualite": "textures métalliques détaillées, rendu vintagé",
                        "interdits": ["pas d'éléments numériques modernes"],
                        "mots_cles": "steampunk, victorien, mécanique, vapeur, rétro-futuriste"
                    },
                    {
                        "nom": "Manga",
                        "sujet_principal": "Un jeune héros aux pouvoirs extraordinaires",
                        "description_sujet": "coiffure stylisée, expression intense, vêtements dynamiques",
                        "contexte": "au milieu d'un combat explosif dans un paysage urbain dévasté",
                        "style_artistique": "anime de haute qualité, style studio Ghibli meets Shonen",
                        "mouvement_artistique": "manga moderne, influence japonaise",
                        "composition": "angles dramatiques, lignes de vitesse",
                        "lumiere": "éclats d'énergie, reflets dans les yeux, effets spéciaux lumineux",
                        "couleurs": "palette vibrante, couleurs saturées, contrastes marqués",
                        "ambiance": "dynamique, intense, émotionnelle",
                        "details_visuels": "effets de particules, lignes de mouvement, expressions faciales exagérées",
                        "elements_symboliques": "un artefact donnant des pouvoirs, une créature mascotte",
                        "qualite": "style cel-shading, contours nets, couleurs plates mais expressives",
                        "interdits": ["pas de réalisme photographique"],
                        "mots_cles": "manga, anime, japonais, dynamique, énergie"
                    },
                    {
                        "nom": "Photo Vintage",
                        "sujet_principal": "Un portrait d'époque en noir et blanc",
                        "description_sujet": "regard pénétrant, vêtements historiques, pose naturelle",
                        "contexte": "dans un studio photographique ancien avec fond texturé",
                        "style_artistique": "photographie argentique, procédé au collodion humide",
                        "mouvement_artistique": "réalisme historique, documentaire",
                        "composition": "cadrage classique, pose étudiée",
                        "lumiere": "éclairage studio vintage, contraste élevé, grain photographique",
                        "couleurs": "noir et blanc, sépia, tons neutres",
                        "ambiance": "nostalgique, intemporelle, authentique",
                        "details_visuels": "grain argentique, légères rayures, vignettage",
                        "elements_symboliques": "un accessoire d'époque, un cadre ornemental",
                        "qualite": "haute résolution, détails fins, aspect vieilli authentique",
                        "interdits": ["pas de couleurs vives", "pas d'effets numériques modernes"],
                        "mots_cles": "vintage, photographie, historique, nostalgique, noir et blanc"
                    }
                ]
            }
            
            try:
                with open(template_file, 'w', encoding='utf-8') as f:
                    json.dump(templates_par_defaut, f, ensure_ascii=False, indent=2)
                return {t["nom"]: t for t in templates_par_defaut["templates"]}
            except Exception as e:
                messagebox.showerror("❌ Erreur", f"Erreur lors de la création des templates :\n{str(e)}")
                return {}
    
    # Charger les templates
    templates = charger_templates()
    template_selectionne = tk.StringVar(value="Aucun")

    # === Fonction pour appliquer un template ===
    def appliquer_template(event=None):
        nom_template = template_selectionne.get()
        if nom_template != "Aucun" and nom_template in templates:
            template = templates[nom_template]
            
            # Appliquer les valeurs du template aux variables
            sujet_principal.set(template.get("sujet_principal", ""))
            description_sujet.set(template.get("description_sujet", ""))
            contexte.set(template.get("contexte", ""))
            style_artistique.set(template.get("style_artistique", ""))
            mouvement_artistique.set(template.get("mouvement_artistique", ""))
            composition.set(template.get("composition", ""))
            lumiere.set(template.get("lumiere", ""))
            couleurs.set(template.get("couleurs", ""))
            ambiance.set(template.get("ambiance", ""))
            details_visuels.set(template.get("details_visuels", ""))
            elements_symboliques.set(template.get("elements_symboliques", ""))
            qualite.set(template.get("qualite", ""))
            mots_cles.set(template.get("mots_cles", ""))
            
            # Réinitialiser les interdits
            interdits_modernes.set(False)
            interdits_fond_blanc.set(False)
            interdits_cartoon_simple.set(False)
            interdits_humains.set(False)
            
            # Appliquer les interdits du template
            for interdit in template.get("interdits", []):
                if "modernes" in interdit:
                    interdits_modernes.set(True)
                if "fonds blancs" in interdit:
                    interdits_fond_blanc.set(True)
                if "cartoon simple" in interdit:
                    interdits_cartoon_simple.set(True)
                if "humains" in interdit:
                    interdits_humains.set(True)
            
            messagebox.showinfo("✅ Template appliqué", f"Le template '{nom_template}' a été appliqué avec succès.")

    # === Exporter le JSON ===
    def exporter_json():
        try:
            interdits = []
            if interdits_modernes.get(): interdits.append("pas d'éléments modernes")
            if interdits_fond_blanc.get(): interdits.append("pas de fonds blancs")
            if interdits_cartoon_simple.get(): interdits.append("pas de style cartoon simple")
            if interdits_humains.get(): interdits.append("pas d'humains")

            data = {
                "meta": {
                    "version": meta_version.get(),
                    "auteur": meta_auteur.get(),
                    "date_creation": meta_date_creation.get(),
                    "projet": meta_projet.get(),
                    "template": template_selectionne.get()
                },
                "sujet_principal": sujet_principal.get(),
                "description_sujet": description_sujet.get(),
                "contexte": contexte.get(),
                "style_artistique": style_artistique.get(),
                "mouvement_artistique": mouvement_artistique.get(),
                "composition": composition.get(),
                "lumiere": lumiere.get(),
                "couleurs": couleurs.get(),
                "ambiance": ambiance.get(),
                "détails_visuels": details_visuels.get(),
                "éléments_symboliques": elements_symboliques.get(),
                "qualité": qualite.get(),
                "interdits": interdits,
                "paramètres_techniques": {
                    "ratio_aspect": ratio_aspect.get(),
                    "style_dall_e": style_dalle.get(),
                    "niveau_détail": niveau_detail.get()
                },
                "langue_prompt": langue_prompt.get(),
                "variation": variation.get(),
                "mots_clés": [k.strip() for k in mots_cles.get().split(",") if k.strip()]
            }

            fichier = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")],
                title="Enregistrer le prompt"
            )
            if fichier:
                with open(fichier, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("✅ Succès", f"Prompt exporté avec succès :\n{fichier}")

        except Exception as e:
            messagebox.showerror("❌ Erreur", f"Échec de l'export :\n{str(e)}")

    # === Fonction pour placeholders ===
    def add_placeholder(entry, placeholder):
        def on_focus_in(event):
            if entry.get() == placeholder:
                entry.delete(0, tk.END)
                entry.config(fg="black")

        def on_focus_out(event):
            if entry.get() == "":
                entry.delete(0, tk.END)
                entry.insert(0, placeholder)
                entry.config(fg="gray")

        entry.insert(0, placeholder)
        entry.config(fg="gray")
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    # === Frame pour la sélection de template ===
    template_frame = ttk.Frame(scrollable_frame)
    template_frame.pack(pady=10, fill="x", padx=10)

    tk.Label(template_frame, text="🎭 Template:", font=("Helvetica", 10, "bold")).pack(side="left", padx=(0, 10))
    
    template_names = ["Aucun"] + list(templates.keys())
    template_dropdown = ttk.Combobox(
        template_frame, 
        textvariable=template_selectionne, 
        values=template_names,
        state="readonly",
        width=30
    )
    template_dropdown.pack(side="left", padx=(0, 10))
    template_dropdown.bind("<<ComboboxSelected>>", appliquer_template)
    
    tk.Button(
        template_frame, 
        text="Appliquer", 
        command=appliquer_template,
        bg="#4CAF50",
        fg="white"
    ).pack(side="left")

    # === Notebook dans le cadre défilable ===
    notebook = ttk.Notebook(scrollable_frame)
    notebook.pack(pady=10, padx=10, fill="both", expand=True)

    # --- Onglet Métadonnées ---
    tab_meta = ttk.Frame(notebook)
    notebook.add(tab_meta, text="📝 Métadonnées")

    tk.Label(tab_meta, text="Version").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    e1 = tk.Entry(tab_meta, textvariable=meta_version, width=50)
    e1.grid(row=0, column=1, padx=10, pady=5)
    add_placeholder(e1, "ex: 1.0, 2.1, beta")

    tk.Label(tab_meta, text="Auteur").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    e2 = tk.Entry(tab_meta, textvariable=meta_auteur, width=50)
    e2.grid(row=1, column=1, padx=10, pady=5)
    add_placeholder(e2, "Votre nom, pseudonyme ou studio")

    tk.Label(tab_meta, text="Date de création").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    e3 = tk.Entry(tab_meta, textvariable=meta_date_creation, width=50)
    e3.grid(row=2, column=1, padx=10, pady=5)
    add_placeholder(e3, "YYYY-MM-DD")

    tk.Label(tab_meta, text="Projet").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    e4 = tk.Entry(tab_meta, textvariable=meta_projet, width=50)
    e4.grid(row=3, column=1, padx=10, pady=5)
    add_placeholder(e4, "Nom du projet artistique ou série")

    # --- Onglet Sujet ---
    tab_sujet = ttk.Frame(notebook)
    notebook.add(tab_sujet, text="🎯 Sujet")

    tk.Label(tab_sujet, text="Sujet principal").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    e5 = tk.Entry(tab_sujet, textvariable=sujet_principal, width=70)
    e5.grid(row=0, column=1, padx=10, pady=5)
    add_placeholder(e5, "ex: Un dragon endormi, Une forêt vivante, Une ville flottante")

    tk.Label(tab_sujet, text="Description du sujet").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    e6 = tk.Entry(tab_sujet, textvariable=description_sujet, width=70)
    e6.grid(row=1, column=1, padx=10, pady=5)
    add_placeholder(e6, "Détails physiques, posture, émotion, vêtements, accessoires...")

    tk.Label(tab_sujet, text="Contexte").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    e7 = tk.Entry(tab_sujet, textvariable=contexte, width=70)
    e7.grid(row=2, column=1, padx=10, pady=5)
    add_placeholder(e7, "Lieu, environnement, arrière-plan, conditions météo...")

    # --- Onglet Style ---
    tab_style = ttk.Frame(notebook)
    notebook.add(tab_style, text="🎨 Style")

    tk.Label(tab_style, text="Style artistique").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    e8 = tk.Entry(tab_style, textvariable=style_artistique, width=70)
    e8.grid(row=0, column=1, padx=10, pady=5)
    add_placeholder(e8, "ex: aquarelle numérique, cyberpunk 80s, pixel art 16-bit")

    tk.Label(tab_style, text="Mouvement artistique").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    e9 = tk.Entry(tab_style, textvariable=mouvement_artistique, width=70)
    e9.grid(row=1, column=1, padx=10, pady=5)
    add_placeholder(e9, "ex: surrealisme, futurisme, romantisme numérique")

    tk.Label(tab_style, text="Composition").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    e10 = tk.Entry(tab_style, textvariable=composition, width=70)
    e10.grid(row=2, column=1, padx=10, pady=5)
    add_placeholder(e10, "ex: plan large, symétrie radiale, règle des tiers, profondeur de champ")

    tk.Label(tab_style, text="Lumière").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    e11 = tk.Entry(tab_style, textvariable=lumiere, width=70)
    e11.grid(row=3, column=1, padx=10, pady=5)
    add_placeholder(e11, "ex: contre-jour, lumière dorée du matin, néons urbains, clair-obscur")

    tk.Label(tab_style, text="Couleurs").grid(row=4, column=0, sticky="w", padx=10, pady=5)
    e12 = tk.Entry(tab_style, textvariable=couleurs, width=70)
    e12.grid(row=4, column=1, padx=10, pady=5)
    add_placeholder(e12, "ex: tons pastel, palette monochrome bleue, contrastes chaud/froid")

    tk.Label(tab_style, text="Ambiance").grid(row=5, column=0, sticky="w", padx=10, pady=5)
    e13 = tk.Entry(tab_style, textvariable=ambiance, width=70)
    e13.grid(row=5, column=1, padx=10, pady=5)
    add_placeholder(e13, "ex: mystérieuse, joyeuse, angoissante, paisible, héroïque")

    # --- Onglet Détails ---
    tab_details = ttk.Frame(notebook)
    notebook.add(tab_details, text="🔍 Détails")

    tk.Label(tab_details, text="Détails visuels").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    e14 = tk.Entry(tab_details, textvariable=details_visuels, width=70)
    e14.grid(row=0, column=1, padx=10, pady=5)
    add_placeholder(e14, "ex: poussière en suspension, reflets sur le métal, brume légère, texture du tissu")

    tk.Label(tab_details, text="Éléments symboliques").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    e15 = tk.Entry(tab_details, textvariable=elements_symboliques, width=70)
    e15.grid(row=1, column=1, padx=10, pady=5)
    add_placeholder(e15, "ex: un livre ancien, un miroir brisé, une fleur rare, une horloge arrêtée")

    tk.Label(tab_details, text="Qualité").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    e16 = tk.Entry(tab_details, textvariable=qualite, width=70)
    e16.grid(row=2, column=1, padx=10, pady=5)
    add_placeholder(e16, "ex: 8K, ultra-détaillé, textures réalistes, sharp focus, ray tracing")

    # --- Onglet Interdits ---
    tab_interdits = ttk.Frame(notebook)
    notebook.add(tab_interdits, text="🚫 Interdits")

    tk.Checkbutton(tab_interdits, text="Pas d'éléments modernes (ex: smartphones, voitures)", variable=interdits_modernes).grid(row=0, column=0, sticky="w", padx=20, pady=5)
    tk.Checkbutton(tab_interdits, text="Pas de fonds blancs ou plats", variable=interdits_fond_blanc).grid(row=1, column=0, sticky="w", padx=20, pady=5)
    tk.Checkbutton(tab_interdits, text="Pas de style cartoon simple ou enfantin", variable=interdits_cartoon_simple).grid(row=2, column=0, sticky="w", padx=20, pady=5)
    tk.Checkbutton(tab_interdits, text="Pas d'humains (si sujet non-humain)", variable=interdits_humains).grid(row=3, column=0, sticky="w", padx=20, pady=5)

    # --- Onglet Techniques ---
    tab_params = ttk.Frame(notebook)
    notebook.add(tab_params, text="⚙️ Paramètres")

    tk.Label(tab_params, text="Ratio d'aspect").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    ttk.Combobox(tab_params, textvariable=ratio_aspect, values=["1:1", "4:3", "16:9", "9:16", "21:9"], state="readonly").grid(row=0, column=1, padx=10, pady=5)

    tk.Label(tab_params, text="Style DALL·E").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    ttk.Combobox(tab_params, textvariable=style_dalle, values=["vivid", "natural"], state="readonly").grid(row=1, column=1, padx=10, pady=5)

    tk.Label(tab_params, text="Niveau de détail").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    ttk.Combobox(tab_params, textvariable=niveau_detail, values=["maximum", "high", "medium"], state="readonly").grid(row=2, column=1, padx=10, pady=5)

    tk.Label(tab_params, text="Langue du prompt").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    lang_entry = tk.Entry(tab_params, textvariable=langue_prompt, width=30)
    lang_entry.grid(row=3, column=1, sticky="w", padx=10, pady=5)
    add_placeholder(lang_entry, "français, anglais, espagnol...")

    tk.Label(tab_params, text="Nombre de variations").grid(row=4, column=0, sticky="w", padx=10, pady=5)
    tk.Spinbox(tab_params, from_=1, to=10, textvariable=variation, width=10).grid(row=4, column=1, sticky="w", padx=10, pady=5)

    tk.Label(tab_params, text="Mots-clés (séparés par des virgules)").grid(row=5, column=0, sticky="w", padx=10, pady=5)
    keywords_entry = tk.Entry(tab_params, textvariable=mots_cles, width=70)
    keywords_entry.grid(row=5, column=1, padx=10, pady=5)
    add_placeholder(keywords_entry, "ex: nature, magie, ancien, lumière, ruines, puissance")

    # === Bouton d'export en bas du cadre défilable ===
    btn_frame = ttk.Frame(scrollable_frame)
    btn_frame.pack(pady=20)

    btn_export = tk.Button(
        btn_frame,
        text="💾 Exporter en JSON",
        font=("Helvetica", 12, "bold"),
        bg="#2E86AB",
        fg="white",
        command=exporter_json,
        padx=25,
        pady=10,
        relief="flat"
    )
    btn_export.pack()

    # === Permettre le scroll avec la molette de la souris ===
    def _on_mousewheel(event):
        main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    main_canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
    main_canvas.bind_all("<Button-4>", lambda e: main_canvas.yview_scroll(-1, "units"))  # Linux haut
    main_canvas.bind_all("<Button-5>", lambda e: main_canvas.yview_scroll(1, "units"))   # Linux bas

    # Nettoyer le bind à la fermeture
    root.bind("<Destroy>", lambda e: main_canvas.unbind_all("<MouseWheel>"))

    # === Lancer l'interface ===
    root.mainloop()

if __name__ == "__main__":
    create_universal_prompt_gui()
