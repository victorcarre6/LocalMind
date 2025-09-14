from PromptToolkit import *
import json
from typing import Dict, Any
from dataclasses import asdict


prompt1 = Prompt(
    instruction="Agis comme un professeur de physique. Explique la théorie de la relativité restreinte.",
    context="Public : lycéens.",
    examples=[
        "Exemple : 'Imaginez que vous êtes dans un train qui roule très vite...'",
    ],
    constraints=[
        "Utiliser des analogies visuelles.",
        "Éviter les équations complexes.",
    ],
    output_format="Texte pédagogique."
)


prompt2 = Prompt(
    instruction="ELI5 : C'est quoi un algorithme ?",
    context="Public : enfant de 5 ans.",
    constraints=[
        "Utiliser une analogie avec une recette de cuisine.",
        "1 paragraphe maximum.",
    ],
    output_format="Texte enfantin."
)

prompt3 = Prompt(
    instruction="Mode concis : Quelles sont les 3 différences entre SQL et NoSQL ?",
    context="Pour un entretien technique.",
    constraints=[
        "1 phrase par différence.",
        "Pas de détails superflus.",
    ],
    output_format="Liste courte."
)

class PromptTemplate:
    def __init__(self, template_name: str, base_prompt: Prompt):
        self.template_name = template_name
        self.base_prompt = base_prompt

    def generate(self, **kwargs) -> Prompt:
        """Génère un prompt personnalisé à partir du template."""
        # Récupère les attributs du base_prompt
        base_dict = asdict(self.base_prompt)

        # Remplace les placeholders dans instruction et context
        instruction = base_dict.get("instruction", "").format(**kwargs)
        context = base_dict.get("context", "").format(**kwargs)

        # Met à jour les valeurs avec les kwargs
        base_dict["instruction"] = instruction
        base_dict["context"] = context

        # Ajoute les exemples et contraintes supplémentaires
        if "examples" in kwargs:
            base_dict["examples"] = base_dict.get("examples", []) + kwargs["examples"]
        if "constraints" in kwargs:
            base_dict["constraints"] = base_dict.get("constraints", []) + kwargs["constraints"]

        # Crée un nouvel objet Prompt avec les valeurs mises à jour
        return Prompt(**base_dict)

# Exemple d'utilisation
tech_analysis_template = PromptTemplate(
    template_name="analyse_technique",
    base_prompt=Prompt(
        instruction="Compare {tool1} et {tool2} pour {use_case}.",
        context="Public : {audience}. Focus sur {criteria}.",
        output_format="Markdown (tableau comparatif)."
    )
)

custom_prompt = tech_analysis_template.generate(
    tool1="React",
    tool2="Vue.js",
    use_case="une application web en 2025",
    audience="développeurs full-stack",
    criteria="performances, communauté, courbe d'apprentissage",
    examples=["Exemple : 'React utilise un DOM virtuel, tandis que Vue.js...'"],
    constraints=["Citer des benchmarks 2024-2025.", "Inclure des extraits de code."]
)

print(custom_prompt.format())

#print(prompt.format())
