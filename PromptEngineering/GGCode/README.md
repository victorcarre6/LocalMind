# Prompt Generator for Developers (GGCode)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![JSON Configurable](https://img.shields.io/badge/Config-JSON-blueviolet)

Un générateur de prompts intelligent et configurable pour développeurs, conçu pour automatiser la création de prompts précis destinés aux modèles linguistiques (LLMs) comme ChatGPT, Claude, Gemini, ou Hugging Face. Basé sur des **templates JSON**, il permet de générer des instructions claires, structurées et techniques pour des tâches de développement variées : génération de code, tests, documentation, optimisation, analyse NLP, systèmes artificiels, etc.

---

## 🌟 Fonctionnalités

- ✅ **Templates JSON personnalisables** : définissez vos propres modèles de prompts.
- 🔧 **Profils modulables** : adaptez le ton et le style (ex: "expert", "expérimenté").
- 🐍 **Support multi-langage** : principalement Python, mais extensible à d'autres.
- 📚 **Contraintes techniques détaillées** : bibliothèques, normes de code (PEP8), tests, commentaires.
- 🧠 **Paradigmes de programmation** : NLP, fonctionnel, orienté objet, simulation, web sémantique, etc.
- 📊 **Métriques & évaluation** : intégration avec BLEU, ROUGE, perplexité, entropie, etc.
- 🌐 **Interface web interactive** (via Streamlit) : visualisation, test, export.
- 💾 **Export en JSON/CSV** : pour l'analyse, la traçabilité et l'automatisation.
- 🤖 **Compatibilité LLM** : OpenAI, Azure, Hugging Face, Google Gemini, AWS Bedrock.

---

## 📁 Structure des Templates

Les templates sont stockés dans `templates.json` et contiennent :

```json
{
  "Nom du Template": {
    "langage": "Python",
    "fonctionnalites": "Description des fonctionnalités attendues",
    "contraintes": "Exigences techniques (fichiers, API, export)",
    "bibliotheques": "Liste des dépendances (ex: pandas, streamlit)",
    "niveau": "basique | intermédiaire | avancé",
    "paradigmes": ["NLP", "Fonctionnel", "Simulation"],
    "comments": true,
    "tests": true,
    "docs": true,
    "style": "PEP8 (Python)",
    "tags": ["python", "prompt-engineering", "nlp"]
  }
}