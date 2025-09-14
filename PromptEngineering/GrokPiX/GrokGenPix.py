import json
import re
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

class PromptGenerator:
    """Générateur de prompts optimisés pour le générateur d'images de Grok 3"""
    
    def __init__(self, config_file: str = "DPrompt.json"):
        self.config_file = config_file
        self.config = self.charger_config()
        self.historique = []
        
    def charger_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier JSON"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Fichier {self.config_file} non trouvé. Utilisation de la configuration par défaut.")
            return self.config_par_defaut()
        except json.JSONDecodeError:
            print(f"❌ Erreur de décodage JSON dans {self.config_file}. Utilisation de la configuration par défaut.")
            return self.config_par_defaut()
    
    def config_par_defaut(self) -> Dict[str, Any]:
        """Retourne une configuration par défaut basée sur DPrompt.json si non trouvé"""
        return {
            "sujet_principal": "un robot jardinier vintage",
            "description_sujet": "en costume en cuir brun et lunettes rondes, arrosant des fleurs lumineuses",
            "contexte": "dans un jardin botanique futuriste niché dans une forêt tropicale",
            "style_artistique": "réalisme cinématographique",
            "mouvement_artistique": "rétro-futurisme",
            "composition": "plan large avec profondeur de champ",
            "lumiere": "lumière dorée du lever du soleil",
            "couleurs": "verts émeraude, ors chauds",
            "ambiance": "poétique, paisible",
            "détails_visuels": "gouttes de rosée qui scintillent",
            "éléments_symboliques": "une horloge ancienne intégrée dans un tronc d'arbre",
            "qualité": "ultra-détaillé, 8K",
            "interdits": "pas de visages humains, pas de ciel nuageux",
            "langue_prompt": "français",
            "variation": 3,
            "ratio_aspect": "4:3"
        }
    
    def generer_variations(self, nombre_variations: int) -> List[str]:
        """Génère plusieurs variations du prompt"""
        variations = []
        for i in range(nombre_variations):
            variation_config = self.config.copy()
            
            if random.random() > 0.7 and "description_sujet" in variation_config:
                variations_desc = [
                    "tenant un arrosoir antique en cuivre",
                    "avec des outils de jardinage chromés",
                    "entouré de papillons mécaniques"
                ]
                variation_config["description_sujet"] = random.choice(variations_desc)
            
            if random.random() > 0.6 and "lumiere" in variation_config:
                variations_lumiere = [
                    "lumière d'après-midi tamisée par la canopée",
                    "éclairage dramatique avec des rayons crépusculaires",
                    "lueur douce et diffuse d'une serre futuriste"
                ]
                variation_config["lumiere"] = random.choice(variations_lumiere)
            
            if random.random() > 0.5 and "ambiance" in variation_config:
                variations_ambiance = [
                    "mystérieuse et enchantée",
                    "sereine et méditative",
                    "nostalgique et mélancolique"
                ]
                variation_config["ambiance"] = random.choice(variations_ambiance)
            
            variations.append(self.optimiser_pour_grok(self.generer_prompt(variation_config)))
        
        return variations
    
    def generer_prompt(self, config: Optional[Dict[str, Any]] = None) -> str:
        """Génère un prompt de base à partir de la configuration"""
        if config is None:
            config = self.config
            
        elements = []

        # 1. Sujet principal + description
        sujet = config.get("sujet_principal", "")
        if config.get("description_sujet"):
            sujet += ", " + config["description_sujet"]
        if sujet:
            elements.append(sujet)

        # 2. Contexte / scène
        if config.get("contexte"):
            elements.append(config['contexte'])

        # 3. Composition
        if config.get("composition"):
            elements.append(f"composition: {config['composition']}")

        # 4. Détails visuels
        if config.get("détails_visuels"):
            elements.append(f"détails: {config['détails_visuels']}")

        # 5. Éléments symboliques
        if config.get("éléments_symboliques"):
            elements.append(f"symbolisme: {config['éléments_symboliques']}")

        # 6. Lumière
        if config.get("lumiere"):
            elements.append(f"éclairage: {config['lumiere']}")

        # 7. Couleurs
        if config.get("couleurs"):
            elements.append(f"couleurs: {config['couleurs']}")

        # 8. Ambiance
        if config.get("ambiance"):
            elements.append(f"ambiance: {config['ambiance']}")

        # 9. Style artistique + mouvement
        styles = []
        if config.get("style_artistique"):
            styles.append(config["style_artistique"])
        if config.get("mouvement_artistique"):
            styles.append(config["mouvement_artistique"])
        if styles:
            elements.append("style: " + ", ".join(styles))

        # 10. Qualité technique
        if config.get("qualité"):
            elements.append(f"technique: {config['qualité']}")

        # 11. Négations
        if config.get("interdits"):
            interdit = config["interdits"]
            prohibitions = interdit.replace("pas de", "sans").replace("pas", "sans")
            elements.append(f"exclure: {prohibitions}")

        # 12. Ratio d'aspect
        if config.get("ratio_aspect"):
            elements.append(f"ratio: {config['ratio_aspect']}")

        # Assemblage fluide
        prompt = self.assembler_elements(elements)
        prompt = re.sub(r",\s*,", ", ", prompt)
        prompt = re.sub(r"\s+", " ", prompt).strip()
        
        # Ajout au historique
        self.historique.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "config": config.copy()
        })
        
        return prompt
    
    def assembler_elements(self, elements: List[str]) -> str:
        """Assemble les éléments du prompt de manière naturelle"""
        if not elements:
            return ""
        prompt = elements[0]
        for i, element in enumerate(elements[1:], 1):
            if element.startswith(('composition:', 'détails:', 'symbolisme:', 'éclairage:', 
                                 'couleurs:', 'ambiance:', 'style:', 'technique:', 'exclure:', 'ratio:')):
                prompt += ", " + element
            else:
                connecteurs = [", ", ", ", ", ", " avec ", " et ", " dans "]
                prompt += random.choice(connecteurs) + element
        return prompt
    
    def optimiser_pour_grok(self, prompt: str) -> str:
        """Optimise le prompt pour le générateur d'images de Grok 3"""
        # Ajouter des paramètres spécifiques à Grok 3
        optimized_prompt = prompt + ", optimisé pour Grok 3, rendu ultra-détaillé, résolution maximale (jusqu'à 8K), style artistique compatible avec l'analyse d'images, profondeur de champ cinématographique, couleurs vives et harmonieuses"
        
        # S'assurer que les interdictions sont claires pour Grok
        if "exclure" in prompt:
            optimized_prompt = optimized_prompt.replace("exclure", "éviter strictement")
        
        # Ajuster le style pour correspondre aux capacités de Grok
        if "style" in prompt and any(s in prompt for s in ["Ghibli", "Sugimori", "ukiyo-e"]):
            optimized_prompt += ", rendu inspiré de l'animation japonaise traditionnelle et des artworks Pokémon"
        
        return optimized_prompt
    
    def exporter_historique(self, fichier: str = "historique_prompts.json"):
        """Exporte l'historique des prompts générés"""
        with open(fichier, 'w', encoding='utf-8') as f:
            json.dump(self.historique, f, ensure_ascii=False, indent=2)
    
    def suggerer_améliorations(self) -> List[str]:
        """Suggère des améliorations basées sur la configuration actuelle"""
        suggestions = []
        if not self.config.get("détails_visuels"):
            suggestions.append("Ajouter des détails visuels spécifiques pour plus de précision")
        if not self.config.get("lumiere"):
            suggestions.append("Définir un type d'éclairage pour renforcer l'ambiance")
        if not self.config.get("éléments_symboliques"):
            suggestions.append("Envisager un élément symbolique pour ajouter une dimension narrative")
        if len(self.config.get("qualité", "")) < 10:
            suggestions.append("Préciser les détails techniques (résolution, moteur de rendu, etc.)")
        return suggestions

# === Interface en ligne de commande améliorée ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Générateur de prompts optimisés pour Grok 3")
    parser.add_argument("--config", "-c", default="DPrompt.json", help="Fichier de configuration JSON")
    parser.add_argument("--variations", "-v", type=int, help="Nombre de variations à générer")
    parser.add_argument("--export", "-e", action="store_true", help="Exporter l'historique")
    parser.add_argument("--suggestions", "-s", action="store_true", help="Afficher des suggestions d'amélioration")
    
    args = parser.parse_args()
    
    # Initialisation du générateur
    generateur = PromptGenerator(args.config)
    
    # Génération du prompt principal optimisé
    prompt_principal = generateur.optimiser_pour_grok(generateur.generer_prompt())
    
    print("🎨 PROMPT OPTIMISÉ POUR GROK 3 (DALL·E Pro Ultra) :\n")
    print(prompt_principal + "\n")
    
    # Suggestions d'amélioration
    if args.suggestions:
        suggestions = generateur.suggerer_améliorations()
        if suggestions:
            print("💡 SUGGESTIONS D'AMÉLIORATION :")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
            print()
    
    # Génération de variations optimisées
    nb_variations = args.variations if args.variations else generateur.config.get("variation", 1)
    if nb_variations > 1:
        print(f"🔄 VARIATIONS OPTIMISÉES POUR GROK 3 ({nb_variations} versions) :\n")
        variations = generateur.generer_variations(nb_variations)
        for i, variation in enumerate(variations, 1):
            print(f"Version {i}:")
            print(variation + "\n")
    
    # Export de l'historique
    if args.export:
        generateur.exporter_historique()
        print("📁 Historique exporté dans 'historique_prompts.json'")
    
    print("✨ Utilisez ce prompt avec le générateur d'images de Grok 3 pour des résultats optimaux !")