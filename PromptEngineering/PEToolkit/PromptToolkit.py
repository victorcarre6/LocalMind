"""
Prompt Engineering Toolkit (PET) - Un système complet pour la création, transformation et gestion de prompts
Version compatible Python 3.8 et 3.9
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple, Union
import inspect
from functools import reduce
import json

@dataclass
class Prompt:
    """
    Structure principale pour représenter un prompt avec tous ses composants
    """
    instruction: str
    context: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_format: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_example(self, example: str) -> None:
        """Ajoute un exemple au prompt"""
        self.examples.append(example)

    def add_constraint(self, constraint: str) -> None:
        """Ajoute une contrainte au prompt"""
        self.constraints.append(constraint)

    def format(self, include_metadata: bool = False) -> str:
        """
        Formatte le prompt en une chaîne de caractères prête à être utilisée
        
        Args:
            include_metadata: Si True, inclut les métadonnées dans le formatage
        """
        parts = []
        
        if self.instruction:
            parts.append(f"## Instruction\n{self.instruction}")
        
        if self.context:
            parts.append(f"\n## Context\n{self.context}")
        
        if self.examples:
            parts.append("\n## Examples\n" + "\n".join(f"- {ex}" for ex in self.examples))
        
        if self.constraints:
            parts.append("\n## Constraints\n" + "\n".join(f"- {c}" for c in self.constraints))
        
        if self.output_format:
            parts.append(f"\n## Output Format\n{self.output_format}")
        
        if include_metadata and self.metadata:
            parts.append(f"\n## Metadata\n{json.dumps(self.metadata, indent=2)}")
        
        return "\n".join(parts)

    def validate(self) -> bool:
        """Valide que le prompt est bien formé"""
        basic_validation = (
            bool(self.instruction) and 
            len(self.instruction) >= 10 and
            any(verb in self.instruction.lower() for verb in ["classify", "analyze", "describe", "generate", "write"])
        )
        
        examples_validation = (
            not self.examples or
            all("->" in ex or ":" in ex for ex in self.examples)
        )
        
        return basic_validation and examples_validation

    def copy(self) -> 'Prompt':
        """Crée une copie du prompt"""
        return Prompt(
            instruction=self.instruction,
            context=self.context,
            examples=self.examples.copy(),
            constraints=self.constraints.copy(),
            output_format=self.output_format,
            metadata=self.metadata.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le prompt en dictionnaire"""
        return {
            "instruction": self.instruction,
            "context": self.context,
            "examples": self.examples,
            "constraints": self.constraints,
            "output_format": self.output_format,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Prompt':
        """Crée un prompt à partir d'un dictionnaire"""
        return cls(
            instruction=data.get("instruction", ""),
            context=data.get("context"),
            examples=data.get("examples", []),
            constraints=data.get("constraints", []),
            output_format=data.get("output_format"),
            metadata=data.get("metadata", {})
        )

# ======================================
# Transformations de prompts
# ======================================

def prompt_transformation(func: Callable[..., Prompt]) -> Callable[..., Prompt]:
    """Décorateur pour enregistrer les transformations"""
    func.is_transformation = True
    return func

@prompt_transformation
def make_formal(prompt: Prompt) -> Prompt:
    """Rend le prompt plus formel"""
    new_prompt = prompt.copy()
    if not new_prompt.instruction.endswith('.'):
        new_prompt.instruction += '.'
    new_prompt.instruction += " Please use formal language."
    new_prompt.add_constraint("No colloquial language")
    return new_prompt

@prompt_transformation
def capitalize_examples(prompt: Prompt) -> Prompt:
    """Capitalise tous les exemples"""
    new_prompt = prompt.copy()
    new_prompt.examples = [ex.capitalize() for ex in new_prompt.examples]
    return new_prompt

@prompt_transformation
def add_explanation_requirement(prompt: Prompt, explanation_phrase: str = "Explain your reasoning:") -> Prompt:
    """Ajoute une demande d'explication"""
    new_prompt = prompt.copy()
    if new_prompt.output_format:
        new_prompt.output_format += f"\n{explanation_phrase}"
    else:
        new_prompt.output_format = explanation_phrase
    return new_prompt

# ======================================
# Pipelines et utilitaires
# ======================================

def pipeline(*transformations: Callable[[Prompt], Prompt]) -> Callable[[Prompt], Prompt]:
    """
    Crée une pipeline de transformations
    
    Args:
        *transformations: Fonctions de transformation à appliquer séquentiellement
    
    Returns:
        Une fonction qui prend un prompt et retourne le prompt transformé
    """
    def apply(prompt: Prompt) -> Prompt:
        return reduce(lambda p, fn: fn(p), transformations, prompt)
    return apply

def generate_variations(prompt: Prompt, variations: List[Callable[[Prompt], Prompt]]) -> List[Prompt]:
    """
    Génère des variations d'un prompt en appliquant différentes transformations
    
    Args:
        prompt: Prompt original
        variations: Liste de fonctions de transformation
    
    Returns:
        Liste des prompts transformés
    """
    return [transformation(prompt.copy()) for transformation in variations]

# ======================================
# Templates de prompts
# ======================================

def classification_template(
    categories: List[str],
    description: str,
    examples: Optional[List[Tuple[str, str]]] = None
) -> Prompt:
    """
    Template pour un prompt de classification
    
    Args:
        categories: Liste des catégories possibles
        description: Description du contexte de classification
        examples: Liste de tuples (input, output) comme exemples
    """
    prompt = Prompt(
        instruction=f"Classify the following text into one of these categories: {', '.join(categories)}",
        context=description,
        constraints=[
            "Select only one category",
            "Be concise but accurate"
        ],
        output_format="Category: <selected_category>\nJustification: <brief_explanation>"
    )
    
    if examples:
        for input_text, output_text in examples:
            prompt.add_example(f"{input_text} -> {output_text}")
    
    return prompt

def text_generation_template(
    style: str,
    requirements: List[str],
    length_constraint: Optional[str] = None
) -> Prompt:
    """
    Template pour un prompt de génération de texte
    
    Args:
        style: Style d'écriture demandé (ex: "formal", "creative")
        requirements: Liste des exigences pour le texte
        length_constraint: Contrainte de longueur (ex: "50 words")
    """
    constraints = [f"Write in {style} style"] + requirements
    if length_constraint:
        constraints.append(f"Length: {length_constraint}")
    
    return Prompt(
        instruction="Generate text according to the requirements below",
        constraints=constraints,
        output_format="Generated text: <your_text>"
    )

# ======================================
# Interface Utilisateur
# ======================================

def interactive_prompt_creator() -> Prompt:
    """Crée un prompt de manière interactive"""
    print("=== Prompt Creator ===")
    instruction = input("Instruction (required): ").strip()
    while not instruction:
        print("Instruction cannot be empty!")
        instruction = input("Instruction (required): ").strip()
    
    context = input("Context (optional, press Enter to skip): ").strip() or None
    output_format = input("Output format (optional, press Enter to skip): ").strip() or None
    
    prompt = Prompt(
        instruction=instruction,
        context=context,
        output_format=output_format
    )
    
    print("\nAdd examples (one per line, enter empty line to finish):")
    while True:
        example = input("Example (input -> output): ").strip()
        if not example:
            break
        prompt.add_example(example)
    
    print("\nAdd constraints (one per line, enter empty line to finish):")
    while True:
        constraint = input("Constraint: ").strip()
        if not constraint:
            break
        prompt.add_constraint(constraint)
    
    print("\n=== Created Prompt ===")
    print(prompt.format())
    
    if not prompt.validate():
        print("\nWarning: The prompt might need improvements (too short or missing action verb)")
    
    return prompt

#Fonctions pour exemples GPT
def information_extraction_template(entity_types: List[str], context: str) -> Prompt:
    return Prompt(
        instruction=f"Extract the following entities from the text: {', '.join(entity_types)}",
        context=context,
        constraints=[
            "List entities exactly as they appear in the text",
            "Do not include duplicates",
            "If an entity is not found, write 'None'"
        ],
        output_format="Entities: <entity_list>"
    )
    
def critical_analysis_template(criteria: List[str], context: str) -> Prompt:
    return Prompt(
        instruction="Analyze the following text according to the specified criteria",
        context=context,
        constraints=[
            f"Address each criterion: {', '.join(criteria)}",
            "Support each point with evidence from the text",
            "Be objective and concise"
        ],
        output_format="\n".join([f"{c}: <analysis>" for c in criteria])
    )
    
def comparison_template(items: List[str], aspects: List[str]) -> Prompt:
    return Prompt(
        instruction=f"Compare the following items: {', '.join(items)}",
        constraints=[
            f"Cover the following aspects: {', '.join(aspects)}",
            "Provide a final summary highlighting the most important differences"
        ],
        output_format="Comparison table:\n<your_table>\n\nSummary:\n<your_summary>"
    )    
    
def multi_level_rephrase_template(levels: List[str]) -> Prompt:
    return Prompt(
        instruction="Rephrase the given text in different levels of complexity",
        constraints=[
            f"Provide versions for: {', '.join(levels)}",
            "Preserve the core meaning in each version",
            "Avoid adding new information"
        ],
        output_format="\n".join([f"{level}: <text_version>" for level in levels])
    )

def brainstorming_template(topic: str, categories: List[str]) -> Prompt:
    return Prompt(
        instruction=f"Generate creative ideas related to: {topic}",
        constraints=[
            f"Provide at least 3 ideas for each category: {', '.join(categories)}",
            "Avoid repeating the same concept",
            "Include both conventional and unconventional ideas"
        ],
        output_format="\n".join([f"{cat}: <idea_list>" for cat in categories])
    )

#Fonctions pour exemples Anthropic
# Nouveaux templates et transformations

def chain_of_thought_template(problem_type: str, context: str) -> Prompt:
    """Template pour raisonnement étape par étape"""
    return Prompt(
        instruction=f"Solve this {problem_type} problem step by step",
        context=context,
        constraints=[
            "Break down the problem into smaller steps",
            "Show your reasoning for each step",
            "Clearly state any assumptions made"
        ],
        output_format="Step 1: <reasoning>\nStep 2: <reasoning>\n...\nFinal Answer: <solution>"
    )

def role_play_template(role: str, scenario: str, objectives: List[str]) -> Prompt:
    """Template pour jeu de rôle avec persona"""
    return Prompt(
        instruction=f"Act as a {role} in the following scenario",
        context=scenario,
        constraints=[
            f"Stay in character as {role} throughout",
            "Use appropriate tone and vocabulary for this role",
            f"Focus on achieving: {', '.join(objectives)}"
        ],
        output_format="Response: <roleplay_response>"
    )

def structured_analysis_template(subject: str, framework: str, dimensions: List[str]) -> Prompt:
    """Template pour analyse structurée (SWOT, 5W1H, etc.)"""
    return Prompt(
        instruction=f"Analyze {subject} using the {framework} framework",
        constraints=[
            f"Address each dimension: {', '.join(dimensions)}",
            "Provide specific examples or evidence",
            "Maintain objectivity and balance"
        ],
        output_format="\n".join([f"{dim}:\n- <point1>\n- <point2>" for dim in dimensions])
    )

def few_shot_learning_template(task: str, examples: List[Tuple[str, str]], pattern_description: str) -> Prompt:
    """Template pour apprentissage few-shot avec exemples"""
    prompt = Prompt(
        instruction=f"Complete the following {task} following the pattern shown in the examples",
        context=f"Pattern: {pattern_description}",
        constraints=[
            "Follow the exact same format as the examples",
            "Maintain consistency with the demonstrated pattern"
        ],
        output_format="Result: <your_completion>"
    )
    
    for input_ex, output_ex in examples:
        prompt.add_example(f"Input: {input_ex}\nOutput: {output_ex}")
    
    return prompt

def debate_template(topic: str, position: str, opposing_arguments: List[str]) -> Prompt:
    """Template pour débat structuré"""
    return Prompt(
        instruction=f"Argue {position} the following topic: {topic}",
        context=f"Consider these opposing arguments: {', '.join(opposing_arguments)}",
        constraints=[
            "Provide at least 3 strong supporting arguments",
            "Address potential counterarguments",
            "Use logical reasoning and evidence",
            "Maintain respectful tone"
        ],
        output_format="Position: <your_stance>\n\nArguments:\n1. <argument1>\n2. <argument2>\n3. <argument3>\n\nCounterargument responses:\n<responses>"
    )

def creative_constraints_template(creative_type: str, mandatory_elements: List[str], forbidden_elements: List[str]) -> Prompt:
    """Template pour créativité sous contraintes"""
    return Prompt(
        instruction=f"Create a {creative_type} that incorporates specific constraints",
        constraints=[
            f"Must include: {', '.join(mandatory_elements)}",
            f"Must NOT include: {', '.join(forbidden_elements)}",
            "Be original and creative within these constraints"
        ],
        output_format=f"{creative_type.capitalize()}: <your_creation>\n\nConstraint compliance:\n<explanation>"
    )

def error_analysis_template(domain: str, error_types: List[str]) -> Prompt:
    """Template pour analyse d'erreurs"""
    return Prompt(
        instruction=f"Analyze the errors in this {domain} content",
        constraints=[
            f"Identify errors in these categories: {', '.join(error_types)}",
            "Provide corrections for each error found",
            "Explain why each correction is necessary"
        ],
        output_format="Error Analysis:\n" + "\n".join([f"{error_type}:\n- Error: <description>\n- Correction: <fix>\n- Reason: <explanation>" for error_type in error_types])
    )

def scenario_planning_template(situation: str, variables: List[str], time_horizon: str) -> Prompt:
    """Template pour planification de scénarios"""
    return Prompt(
        instruction=f"Create multiple scenarios for: {situation}",
        context=f"Time horizon: {time_horizon}",
        constraints=[
            f"Consider these key variables: {', '.join(variables)}",
            "Develop at least 3 distinct scenarios (optimistic, realistic, pessimistic)",
            "Include probability estimates and key assumptions"
        ],
        output_format="Scenario 1 (Optimistic):\n<description>\nProbability: <estimate>\n\nScenario 2 (Realistic):\n<description>\nProbability: <estimate>\n\nScenario 3 (Pessimistic):\n<description>\nProbability: <estimate>"
    )

# === Nouvelles transformations ===

@prompt_transformation
def add_confidence_scoring(prompt: Prompt) -> Prompt:
    """Ajoute une demande de score de confiance"""
    new_prompt = prompt.copy()
    if new_prompt.output_format:
        new_prompt.output_format += "\nConfidence Score (1-10): <score>\nReasoning for confidence: <explanation>"
    else:
        new_prompt.output_format = "Confidence Score (1-10): <score>\nReasoning for confidence: <explanation>"
    return new_prompt

@prompt_transformation
def add_alternative_perspectives(prompt: Prompt, num_perspectives: int = 2) -> Prompt:
    """Demande des perspectives alternatives"""
    new_prompt = prompt.copy()
    new_prompt.add_constraint(f"Provide {num_perspectives} alternative viewpoints or approaches")
    if new_prompt.output_format:
        new_prompt.output_format += f"\n\nAlternative perspectives:\n" + "\n".join([f"Perspective {i+1}: <viewpoint>" for i in range(num_perspectives)])
    return new_prompt

@prompt_transformation
def add_source_requirements(prompt: Prompt, source_types: List[str]) -> Prompt:
    """Exige des sources spécifiques"""
    new_prompt = prompt.copy()
    new_prompt.add_constraint(f"Reference sources from: {', '.join(source_types)}")
    if new_prompt.output_format:
        new_prompt.output_format += "\n\nSources used:\n- <source1>\n- <source2>"
    return new_prompt

@prompt_transformation
def make_socratic(prompt: Prompt) -> Prompt:
    """Transforme en style socratique avec questions"""
    new_prompt = prompt.copy()
    new_prompt.instruction = "Instead of direct answers, guide through thought-provoking questions: " + new_prompt.instruction
    new_prompt.add_constraint("Use questions to lead to discovery rather than providing direct answers")
    return new_prompt


 
