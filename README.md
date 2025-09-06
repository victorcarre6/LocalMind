# LLM Assistant — Local Memorization & Chat Interface

> A personal assistant based on local language models (LLM), with **persistent memory** and a **chat-friendly interface**, designed to run without external API calls to ensure data privacy.

> Objective: Provide rich and personalized context for every interaction with the LLM, integrating immediate memory and summarized past conversations.

---

## Key Features

- **Advanced Memorization**  
  - Automatic saving of conversations in a SQLite database.  
  - Clear distinction between:
    - **RECENT EXCHANGES** → flash memory, last exchanges between user and assistant. Prioritized in responses.  
    - **OLDER CONVERSATIONS** → long-term memory, used as secondary context or examples.  
  - Direct storage in the database of a **compressed summary** (`llm_output_summary`) to speed up prompt generation.  
  - Vector search with **FAISS** and keyword filtering.

- **Graphical Interface**  
  - **Chat-style interface**, input box at the bottom.  
  - Buttons: Help, Settings (model and memory configuration), More (data vizualisation).  
  - Supports **streaming responses** (token-by-token display) -WIP-.

- **Model Flexibility**  
  - Easy local model loading (GGUF via llama.cpp) configured in `config.json`.  
  - Simple model switching by updating the path.  
  - Configurable **summarization model** (default: [`Falconsai/text_summarization`](https://huggingface.co/Falconsai/text_summarization)).

- **Optimizations**  
  - Summaries generated **once at insertion**.  
  - Batch summarization for faster insertion.  
  - Automatic removal of `<think>` tags before storing in the database.  

---

## Architecture

- **`scripts/gui.py`**: User interface (chat + settings).  
- **`scripts/llm_executor.py`**: LLM response generation.  
- **`scripts/memorization.py`**: Memory management, summarization, database insertion.  
- **`datas/conversations.db`**: SQLite database containing:
  - `conversations` (inputs, outputs, summaries, timestamps, model info).  
  - `vectors` (keywords and embeddings).  
  - `conversation_vectors` (full user_input embeddings).  
  - `hash_index` (duplicate detection).  
- **`resources/config.json`**: Configuration for paths, models, and memory parameters.  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/victorcarre6/llm-assistant
cd llm-assistant
```

2.	Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3.	Install dependencies:

```bash
pip install -r requirements.txt
```

4.	Download required models:

  •	Summarization model (default HuggingFace):
```bash
python -m spacy download fr_core_news_lg
python -m spacy download en_core_web_lg
```

  •	Local LLM model (GGUF Qwen3 example): configure path in resources/config.json:
```bash
"model_path": "resources/models/Qwen3-0.6B/Qwen3-0.6B-Q5_K_M.gguf"
```

## Usage
Run the GUI:
```bash
python scripts/gui.py
```

	•	Type your question in the input box.
	•	Click the arrow to generate the response.
	•	Conversations are automatically saved in the database with their summaries.
	•	Use the Settings menu to adjust:
	•	Number of contexts (context_count),
	•	Number of keywords (keyword_count),
	•	Immediate memory,
	•	Display of <think> blocks togglable.


## Configuration (config.json)

Minimal example:
```json
{
  "environment": {
    "venv_activate_path": ".venv/bin/activate",
    "conversations_path": "datas/conversations"
  },
  "scripts": {
    "sync_script_path": "scripts/memory_sync.py",
    "llm_script_path": "scripts/llm_executor.py"
  },
  "data": {
    "db_path": "datas/conversations.db",
    "stopwords_file_path": "resources/stopwords_fr.json"
  },
  "models": {
    "summarizing_model": "Falconsai/text_summarization",
    "llm": {
      "model_path": "resources/models/Qwen3-0.6B/Qwen3-0.6B-Q5_K_M.gguf",
      "sampler_params": {
        "seed": 42,
        "top_k": 20,
        "top_p": 0.90,
        "min_p": 0.05,
        "temp": 0.65,
        "penalty_last_n": 64,
        "penalty_repeat": 1.1,
        "mirostat": 0
      },
      "system_prompt": "Answer scientifically prioritizing RECENT EXCHANGES..."
    }
  },
  "memory_parameters": {
    "keyword_multiplier": 2,
    "similarity_threshold": 0.2
  }
}
```

## Memory Management
•	Recent exchanges are always injected into the prompt.
•	Older conversations are retrieved using:
•	Keyword filtering (KeyBERT),
•	FAISS vector search,
•	Summaries stored in the database (llm_output_summary).

Example prompt structure:
```bash
SYSTEM: Answer scientifically...
### RECENT EXCHANGES ###
<|im_start|>user ...
<|im_start|>assistant ...
### OLDER CONVERSATIONS ###
<|im_start|>user ...
<|im_start|>assistant (summary) ...
```

## Performance
•	Summaries are generated once at insertion → fast prompt generation.
•	Fast vector search using FAISS.
•	Ability to limit the number of contexts (context_count) and recent exchanges (max 3 by default).

## Roadmap
•	Optional integration with sqlite-vec for semantic search.
•	Improved reranking of contexts.
•	Support for additional LLM backends (vLLM, Ollama, LM Studio).
•	Interactive dashboard for conversation visualization.


## Contribution

- [easy-llama](https://github.com/ddh0/easy-llama)

Contributions are welcome: issues, pull requests, suggestions.

## 📜 License

MIT — free and open source.