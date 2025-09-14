# LocalMind — A chat interface to discuss with local models with persistent memory

<p align="center">
  <img src="https://img.shields.io/github/stars/victorcarre6/llm-assistant?style=social" />
  <img src="https://img.shields.io/github/forks/victorcarre6/llm-assistant?style=social" />
  <img src="https://img.shields.io/github/license/victorcarre6/llm-assistant" />
</p>
<p align="center">
  <a href="https://ko-fi.com/victorcarre">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" />
  </a>
</p>

> **LocalMind** is a chat interface allowing the use of local LLM with persistent memory, running fully offline for privacy.
> This project is an expansion of a previous project, [LLM Memorization](https://github.com/victorcarre6/llm-memorization), allowing automatic saves and summaries of your conversations in a local database, to provide relevant context in every chat.
> It is built on top of [easy-llama](https://github.com/ddh0/easy-llama), a lightweight Python backend that makes it seamless to interact with local models via llama.cpp.

<p align="center">
<img width="1198" height="706" alt="image" src="https://github.com/user-attachments/assets/d23b47da-7d5b-4f94-a332-caf2220083b0" />
</p>
---
## Features

### **Memory System**
The dual-memory architecture allows the assistant to deliver a contextually rich and coherent responses:
  - **Short-term memory** captures recent exchanges, ensuring the assistant maintains immediate context and continuity in conversations.
  - **Long-term memory** stores older conversations in a summarized form, allowing the assistant to recall past information without overwhelming the prompt size.
  - Summaries (`llm_output_summary`) are generated once at insertion for efficiency and stored directly for fast prompt retrieval.
  - Supports advanced retrieval methods including vector search and keyword filtering to access relevant long-term memories.
  - Conversations are automatically saved in the database.

<p align="center">
<img width="1122" height="593" alt="image" src="https://github.com/user-attachments/assets/209896b7-8e25-4c49-b4ac-2236659e49bd" />
</p>

### **Workspaces**
Context-specific memory management:
  - Each workflow represents a distinct context/project within the assistant, with its own long-term memory, maintaining relevant context for each specific use case.
  - Adding, editing, or deleting workflows directly affects the conversations stored in the database (specific memory can be shared or deleted).
  - Supports ephemeral mode, where conversations are not saved, for temporary testing or sensitive queries.

<p align="center">
<img width="545" height="630" alt="image" src="https://github.com/user-attachments/assets/1dc7639a-0e56-4d4c-929d-36d366f85a24" />
</p>

### **Memory and model tuning**
This assistant is designed to be fully customizable by the user.
  - Tune the number of recent exchanges included in the prompt to balance context and model input size.
  - Adjust the number and minimal relevance of long-term memory retrieved.
  - Enable or disable reasoning mode if the model used supports it, and choose to show the thinking block.
  - Configure model parameters directly from the settings menu for flexibility and experimentation.
  - Swap the local model by editing the `model_path` in `resources/config.json`, making it easy to experiment with different `.gguf` models.

<p align="center">
<img width="545" height="680" alt="image" src="https://github.com/user-attachments/assets/72703ec0-77bb-4083-8254-5e42deb69bc0" />
</p>

### **Data visualization**
Use the **Info** button to open a detailed window displaying:
  - Recent exchanges and older conversations.
  - Summaries (`llm_output_summary`) stored in the database.
  - Keywords from your original input and from the transitory generated prompt.
  - A heatmap correlation graph, showing semantic similarity in the transitory prompt.
  - Exchanges (`user_input` and `llm_output`) used as contexts, sorted by their `similarity_score`.
  - General information about your memory database.

<p align="center">
<img width="1476" height="671" alt="image" src="https://github.com/user-attachments/assets/3120ec49-c2bd-45ee-93f6-12244c3a0b04" />
</p>

### **Choice of local LLM**
Due to technical limitations, this project and its benchmark were developed using a small 0.6B parameter model (`Qwen3-0.6B`).
This project has been designed to work with any local model quantized in a `.gguf` format.
I would be **highly interested** to get results using heavier models.

---
## Architecture Overview

- `scripts/gui.py`: Chat UI & settings
- `scripts/llm_executor.py`: LLM response generation
- `scripts/main.py`: Memory, summarization, DB logic
- `datas/conversations_example.db`: An example database containing exchanges about high-level chemistry (En/Fr) using multiple models.
  - `conversations`: inputs, outputs, summaries, timestamps
  - `vectors`: keywords, embeddings
  - `conversation_vectors`: full user input embeddings
  - `hash_index`: duplicate detection
- `resources/config.json`: Paths, models, memory parameters

---
## Installation

1. **Clone the repo**
```bash
git clone https://github.com/victorcarre6/llm-assistant
cd llm-assistant
```

2. **Create a virtual environment**
```bash
python3 -m venv .venv

source .venv/bin/activate # macOS/Linux
.venv\Scripts\activate # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download models**

- Summarization (HuggingFace):
```bash
python -m spacy download fr_core_news_lg
python -m spacy download en_core_web_lg
```

- Local LLM (.GGUF):
Set path in `resources/config.json`:

```bash
"model_path": "resources/models/model_name/model_name.gguf"
```

For initial testing, I advise to use `Qwen3-0.6B` as it was the model used to develop this project.

5. **Run the assistant**

```bash
python scripts/gui.py
```

---

## Coming soon

- Profiles import and export, to selectively exchange memory between users
- Agentic mode :
  - Document integration
  - Web search

---

## Contribution and support

This project has been made (tremendously!) easier thanks to [easy-llama](https://github.com/ddh0/easy-llama).

Contributions are welcome! Feel free to reach out for issues or suggestions.
You can support my work on [ko-fi](https://ko-fi.com/victorcarre)

---
## 📜 License

MIT — free and open source.
