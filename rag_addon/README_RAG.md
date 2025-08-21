# Dracula RAG Addon (Ollama-first)

Files to drop into your repo root:

- `src/rag.py` — TF-IDF retriever + local Ollama / optional HF backend
- `demo_rag_cli.py` — tiny CLI tester
- `app_rag.py` — Gradio QA demo (optional)
- `src/__init__.py`, `scripts/__init__.py` — package markers

## Quickstart (local, offline-capable)

1) **Install Ollama** (Windows installer), then pull a small instruct model:
   ```powershell
   ollama pull llama3.2:3b
   ```
   Alternatives: `qwen2.5:1.5b-instruct`, `tinyllama:1.1b` (even smaller).

2) **Activate your env** and ensure deps:
   ```powershell
   conda activate dracula_env
   python -m pip install requests
   ```

3) **Ask a question (CLI):**
   ```powershell
   python demo_rag_cli.py --q "What happens in Jonathan Harker's first entry?" --backend ollama --model llama3.2:3b
   ```

4) **Optional UI:**
   ```powershell
   python app_rag.py
   ```

### Verify Ollama is up
```powershell
# Should list models (or be empty if none pulled yet)
Invoke-WebRequest http://localhost:11434/api/tags | Select-Object -Expand Content
```
If unreachable, start the server: `ollama serve` (or restart the Ollama app).

### Fully offline?
Yes—after the model is pulled once and your Python deps are installed, set `--backend ollama` and you can disconnect from the internet.

### Tuning
- Use `--topk 3..6` for passage count.
- Swap model via `--model`, e.g. `qwen2.5:1.5b-instruct`.
- Edit the prompt in `build_prompt()` to change tone/citation style.
