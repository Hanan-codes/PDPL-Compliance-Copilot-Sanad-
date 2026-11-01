
## Quick start

**Requirements:** Python 3.10+, about 8 GB RAM, internet access on first run (retrieval models are downloaded from Hugging Face), and one LLM provider (Gemini API key, or Ollama).

```bash
git clone https://github.com/<your-user>/sanad.git
cd sanad
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Configure a provider.**

- *Gemini:* copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `GEMINI_API_KEY`. Never commit this file.
- *Ollama:* install Ollama, then run `ollama pull qwen2.5:7b` and `ollama serve`.

**Run.**

```bash
streamlit run app.py
```

In the sidebar, choose a provider and click **Connect / Initialise**. The first start loads the models and can take several minutes.

## Project structure
