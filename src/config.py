import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"missing env {key}")
    return val

OPENAI_API_KEY = _env("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

PINECONE_API_KEY = _env("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")

SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _env("SUPABASE_SERVICE_KEY")

DROPBOX_APP_SECRET = _env("DROPBOX_APP_SECRET")

PROMPTS_PATH = str(Path(__file__).with_name("prompts.yml"))
