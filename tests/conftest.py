import os
import sys
from pathlib import Path

# Keep the suite hermetic: do not load .env and do not inherit real provider
# credentials, so tests exercise the deterministic fallback rather than hitting
# a live LLM API. Must run before src.settings is imported by any test.
os.environ["DISABLE_DOTENV"] = "1"
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ["LLM_PROVIDER"] = "anthropic"
os.environ["RUN_RECORDS_DIR"] = ""

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
