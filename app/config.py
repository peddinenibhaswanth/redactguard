"""Central config. Every tunable that would otherwise be a magic number
scattered across modules lives here, loaded once from environment/.env."""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Spans with confidence below this are still redacted (fail-safe) but flagged
# needs_human_review=True in the report. Single source of truth - Phase 3's
# local model will have different confidence calibration than the API model,
# so this is the one place to retune it.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Which provider the pipeline's llm_detect node calls first.
LLM_DETECTOR_PROVIDER = os.getenv("LLM_DETECTOR_PROVIDER", "gemini")  # "gemini" | "groq"

# Which provider eval/generate_synthetic_data.py uses to author ground-truth
# documents. Deliberately different from LLM_DETECTOR_PROVIDER by default so
# the detector isn't evaluated against its own generation patterns.
EVAL_GENERATOR_PROVIDER = os.getenv("EVAL_GENERATOR_PROVIDER", "groq")  # "gemini" | "groq"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Generic OpenAI-compatible provider - OpenRouter, Together, Mistral,
# Cerebras, a local vLLM, anything speaking /chat/completions. Adding a
# provider is configuration, not code. Left blank the chain simply skips it.
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", os.getenv("CEREBRAS_API_KEY", ""))
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "")

# Order providers are tried in. Every one is attempted before giving up -
# a single fallback was not enough when both free tiers hit their daily
# limits mid-eval and detection silently degraded to regex-only.
PROVIDER_CHAIN = [
    p.strip() for p in os.getenv("PROVIDER_CHAIN", "gemini,groq,openai_compat").split(",") if p.strip()
]

# Cap on redact->verify retries before giving up and flagging for manual handling.
MAX_REDACT_RETRIES = int(os.getenv("MAX_REDACT_RETRIES", "2"))

# Trigger for routing a PDF page to OCR instead of the direct text extractor.
OCR_TEXT_LEN_THRESHOLD = 20

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# Which fine-tuned LoRA adapter app/detection/local_model_detector.py loads.
# Overridable so the SFT-only and SFT+DPO checkpoints can be evaluated
# against each other without editing code.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_ADAPTER_PATH = os.getenv(
    "LOCAL_ADAPTER_PATH", os.path.join(_REPO_ROOT, "phase3_finetune", "final_adapter")
)
