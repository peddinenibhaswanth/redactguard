"""Single source of truth for the SFT/DPO prompt format, shared by
prepare_sft_dataset.py, prepare_dpo_dataset.py, and (at inference time)
app/detection/local_model_detector.py. ChatML format (<|im_start|>/<|im_end|>)
matches Qwen2.5-Instruct's actual chat template, so this is written directly
rather than built through a tokenizer that isn't available outside Colab/the
downloaded model.
"""
SYSTEM_MSG = (
    "You are a PII detection specialist. Given a text span from a document, "
    "classify whether it contains sensitive information."
)

PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Document context: "{context}"
Span to classify: "{span_text}"<|im_end|>
<|im_start|>assistant
"""


def format_prompt(context: str, span_text: str) -> str:
    return PROMPT_TEMPLATE.format(system=SYSTEM_MSG, context=context, span_text=span_text)


def format_full_example(context: str, span_text: str, label_json: str) -> str:
    return format_prompt(context, span_text) + label_json + "<|im_end|>"


def context_window(text: str, start: int, end: int, window: int = 80) -> str:
    return text[max(0, start - window) : min(len(text), end + window)]
