"""
Model loader for LexProbe API.
Loads the base Llama-3.1-8B in 4-bit and applies the QLoRA adapter.
Provides a single `analyze(text)` function that runs all clause types.
"""

import os
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()

BASE_MODEL_ID  = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH   = os.environ.get("ADAPTER_PATH", "models/llama-3.1-8b-cuad-lora-v2/final_adapter")
HF_TOKEN       = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = 128
MAX_CONTEXT_CHARS = 3000

CLAUSE_TYPES = [
    "Auto-Renewal",
    "Governing Law",
    "Late Payment / Penalty",
    "Limitation of Liability",
    "Non-Compete",
    "Termination for Convenience",
]

# Module-level singletons — loaded once on startup
_tokenizer = None
_model = None


def _build_prompt(clause_type: str, text: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a legal contract analyst. Your job is to determine whether a specific "
        "clause type is present in the provided contract excerpt.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Contract excerpt:\n{text[:MAX_CONTEXT_CHARS]}\n\n"
        f'Question: Does this excerpt contain a "{clause_type}" clause?\n'
        '- If YES: respond with "Found: " followed by the exact relevant text.\n'
        '- If NO: respond with exactly "None".\n\n'
        "Answer:"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def load_model():
    """Load model + adapter into VRAM. Called once at API startup."""
    global _tokenizer, _model

    if _model is not None:
        return  # Already loaded

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set in environment / .env file.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[model] Loading base model {BASE_MODEL_ID} in 4-bit...")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )

    print(f"[model] Applying LoRA adapter from {ADAPTER_PATH}...")
    _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    _model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[model] Ready. VRAM in use: {vram:.2f} GB")


def analyze(text: str) -> list[dict]:
    """
    Run the model across all clause types for the given contract text.

    Returns a list of dicts:
      [{"clause_type": str, "detected": bool, "extracted_text": str | None}]
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    results = []
    for clause_type in CLAUSE_TYPES:
        prompt = _build_prompt(clause_type, text)
        inputs = _tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to("cuda")

        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        generated = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        detected = not generated.lower().startswith("none")
        extracted = None
        if detected:
            # Strip the "Found: " prefix if present
            extracted = generated.removeprefix("Found: ").strip()

        results.append({
            "clause_type": clause_type,
            "detected":    detected,
            "extracted_text": extracted,
        })

    return results
