"""Robust action extraction shared by all TranslationAgents.

Parses from the model's GENERATED span only (not the prompt), prefers an explicit
final answer, then a case-insensitive keyword, then a last-resort random-valid action
(flagged and counted so the run can report a parse-failure rate).
"""
import re
import numpy as np

_PARSE_FAILURES = 0


def reset_parse_failures():
    global _PARSE_FAILURES
    _PARSE_FAILURES = 0


def PARSE_FAILURES_GET():
    return _PARSE_FAILURES


def extract_generated_text(full_decoded: str, prompt_text: str) -> str:
    """Return only the newly generated text. Works for ChatML (Qwen/DeepSeek) and Llama.

    Resolution order:
      1. Prefix-strip: if full_decoded starts with prompt_text, strip it.
      2. Special-token markers: look for raw ChatML/Llama delimiters (relevant when
         special tokens are NOT stripped).
      3. Text-level "assistant" split: the ChatML template emits the literal word
         "assistant" (plain text) before the model's turn even after
         skip_special_tokens=True.  Split on the last case-insensitive occurrence;
         if the tail is non-empty, return it stripped of leading whitespace.
      4. Last resort: return full_decoded unchanged (documented fallback).
    """
    # 1) Prefix-strip
    if prompt_text and full_decoded.startswith(prompt_text):
        return full_decoded[len(prompt_text):]
    # 2) Special-token markers (useful when tokens are NOT stripped)
    for marker in ("<|im_start|>assistant", "[/INST]", "<|assistant|>"):
        if marker in full_decoded:
            return full_decoded.split(marker)[-1]
    # 3) Text-level "assistant" split (survives skip_special_tokens=True)
    lower = full_decoded.lower()
    idx = lower.rfind("assistant")
    if idx != -1:
        tail = full_decoded[idx + len("assistant"):]
        if tail.strip():
            return tail.lstrip()
    # 4) Last resort: return the full decoded string unchanged
    return full_decoded


def extract_choice_index(generated_text: str, valid_1based, keyword_map=None):
    """Return (action_1based, is_fallback). valid_1based e.g. [1,2,3,4]."""
    global _PARSE_FAILURES
    valid_str = {str(v) for v in valid_1based}
    # 1) explicit "answer/action: N" or "final answer N"
    m = re.findall(r"(?:answer|action|choice)\D{0,10}(\d+)", generated_text, flags=re.I)
    for cand in reversed(m):
        if cand in valid_str:
            return int(cand), False
    # 2) last valid standalone digit token
    tokens = re.findall(r"\d+", generated_text)
    for cand in reversed(tokens):
        if cand in valid_str:
            return int(cand), False
    # 3) case-insensitive keyword
    if keyword_map:
        low = generated_text.lower()
        best_pos, best_val = -1, None
        for kw, val in keyword_map.items():
            pos = low.rfind(kw.lower())
            if pos > best_pos and val in valid_1based:
                best_pos, best_val = pos, val
        if best_val is not None:
            return best_val, False
    # 4) hard failure -> random valid action, flagged + counted
    _PARSE_FAILURES += 1
    return int(np.random.choice(list(valid_1based))), True
