from env.action_parsing import extract_choice_index, extract_generated_text, reset_parse_failures, PARSE_FAILURES_GET


def test_prefers_explicit_final_choice_not_stray_digits():
    # Stray coordinate "3" appears mid-reasoning; the final answer is action 2.
    text = "I am at cell 3, moving toward the goal. Final answer: 2"
    idx, fb = extract_choice_index(text, valid_1based=[1, 2], keyword_map={"left": 1, "right": 2})
    assert idx == 2 and fb is False


def test_keyword_fallback_is_case_insensitive_and_reachable():
    text = "I will go Left to avoid the cliff."   # no digit present
    idx, fb = extract_choice_index(text, valid_1based=[1, 2], keyword_map={"left": 1, "right": 2})
    assert idx == 1 and fb is False


def test_hard_failure_flags_fallback():
    reset_parse_failures()
    idx, fb = extract_choice_index("banana", valid_1based=[1, 2], keyword_map={"left": 1, "right": 2})
    assert fb is True
    assert idx in (1, 2)
    assert PARSE_FAILURES_GET() == 1


def test_generated_text_strips_prompt_for_chatml():
    full = "<|im_start|>system\nchoose 1 or 2<|im_end|>\n<|im_start|>assistant\nAnswer: 2"
    prompt = "<|im_start|>system\nchoose 1 or 2<|im_end|>\n<|im_start|>assistant\n"
    gen = extract_generated_text(full, prompt)
    assert gen.strip() == "Answer: 2"


def test_generated_text_runtime_prefix_strip():
    """Case A: special tokens already stripped; prefix-strip path works."""
    full = "system\nchoose 1 or 2\nassistant\nAnswer: 2"
    prompt = "system\nchoose 1 or 2\nassistant\n"
    gen = extract_generated_text(full, prompt)
    assert gen.strip() == "Answer: 2"


def test_generated_text_runtime_assistant_text_fallback():
    """Case B: special tokens stripped AND prefix-strip fails; 'assistant' text fallback must isolate generated span."""
    full = "system\nchoose 1 or 2\nassistant\nAnswer: 2"
    prompt = "DIFFERENT PROMPT"
    result = extract_generated_text(full, prompt)
    assert "choose 1 or 2" not in result and "2" in result
