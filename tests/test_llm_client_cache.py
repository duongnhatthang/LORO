from llm_client import LLMClient


def test_cache_roundtrip(tmp_path):
    c = LLMClient(model="test-model", backend="api", cache_dir=str(tmp_path))
    msgs = [{"role": "user", "content": "hi"}]
    key = c._cache_key(msgs, temperature=0.9, top_p=0.6, max_new_tokens=10, seed=0)
    c._cache_write(key, "cached-response")
    # A cached key must be returned without calling the backend.
    got = c.chat(msgs, temperature=0.9, top_p=0.6, max_new_tokens=10, seed=0)
    assert got == "cached-response"
