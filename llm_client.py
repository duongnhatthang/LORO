import os
import json
import hashlib


class LLMClient:
    def __init__(self, model, backend="api", cache_dir="data/llm_cache"):
        self.model = model
        self.backend = backend
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, messages, temperature, top_p, max_new_tokens, seed):
        payload = json.dumps({
            "model": self.model, "messages": messages, "temperature": temperature,
            "top_p": top_p, "max_new_tokens": max_new_tokens, "seed": seed,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key):
        return os.path.join(self.cache_dir, key + ".json")

    def _cache_read(self, key):
        p = self._cache_path(key)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)["response"]
        return None

    def _cache_write(self, key, response):
        with open(self._cache_path(key), "w") as f:
            json.dump({"response": response}, f)

    def chat(self, messages, temperature=0.9, top_p=0.6, max_new_tokens=2000, seed=0):
        key = self._cache_key(messages, temperature, top_p, max_new_tokens, seed)
        cached = self._cache_read(key)
        if cached is not None:
            return cached
        if self.backend == "api":
            resp = self._call_api(messages, temperature, top_p, max_new_tokens, seed)
        else:
            resp = self._call_local(messages, temperature, top_p, max_new_tokens, seed)
        self._cache_write(key, resp)
        return resp

    def _call_api(self, messages, temperature, top_p, max_new_tokens, seed):
        from openai import OpenAI
        client = OpenAI(base_url=os.environ["LLM_API_BASE"], api_key=os.environ["LLM_API_KEY"])
        out = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
            top_p=top_p, max_tokens=max_new_tokens, seed=seed,
        )
        return out.choices[0].message.content

    def _call_local(self, messages, temperature, top_p, max_new_tokens, seed):
        raise NotImplementedError("Local backend wired in llm_main.py's existing transformers path")
