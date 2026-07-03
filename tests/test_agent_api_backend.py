from llamagym.agent import Agent


class _FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature, top_p, max_new_tokens, seed):
        self.calls.append(dict(messages=list(messages), temperature=temperature,
                               top_p=top_p, max_new_tokens=max_new_tokens, seed=seed))
        return "  Action: 2  "


class _TinyAgent(Agent):
    def get_system_prompt(self):
        return "sys"

    def format_observation(self, observation):
        return f"obs={observation}"

    def extract_action(self, response):
        return int(response.split(":")[-1])


def _make_agent():
    return _TinyAgent(
        model=None, tokenizer=None, device="cpu",
        generate_config_dict={
            "generate/max_new_tokens": 256, "generate/temperature": 0.9,
            "generate/top_p": 0.6, "generate/top_k": 0, "generate/do_sample": True,
        },
        is_sft=False,
    )


def test_llm_api_backend_routes_to_client_and_strips():
    a = _make_agent()
    fake = _FakeClient()
    a.backend = "api"
    a.llm_client = fake
    resp = a.llm([{"role": "user", "content": "hi"}])
    assert resp == "Action: 2"                      # whitespace stripped
    assert fake.calls[0]["max_new_tokens"] == 256   # sampling cfg threaded through
    assert fake.calls[0]["temperature"] == 0.9
    assert fake.calls[0]["top_p"] == 0.6


def test_act_api_backend_end_to_end():
    a = _make_agent()
    a.backend = "api"
    a.llm_client = _FakeClient()
    action = a.act(observation=5)
    assert action == 2
    # user message + assistant response were recorded (system prompt is seeded)
    roles = [m["role"] for m in a.current_episode_messages]
    assert roles == ["system", "user", "assistant"]


def test_default_backend_is_local():
    a = _make_agent()
    assert a.backend == "local"
    assert a.llm_client is None
