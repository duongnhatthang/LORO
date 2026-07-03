from utils import create_d3rlpy_model


def test_sac_uses_configured_learning_rate():
    model = create_d3rlpy_model(
        "Pendulum-v1", batch_size=256, learning_rate=5e-5, gamma=0.99,
        target_update_interval=1000, gpu=False, model_type="default",
    )
    cfg = model.config
    assert cfg.actor_learning_rate == 5e-5
    assert cfg.critic_learning_rate == 5e-5
