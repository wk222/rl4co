import torch

from rl4co.envs import TSPEnv
from rl4co.models.zoo import AttentionModelPolicy
from rl4co.models.zoo.prefdce.model import PreferenceDCE, build_pref_matrices
from rl4co.utils import RL4COTrainer


def test_pref_matrix_fast_equivalence():
    rewards = torch.randn(1, 8)
    logps = torch.randn(1, 8)
    full, _ = build_pref_matrices(rewards, logps, beta=1.0, alpha=1.0, fast=False)
    fast, _ = build_pref_matrices(rewards, logps, beta=1.0, alpha=1.0, fast=True)
    assert torch.allclose(full, fast, atol=1e-6)


def test_preference_dce_train(tmp_path):
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = PreferenceDCE(
        env,
        policy,
        num_samples=4,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator="cpu")
    trainer.fit(model)
    trainer.test(model)
