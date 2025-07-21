from rl4co.envs import TSPEnv
from rl4co.models.zoo import GRPO, AttentionModelPolicy
from rl4co.utils import RL4COTrainer


def test_grpo_train(tmp_path):
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = GRPO(
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
