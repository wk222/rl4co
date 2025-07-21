import torch

from lightning.pytorch.callbacks import RichModelSummary

from rl4co.envs import TSPEnv
from rl4co.models.zoo import GRPO, AttentionModelPolicy, PreferenceDCE
from rl4co.utils.trainer import RL4COTrainer


def run_experiment(
    model_cls,
    max_epochs: int = 2,
    num_samples: int = 8,
    train_data_size: int = 100_000,
    val_data_size: int = 10_000,
    test_data_size: int = 10_000,
    accelerator: str | None = None,
):
    """Train and test a model on TSP."""
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = model_cls(
        env,
        policy,
        num_samples=num_samples,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        test_data_size=test_data_size,
    )
    if accelerator is None:
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[RichModelSummary(max_depth=3)],
    )
    trainer.fit(model)
    trainer.test(model)


def main():
    run_experiment(GRPO)
    run_experiment(PreferenceDCE)


if __name__ == "__main__":
    main()
