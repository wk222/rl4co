import torch

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import TSPEnv
from rl4co.models.zoo import GRPO, AttentionModelPolicy
from rl4co.utils.trainer import RL4COTrainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = GRPO(
        env, policy, num_samples=8, train_data_size=100_000, val_data_size=10_000
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", filename="epoch_{epoch:03d}", save_last=True
    )
    callbacks = [checkpoint_callback, RichModelSummary(max_depth=3)]
    logger = WandbLogger(project="rl4co", name="grpo-tsp")

    trainer = RL4COTrainer(
        max_epochs=2,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
