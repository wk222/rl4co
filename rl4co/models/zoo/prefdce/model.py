import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.utils.ops import unbatchify


def build_pref_matrices(
    rewards: torch.Tensor,
    logps: torch.Tensor,
    beta: float,
    alpha: float,
    fast: bool = True,
):
    """Compute DCE advantages.

    Args:
        rewards: [batch, m] reward tensor
        logps: [batch, m] log likelihood tensor
        beta: scaling for reward differences
        alpha: scaling for logprob differences
        fast: whether to use the O(m) per sample method
    Returns:
        advantage: [batch, m] tensor
        diff: [batch, m, m] full difference matrix if fast=False else None
    """
    if fast:
        B, m = rewards.shape
        advantage = torch.zeros_like(rewards)
        for k in range(m):
            Rdiff_k = rewards[:, k].unsqueeze(-1) - rewards
            Ldiff_k = logps[:, k].unsqueeze(-1) - logps
            q_k = torch.sigmoid(beta * Rdiff_k)
            p_k = torch.sigmoid(alpha * Ldiff_k)
            diff = p_k - q_k
            diff[:, k] = 0
            advantage[:, k] = 2 * diff.sum(dim=1)
        return advantage, None

    Rdiff = rewards.unsqueeze(-1) - rewards.unsqueeze(-2)
    Ldiff = logps.unsqueeze(-1) - logps.unsqueeze(-2)
    q = torch.sigmoid(beta * Rdiff)
    p = torch.sigmoid(alpha * Ldiff)
    diff = p - q
    diff.diagonal(dim1=-2, dim2=-1).zero_()
    advantage = 2 * diff.sum(dim=-1)
    return advantage, diff


class PreferenceDCE(REINFORCE):
    """REINFORCE variant with pairwise preference cross entropy."""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        num_samples: int = 8,
        alpha: float = 1.0,
        beta: float = 1.0,
        fast_dce: bool = True,
        **kwargs,
    ):
        super().__init__(env, policy, baseline="no", **kwargs)
        self.num_samples = num_samples
        self.alpha = alpha
        self.beta = beta
        self.fast_dce = fast_dce

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
        dataloader_idx: int | None = None,
    ):
        td = self.env.reset(batch)
        out = self.policy(td, self.env, phase=phase, num_samples=self.num_samples)
        if phase == "train":
            out["loss"] = self.calculate_loss(td, batch, out)
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: torch.Tensor | None = None,
        log_likelihood: torch.Tensor | None = None,
    ):
        reward = policy_out["reward"]
        log_likelihood = policy_out["log_likelihood"]
        reward = unbatchify(reward, self.num_samples)
        log_likelihood = unbatchify(log_likelihood, self.num_samples)
        advantage, _ = build_pref_matrices(
            reward, log_likelihood, self.beta, self.alpha, fast=self.fast_dce
        )
        loss = -(advantage * log_likelihood).mean()
        return loss
