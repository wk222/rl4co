import copy

import torch
import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.utils.ops import unbatchify


class GRPO(REINFORCE):
    """Group Rewarded PPO with simple normalized advantages."""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        num_samples: int = 8,
        clip_range: float = 0.2,
        entropy_lambda: float = 0.0,
        max_grad_norm: float | None = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(env, policy, baseline="no", **kwargs)
        self.save_hyperparameters(logger=False)
        self.num_samples = num_samples
        self.clip_range = clip_range
        self.entropy_lambda = entropy_lambda
        self.max_grad_norm = max_grad_norm
        self.policy_old = copy.deepcopy(self.policy)
        self.automatic_optimization = False

    def shared_step(
        self, batch, batch_idx: int, phase: str, dataloader_idx: int | None = None
    ):
        td = self.env.reset(batch)
        if phase == "train":
            with torch.no_grad():
                out_old = self.policy_old(
                    td, self.env, phase=phase, num_samples=self.num_samples
                )
            out_new = self.policy(
                td,
                self.env,
                phase=phase,
                num_samples=self.num_samples,
                actions=out_old["actions"],
            )
            reward = unbatchify(out_old["reward"], self.num_samples)
            logp_old = unbatchify(out_old["log_likelihood"], self.num_samples)
            logp_new = unbatchify(out_new["log_likelihood"], self.num_samples)

            adv = reward - reward.mean(dim=1, keepdim=True)
            adv = adv / (reward.std(dim=1, keepdim=True) + 1e-8)
            ratio = torch.exp(logp_new - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
            surrogate_loss = -torch.min(surr1, surr2).mean()
            entropy = -(logp_new.exp() * logp_new).sum(dim=-1).mean()
            loss = surrogate_loss - self.entropy_lambda * entropy

            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            if self.max_grad_norm is not None:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.max_grad_norm,
                    gradient_clip_algorithm="norm",
                )
            opt.step()
            self.policy_old.load_state_dict(self.policy.state_dict())

            out = {
                "loss": loss.detach(),
                "surrogate_loss": surrogate_loss.detach(),
                "entropy": entropy.detach(),
            }
        else:
            out = self.policy(td, self.env, phase=phase, num_samples=self.num_samples)
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
