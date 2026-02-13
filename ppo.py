"""
순수 PyTorch PPO 구현 (Full GPU).
모든 데이터가 GPU 텐서로 처리됩니다. CPU-GPU 전송 없음.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler


IPS = 256
CH = 8


# ============================================================================
# Feature Extractor
# ============================================================================

class NatureCNN(nn.Module):
    def __init__(self, in_channels, features_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, IPS, IPS)
            n_flatten = self.cnn(dummy).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(self.cnn(x))


class MultiInputFeatureExtractor(nn.Module):
    def __init__(self, features_dim_per_key=64):
        super().__init__()
        self.extractors = nn.ModuleDict({
            "state_record": NatureCNN(CH, features_dim_per_key),
            "state":        NatureCNN(CH, features_dim_per_key),
            "pre_model":    NatureCNN(CH, features_dim_per_key),
            "recon_image":  NatureCNN(1, features_dim_per_key),
            "target_image": NatureCNN(1, features_dim_per_key),
        })
        self.features_dim = features_dim_per_key * 5

    def forward(self, obs_dict):
        features = []
        for key in ["state_record", "state", "pre_model", "recon_image", "target_image"]:
            x = obs_dict[key]
            if x.dim() == 5:
                x = x.squeeze(1)
            features.append(self.extractors[key](x.float()))
        return torch.cat(features, dim=-1)


# ============================================================================
# Actor-Critic Policy
# ============================================================================

class ActorCriticPolicy(nn.Module):
    def __init__(self, features_dim_per_key=64, net_arch_pi=None, net_arch_vf=None):
        super().__init__()
        self.feature_extractor = MultiInputFeatureExtractor(features_dim_per_key)
        features_dim = self.feature_extractor.features_dim
        num_actions = CH * IPS * IPS

        if net_arch_pi is None:
            net_arch_pi = [256, 256]
        pi_layers = []
        prev_dim = features_dim
        for hidden_dim in net_arch_pi:
            pi_layers.append(nn.Linear(prev_dim, hidden_dim))
            pi_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        pi_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*pi_layers)

        if net_arch_vf is None:
            net_arch_vf = [256, 256]
        vf_layers = []
        prev_dim = features_dim
        for hidden_dim in net_arch_vf:
            vf_layers.append(nn.Linear(prev_dim, hidden_dim))
            vf_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        vf_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*vf_layers)

    def forward(self, obs_dict):
        features = self.feature_extractor(obs_dict)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, obs_dict, action=None, deterministic=False):
        logits, value = self.forward(obs_dict)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs_dict):
        features = self.feature_extractor(obs_dict)
        return self.critic(features)

    def predict(self, obs_dict, deterministic=True):
        with torch.no_grad():
            logits, _ = self.forward(obs_dict)
            action = logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()
        return action


# ============================================================================
# Rollout Buffer (Full GPU)
# ============================================================================

OBS_KEYS = ["state_record", "state", "pre_model", "recon_image", "target_image"]

class RolloutBuffer:
    """
    PPO 롤아웃 버퍼 (Full GPU).
    모든 데이터를 GPU 텐서로 저장. numpy 전환 없음.
    """
    def __init__(self, n_steps, device="cuda", gamma=0.99, gae_lambda=0.95):
        self.n_steps = n_steps
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs_list = []       # list of dict{str: GPU tensor}
        self.actions = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.rewards = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.values = torch.zeros(n_steps, dtype=torch.float32, device=device)

        self.advantages = None
        self.returns = None
        self.pos = 0

    def add(self, obs, action, reward, done, log_prob, value):
        """obs는 GPU 텐서 dict. clone하여 저장."""
        cloned_obs = {k: v.clone() for k, v in obs.items()}
        self.obs_list.append(cloned_obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.pos += 1

    def is_full(self):
        return self.pos >= self.n_steps

    def compute_returns_and_advantages(self, last_value, last_done):
        """GAE 계산 (GPU 텐서)"""
        advantages = torch.zeros(self.pos, device=self.device)
        last_gae_lam = 0.0

        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1].item()
                next_values = self.values[step + 1].item()

            delta = self.rewards[step].item() + self.gamma * next_values * next_non_terminal - self.values[step].item()
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        self.advantages = advantages
        self.returns = advantages + self.values[:self.pos]

    def get_samples(self, batch_size):
        """미니배치 생성 (모든 데이터 GPU)"""
        indices = torch.randperm(self.pos, device=self.device)

        # 관측값을 key별로 미리 stack (1회만 수행)
        stacked_obs = {}
        for key in OBS_KEYS:
            stacked_obs[key] = torch.stack([self.obs_list[i][key] for i in range(self.pos)])
            # (n_steps, 1, C, H, W) → squeeze dim=1 불필요 (get_samples에서 배치로 사용)

        for start in range(0, self.pos, batch_size):
            end = min(start + batch_size, self.pos)
            batch_idx = indices[start:end]

            batch_obs = {key: stacked_obs[key][batch_idx] for key in OBS_KEYS}

            yield (
                batch_obs,
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx],
            )

    def reset(self):
        self.obs_list = []
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.advantages = None
        self.returns = None
        self.pos = 0


# ============================================================================
# PPO Algorithm (Full GPU + AMP)
# ============================================================================

class PPO:
    def __init__(
        self,
        policy: ActorCriticPolicy,
        device="cuda",
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.9,
        learning_rate=1e-4,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        use_amp=True,
    ):
        self.policy = policy.to(device)
        self.device = device

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.buffer = RolloutBuffer(n_steps, device, gamma, gae_lambda)
        self.n_updates = 0

    def collect_step(self, obs, env):
        """obs는 GPU 텐서 dict. 배치 차원 추가 후 policy에 전달."""
        obs_batched = {k: v.unsqueeze(0) if v.dim() == 4 else v for k, v in obs.items()}

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    action, log_prob, entropy, value = self.policy.get_action_and_value(obs_batched)
            else:
                action, log_prob, entropy, value = self.policy.get_action_and_value(obs_batched)

        action_int = action.item()
        log_prob_float = log_prob.float().item()
        value_float = value.float().item()

        next_obs, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated

        self.buffer.add(obs, action_int, float(reward), done, log_prob_float, value_float)

        return next_obs, done, reward, info

    def update(self):
        last_obs = self.buffer.obs_list[-1]
        last_done = self.buffer.dones[self.buffer.pos - 1].item() > 0.5

        if last_done:
            last_value = 0.0
        else:
            obs_batched = {k: v.unsqueeze(0) if v.dim() == 4 else v for k, v in last_obs.items()}
            with torch.no_grad():
                if self.use_amp:
                    with autocast():
                        last_value = self.policy.get_value(obs_batched).float().item()
                else:
                    last_value = self.policy.get_value(obs_batched).item()

        self.buffer.compute_returns_and_advantages(last_value, last_done)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss_val = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in \
                    self.buffer.get_samples(self.batch_size):

                with autocast(enabled=self.use_amp):
                    _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                        batch_obs, action=batch_actions
                    )
                    new_values = new_values.squeeze(-1).float()
                    new_log_probs = new_log_probs.float()
                    entropy = entropy.float()

                    adv = batch_advantages
                    if len(adv) > 1:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(new_values, batch_returns)
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss_val += loss.item()
                n_updates += 1

        self.n_updates += 1
        self.buffer.reset()

        if n_updates > 0:
            return {
                "policy_loss": total_policy_loss / n_updates,
                "value_loss": total_value_loss / n_updates,
                "entropy_loss": total_entropy_loss / n_updates,
                "total_loss": total_loss_val / n_updates,
            }
        return {}

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'n_updates': self.n_updates,
        }, path)
        print(f"PPO model saved to {path}")

    @classmethod
    def load(cls, path, policy, device="cuda", **kwargs):
        ppo = cls(policy=policy, device=device, **kwargs)
        checkpoint = torch.load(path, map_location=device)
        ppo.policy.load_state_dict(checkpoint['policy_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            ppo.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        ppo.n_updates = checkpoint.get('n_updates', 0)
        print(f"PPO model loaded from {path}")
        return ppo
