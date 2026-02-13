"""
순수 PyTorch PPO 구현.
stable-baselines3의 PPO + MultiInputPolicy를 대체합니다.

구성요소:
- MultiInputFeatureExtractor: Dict 관측 공간의 각 입력을 CNN/MLP로 처리 후 concat
- ActorCriticPolicy: Actor(Categorical) + Critic(Value)
- RolloutBuffer: 트랜지션 저장 + GAE 계산
- PPO: PPO 클립 손실 기반 정책 업데이트
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


IPS = 256
CH = 8


# ============================================================================
# Feature Extractor (SB3의 MultiInputPolicy 대체)
# ============================================================================

class NatureCNN(nn.Module):
    """
    SB3의 NatureCNN과 유사한 CNN Feature Extractor.
    입력: (B, C, H, W) -> 출력: (B, features_dim)
    """
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

        # 출력 차원 자동 계산
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
    """
    SB3의 CombinedExtractor와 동일한 역할.
    Dict 관측의 각 키별로 별도의 CNN을 사용한 뒤 concat합니다.

    관측 키 및 shape:
        - "state_record": (1, CH, IPS, IPS) -> int8
        - "state":        (1, CH, IPS, IPS) -> int8
        - "pre_model":    (1, CH, IPS, IPS) -> float32
        - "recon_image":  (1, 1, IPS, IPS) -> float32
        - "target_image": (1, 1, IPS, IPS) -> float32
    """
    def __init__(self, features_dim_per_key=64):
        super().__init__()

        self.extractors = nn.ModuleDict({
            "state_record": NatureCNN(CH, features_dim_per_key),
            "state":        NatureCNN(CH, features_dim_per_key),
            "pre_model":    NatureCNN(CH, features_dim_per_key),
            "recon_image":  NatureCNN(1, features_dim_per_key),
            "target_image": NatureCNN(1, features_dim_per_key),
        })

        self.features_dim = features_dim_per_key * 5  # 5개 키

    def forward(self, obs_dict):
        """
        Args:
            obs_dict: dict of str -> Tensor (B, C, H, W)
        Returns:
            features: Tensor (B, features_dim)
        """
        features = []
        for key in ["state_record", "state", "pre_model", "recon_image", "target_image"]:
            x = obs_dict[key]
            # (1, C, H, W) 형태에서 앞의 1을 제거하여 (C, H, W)로 만들고 배치 차원 유지
            if x.dim() == 5:
                # (B, 1, C, H, W) -> (B, C, H, W)
                x = x.squeeze(1)
            elif x.dim() == 4 and x.shape[0] == 1 and x.shape[1] in [CH, 1]:
                # 이미 (B, C, H, W) 형태
                pass
            features.append(self.extractors[key](x.float()))
        return torch.cat(features, dim=-1)


# ============================================================================
# Actor-Critic Policy
# ============================================================================

class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic 정책 네트워크.
    - Actor: Categorical 분포 (Discrete action space, CH*IPS*IPS = 524288 actions)
    - Critic: 스칼라 상태 가치
    """
    def __init__(self, features_dim_per_key=64, net_arch_pi=None, net_arch_vf=None):
        super().__init__()

        self.feature_extractor = MultiInputFeatureExtractor(features_dim_per_key)
        features_dim = self.feature_extractor.features_dim
        num_actions = CH * IPS * IPS  # 524288

        # Actor 네트워크
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

        # Critic 네트워크
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
        """
        Args:
            obs_dict: dict of str -> Tensor
        Returns:
            logits: (B, num_actions)
            value: (B, 1)
        """
        features = self.feature_extractor(obs_dict)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(self, obs_dict, action=None, deterministic=False):
        """
        액션 샘플링 + 가치 평가.

        Args:
            obs_dict: dict 형태 관측값
            action: 이미 선택된 액션 (None이면 새로 샘플링)
            deterministic: True면 argmax, False면 샘플링

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(obs_dict)
        dist = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, obs_dict):
        """가치만 계산"""
        features = self.feature_extractor(obs_dict)
        return self.critic(features)

    def predict(self, obs_dict, deterministic=True):
        """
        추론용 (SB3의 model.predict()와 동일한 역할).
        Returns: action (int)
        """
        with torch.no_grad():
            logits, _ = self.forward(obs_dict)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
        return action.cpu().numpy()


# ============================================================================
# Rollout Buffer
# ============================================================================

class RolloutBuffer:
    """
    PPO용 롤아웃 버퍼.
    n_steps 분량의 트랜지션을 저장하고 GAE를 계산합니다.
    """
    def __init__(self, n_steps, gamma=0.99, gae_lambda=0.95):
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs_list = []       # list of dict
        self.actions = []        # list of int
        self.rewards = []        # list of float
        self.dones = []          # list of bool
        self.log_probs = []      # list of float
        self.values = []         # list of float

        self.advantages = None
        self.returns = None
        self.pos = 0

    def add(self, obs, action, reward, done, log_prob, value):
        """한 스텝의 트랜지션을 추가"""
        self.obs_list.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.pos += 1

    def is_full(self):
        return self.pos >= self.n_steps

    def compute_returns_and_advantages(self, last_value, last_done):
        """
        GAE (Generalized Advantage Estimation) 계산.
        SB3의 compute_returns_and_advantage와 동일한 로직.
        """
        values = self.values + [last_value]
        dones = self.dones + [last_done]

        advantages = np.zeros(self.pos, dtype=np.float32)
        last_gae_lam = 0.0

        for step in reversed(range(self.pos)):
            next_non_terminal = 1.0 - float(dones[step + 1])
            next_values = values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + np.array(self.values[:self.pos], dtype=np.float32)

        self.advantages = advantages
        self.returns = returns

    def get_samples(self, batch_size, device):
        """
        버퍼에서 미니배치를 생성하여 반환하는 제너레이터.
        """
        indices = np.random.permutation(self.pos)

        actions = np.array(self.actions[:self.pos])
        log_probs = np.array(self.log_probs[:self.pos], dtype=np.float32)

        for start in range(0, self.pos, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # 관측값을 배치로 묶기
            batch_obs = {}
            for key in self.obs_list[0].keys():
                batch_obs[key] = torch.tensor(
                    np.array([self.obs_list[i][key] for i in batch_indices]),
                    dtype=torch.float32, device=device
                )

            yield (
                batch_obs,
                torch.tensor(actions[batch_indices], dtype=torch.long, device=device),
                torch.tensor(log_probs[batch_indices], dtype=torch.float32, device=device),
                torch.tensor(self.advantages[batch_indices], dtype=torch.float32, device=device),
                torch.tensor(self.returns[batch_indices], dtype=torch.float32, device=device),
            )

    def reset(self):
        """버퍼 초기화"""
        self.obs_list = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None
        self.pos = 0


# ============================================================================
# PPO Algorithm
# ============================================================================

class PPO:
    """
    순수 PyTorch PPO 알고리즘.
    stable-baselines3의 PPO와 동일한 하이퍼파라미터/로직을 사용합니다.
    """
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

        self.buffer = RolloutBuffer(n_steps, gamma, gae_lambda)

        # 학습 통계
        self.n_updates = 0

    def collect_step(self, obs, env):
        """
        한 스텝의 데이터를 수집합니다.

        Args:
            obs: dict 형태 관측값 (numpy)
            env: 환경 인스턴스

        Returns:
            next_obs, done, reward, info
        """
        # numpy obs를 텐서로 변환
        obs_tensor = self._obs_to_tensor(obs)

        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action_and_value(obs_tensor)

        action_int = action.item()
        log_prob_float = log_prob.item()
        value_float = value.item()

        # 환경 스텝
        next_obs, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated

        # 버퍼에 저장
        self.buffer.add(obs, action_int, float(reward), done, log_prob_float, value_float)

        return next_obs, done, reward, info

    def update(self):
        """
        PPO 정책 업데이트.
        버퍼가 가득 차면 호출됩니다.

        Returns:
            dict: 학습 통계 (loss 등)
        """
        # GAE 계산을 위한 마지막 가치 추정
        # (마지막 관측에 대한 V(s) 는 이미 buffer에 있으므로 0으로 처리)
        # 에피소드가 끝났으면 last_value=0, 아니면 현재 관측의 가치
        last_obs = self.buffer.obs_list[-1]
        last_done = self.buffer.dones[-1]

        if last_done:
            last_value = 0.0
        else:
            obs_tensor = self._obs_to_tensor(last_obs)
            with torch.no_grad():
                last_value = self.policy.get_value(obs_tensor).item()

        self.buffer.compute_returns_and_advantages(last_value, last_done)

        # PPO 업데이트
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss_val = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in \
                    self.buffer.get_samples(self.batch_size, self.device):

                # 새로운 log_prob, entropy, value 계산
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    batch_obs, action=batch_actions
                )
                new_values = new_values.squeeze(-1)

                # Advantage 정규화
                adv = batch_advantages
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy Loss (PPO Clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy Loss
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # 역전파
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss_val += loss.item()
                n_updates += 1

        self.n_updates += 1

        # 버퍼 초기화
        self.buffer.reset()

        if n_updates > 0:
            return {
                "policy_loss": total_policy_loss / n_updates,
                "value_loss": total_value_loss / n_updates,
                "entropy_loss": total_entropy_loss / n_updates,
                "total_loss": total_loss_val / n_updates,
            }
        return {}

    def _obs_to_tensor(self, obs):
        """numpy dict 관측을 GPU 텐서 dict로 변환"""
        obs_tensor = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs_tensor[key] = torch.tensor(val, dtype=torch.float32, device=self.device).unsqueeze(0)
            elif isinstance(val, torch.Tensor):
                obs_tensor[key] = val.float().to(self.device).unsqueeze(0)
            else:
                obs_tensor[key] = torch.tensor(np.array(val), dtype=torch.float32, device=self.device).unsqueeze(0)
        return obs_tensor

    def save(self, path):
        """모델 저장"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_updates': self.n_updates,
        }, path)
        print(f"PPO model saved to {path}")

    @classmethod
    def load(cls, path, policy, device="cuda", **kwargs):
        """모델 로드"""
        ppo = cls(policy=policy, device=device, **kwargs)
        checkpoint = torch.load(path, map_location=device)
        ppo.policy.load_state_dict(checkpoint['policy_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ppo.n_updates = checkpoint.get('n_updates', 0)
        print(f"PPO model loaded from {path}")
        return ppo
