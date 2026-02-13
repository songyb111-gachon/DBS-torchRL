import sys
import logging
from datetime import datetime
import os
from utils.logger import setup_logger

# 로거 설정
log_file = setup_logger()

print("이 메시지는 콘솔과 파일에 동시에 기록됩니다.")
logging.info("이 메시지도 로그에 기록됩니다.")

import os
import glob
import shutil
from datetime import datetime
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

import torchOptics.optics as tt
import torchOptics.metrics as tm

from env import BinaryHologramEnv
from models import BinaryNet, Dataset512, IPS, CH
from ppo import ActorCriticPolicy, PPO

warnings.filterwarnings('ignore')

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

torch.backends.cudnn.enabled = False

# ============================================================================
# BinaryNet 모델 초기화 및 로드
# ============================================================================
model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False,
                  convBN=False, poolReLU=False, poolBN=False,
                  deconvReLU=False, deconvBN=False).cuda()
test = torch.randn(1, 1, IPS, IPS).cuda()
out = model(test)
print(out.shape)

# ============================================================================
# 데이터셋 준비
# ============================================================================
batch_size = 1
target_dir = '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/'
valid_dir = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
meta = {'wl': (515e-9), 'dx': (7.56e-6, 7.56e-6)}
padding = 0

train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=True, padding=padding)
valid_dataset = Dataset512(target_dir=valid_dir, meta=meta, isTrain=False, padding=padding)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# ============================================================================
# BinaryNet 사전학습 모델 로드
# ============================================================================
model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False, convBN=False,
                  poolReLU=False, poolBN=False, deconvReLU=False, deconvBN=False).cuda()
model.load_state_dict(torch.load('result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002'))
model.eval()

# ============================================================================
# 환경 생성
# ============================================================================
env = BinaryHologramEnv(
    target_function=model,
    trainloader=train_loader,
    max_steps=10000,
    T_PSNR=30,
    T_steps=1,
    T_PSNR_DIFF=1/4,
    num_samples=10000
)

# ============================================================================
# PPO 설정
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 정책 네트워크 생성
policy = ActorCriticPolicy(features_dim_per_key=64)

# PPO 하이퍼파라미터 (기존 SB3 설정과 동일)
ppo_kwargs = dict(
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
)

# 저장 디렉토리
save_dir = "./ppo_pytorch_models/"
os.makedirs(save_dir, exist_ok=True)

# 모델 로드 또는 새로 생성
ppo_model_path = os.path.join(save_dir, "ppo_latest.pt")
resume_training = True

if resume_training and os.path.exists(ppo_model_path):
    print(f"Loading trained PPO model from {ppo_model_path}")
    ppo = PPO.load(ppo_model_path, policy=policy, device=device, **ppo_kwargs)
else:
    if resume_training:
        print(f"Warning: PPO model not found at {ppo_model_path}. Starting training from scratch.")
    print("Starting training from scratch.")
    ppo = PPO(policy=policy, device=device, **ppo_kwargs)

# ============================================================================
# 학습 루프
# ============================================================================
max_episodes = 8000
total_timesteps = 1000000000

episode_count = 0
global_step = 0
episode_rewards = []

print(f"\n{'='*60}")
print(f"PPO Training Start")
print(f"Max Episodes: {max_episodes}")
print(f"n_steps: {ppo_kwargs['n_steps']}, batch_size: {ppo_kwargs['batch_size']}")
print(f"{'='*60}\n")

try:
    while episode_count < max_episodes and global_step < total_timesteps:
        # 에피소드 시작
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # 한 스텝 수집
            next_obs, done, reward, step_info = ppo.collect_step(obs, env)
            episode_reward += reward
            global_step += 1

            # 버퍼가 가득 차면 업데이트
            if ppo.buffer.is_full():
                metrics = ppo.update()
                print(
                    f"\033[94m[PPO Update #{ppo.n_updates}] "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy Loss: {metrics['entropy_loss']:.4f} | "
                    f"Total Loss: {metrics['total_loss']:.4f}\033[0m"
                )

            obs = next_obs

        # 에피소드 종료
        episode_count += 1
        episode_rewards.append(episode_reward)

        print(f"\033[41mEpisode {episode_count}: Total Reward: {episode_reward:.2f}\033[0m")

        # 주기적으로 모델 저장 (10 에피소드마다)
        if episode_count % 10 == 0:
            ppo.save(ppo_model_path)

        # 에피소드 수 도달 시 종료
        if episode_count >= max_episodes:
            print(f"Stopping training at episode {episode_count}")
            break

except KeyboardInterrupt:
    print("\n학습이 사용자에 의해 중단되었습니다.")

# ============================================================================
# 최종 모델 저장
# ============================================================================
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Start {current_date}_model saving at {save_dir}")
ppo_model_save_path = os.path.join(save_dir, f"ppo_{current_date}.pt")
ppo.save(ppo_model_save_path)
print(f"PPO Model saved at {save_dir}")

# 최신 모델 업데이트
print(f"Start latest_model updating at {save_dir}")
ppo_latest_path = os.path.join(save_dir, "ppo_latest.pt")
if os.path.exists(ppo_latest_path):
    os.remove(ppo_latest_path)
shutil.copyfile(ppo_model_save_path, ppo_latest_path)
print(f"Latest model updated at {ppo_latest_path}")
