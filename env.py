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
from torch.utils.data import Dataset, DataLoader

import torchvision

import torchOptics.optics as tt
import torchOptics.metrics as tm

IPS = 256  # 이미지 픽셀 사이즈
CH = 8     # 채널
RW = 800   # 보상

warnings.filterwarnings('ignore')


class BinaryHologramEnv:
    """
    Binary Hologram 환경 클래스.
    gymnasium 의존성 없이 순수 Python으로 구현.
    reset()과 step() 인터페이스는 기존과 동일하게 유지.
    """
    def __init__(self, target_function, trainloader, max_steps=10000, T_PSNR=30, T_steps=1, T_PSNR_DIFF=1/4, num_samples=10000):

        # 행동 공간 크기: 픽셀 하나를 선택하는 인덱스 (CH * IPS * IPS)
        self.num_pixels = CH * IPS * IPS

        # 타겟 함수와 데이터 로더 설정
        self.target_function = target_function
        self.trainloader = trainloader

        # 환경 설정
        self.max_steps = max_steps
        self.T_PSNR = T_PSNR
        self.T_steps = T_steps
        self.T_PSNR_DIFF_o = T_PSNR_DIFF
        self.T_PSNR_DIFF = None
        self.num_samples = num_samples
        self.target_step = self.T_PSNR_DIFF_o * self.num_samples

        # 학습 상태 초기화
        self.state = None
        self.state_record = None
        self.observation = None
        self.steps = None
        self.psnr_sustained_steps = None
        self.flip_count = None
        self.start_time = None
        self.next_print_thresholds = 0
        self.total_start_time = None
        self.target_image_np = None
        self.initial_psnr = None

        # 최고 PSNR_DIFF 추적 변수
        self.max_psnr_diff = float('-inf')

        # PSNR 저장 변수
        self.previous_psnr = None

        # 데이터 로더에서 첫 배치 설정
        self.data_iter = iter(self.trainloader)
        self.target_image = None

        # 에피소드 카운트
        self.episode_num_count = 0

    def _calculate_pixel_importance(self, binary, z):
        """랜덤으로 1만 개 픽셀 플립 후 PSNR 변화량을 순위화 및 양수 변화량 총합 계산"""
        num_samples = self.num_samples
        psnr_changes = []
        positive_psnr_sum = 0  # 양수 PSNR 변화량 총합 초기화

        for _ in range(num_samples):
            random_action = np.random.randint(self.num_pixels)
            channel, pixel_index = divmod(random_action, IPS * IPS)
            row, col = divmod(pixel_index, IPS)

            # 픽셀 플립
            self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]

            binary_temp = torch.tensor(self.state, dtype=torch.float32).cuda()
            binary_temp = tt.Tensor(binary_temp, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})

            sim_temp = tt.simulate(binary_temp, z).abs() ** 2
            result_temp = torch.mean(sim_temp, dim=1, keepdim=True)
            psnr_temp = tt.relativeLoss(result_temp, self.target_image, tm.get_PSNR)

            psnr_change = psnr_temp - self.initial_psnr
            psnr_changes.append(psnr_change)

            # 양수 PSNR 변화량만 누적
            if psnr_change > 0:
                positive_psnr_sum += psnr_change

            # 상태 롤백
            self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]

        step_poly = np.array([num_samples, num_samples*90/100, num_samples*80/100, num_samples*50/100, num_samples*25/100, 1])
        rewards_poly = np.array([-0.5, -0.48, -0.45, -0.35, 0, 1])
        degree_poly = len(step_poly) - 1  # degree = 5
        coefficients_poly = np.polyfit(step_poly, rewards_poly, degree_poly)
        poly_reward = np.poly1d(coefficients_poly)

        # 기본값(내장 문자열 표현)으로 다항식 보상 함수 출력
        print("Polynomial Reward Function Equation:")
        print(poly_reward)

        # PSNR 변화량을 기준으로 순위 매기기 (오름차순 정렬)
        sorted_indices = np.argsort(psnr_changes)
        importance_ranks = np.zeros(num_samples)

        for rank, idx in enumerate(sorted_indices):
            x_val = num_samples - (num_samples - 1) * (rank / (num_samples - 1))
            importance_ranks[idx] = poly_reward(x_val)

        return psnr_changes, importance_ranks, positive_psnr_sum

    def reset(self, seed=None, options=None, z=2e-3):
        torch.cuda.empty_cache()

        self.episode_num_count += 1

        # 이터레이터에서 다음 데이터를 가져옴
        try:
            self.target_image, self.current_file = next(self.data_iter)
        except StopIteration:
            print(f"\033[40;93m[INFO] Reached the end of dataset. Restarting from the beginning.\033[0m")
            self.data_iter = iter(self.trainloader)
            self.target_image, self.current_file = next(self.data_iter)

        print(f"\033[40;93m[Episode Start] Currently using dataset file: {self.current_file}, Episode count: {self.episode_num_count}\033[0m")

        self.target_image = self.target_image.cuda()
        self.target_image_np = self.target_image.cpu().numpy()

        with torch.no_grad():
            model_output = self.target_function(self.target_image)
        self.observation = model_output.cpu().numpy()  # (1, CH, IPS, IPS)

        # 매 에피소드마다 초기화
        self.max_psnr_diff = float('-inf')
        self.steps = 0
        self.flip_count = 0
        self.psnr_sustained_steps = 0
        self.next_print_thresholds = 0

        self.state = (self.observation >= 0.5).astype(np.int8)  # 초기 Binary state
        self.state_record = np.zeros_like(self.state)  # 플립 횟수를 저장하기 위한 배열 초기화

        binary = torch.tensor(self.state, dtype=torch.float32).cuda()
        binary = tt.Tensor(binary, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})

        # 시뮬레이션
        sim = tt.simulate(binary, z).abs()**2
        result = torch.mean(sim, dim=1, keepdim=True)

        # MSE 및 PSNR 계산
        mse = tt.relativeLoss(result, self.target_image, F.mse_loss).detach().cpu().numpy()
        self.initial_psnr = tt.relativeLoss(result, self.target_image, tm.get_PSNR)
        self.previous_psnr = self.initial_psnr

        # 1만 개 픽셀 플립 후 PSNR 변화량 순위화 및 양수 변화량 총합 계산
        rw_start_time = time.time()
        self.psnr_change_list, self.importance_ranks, positive_psnr_sum = self._calculate_pixel_importance(binary, z)
        data_processing_time = time.time() - rw_start_time
        print(
            f"\nTime taken for psnr_change_list: {data_processing_time:.2f} seconds"
        )

        self.T_PSNR_DIFF = self.T_PSNR_DIFF_o * positive_psnr_sum
        print(f"\033[94m[Dynamic Threshold] T_PSNR_DIFF set to: {self.T_PSNR_DIFF:.6f}\033[0m")

        obs = {"state_record": self.state_record,
               "state": self.state,
               "pre_model": self.observation,
               "recon_image": result.cpu().numpy(),
               "target_image": self.target_image_np,
               }

        print(
            f"\033[92mInitial PSNR: {self.initial_psnr:.6f}\033[0m"
            f"\nInitial MSE: {mse:.6f}\033[0m"
        )

        # 다음 출력 기준 PSNR 값 리스트 설정 (0.01 단위로 증가)
        self.next_print_thresholds = [self.initial_psnr + i * 0.01 for i in range(1, 21)]

        self.total_start_time = time.time()

        return obs, {"state": self.state}

    def step(self, action, z=2e-3):
        self.steps += 1

        # 행동을 기반으로 픽셀 좌표 계산
        channel = action // (IPS * IPS)
        pixel_index = action % (IPS * IPS)
        row = pixel_index // IPS
        col = pixel_index % IPS

        # 상태 변경
        self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]
        self.state_record[0, channel, row, col] = self.state_record[0, channel, row, col] + 1

        self.flip_count += 1

        # 시뮬레이션 수행
        binary_after = torch.tensor(self.state, dtype=torch.float32).cuda()
        binary_after = tt.Tensor(binary_after, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})
        sim_after = tt.simulate(binary_after, z).abs()**2
        result_after = torch.mean(sim_after, dim=1, keepdim=True)
        psnr_after = tt.relativeLoss(result_after, self.target_image, tm.get_PSNR)

        obs = {"state_record": self.state_record,
               "state": self.state,
               "pre_model": self.observation,
               "recon_image": result_after.cpu().numpy(),
               "target_image": self.target_image_np,
               }

        # PSNR 변화량 계산
        psnr_change = psnr_after - self.previous_psnr
        psnr_diff = psnr_after - self.initial_psnr

        # 가장 유사한 PSNR 변화량의 순위 점수를 보상으로 사용
        closest_index = np.argmin(np.abs(np.array(self.psnr_change_list) - psnr_change))
        reward = self.importance_ranks[closest_index]

        # psnr_change가 음수인 경우 상태 롤백 수행
        if psnr_change < 0:
            self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]
            self.flip_count -= 1

            return obs, reward, False, False, {}

        self.max_psnr_diff = max(self.max_psnr_diff, psnr_diff)

        success_ratio = self.flip_count / self.steps if self.steps > 0 else 0

        # 출력 추가 (0.01 PSNR 상승마다 출력)
        while self.next_print_thresholds and psnr_after >= self.next_print_thresholds[0]:
            self.next_print_thresholds.pop(0)
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )

        self.previous_psnr = psnr_after

        if psnr_diff >= self.T_PSNR_DIFF or (psnr_after >= self.T_PSNR and psnr_diff < 0.1):
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )
            self.psnr_sustained_steps += 1

            if self.psnr_sustained_steps >= self.T_steps and psnr_diff >= self.T_PSNR_DIFF:
                # 스텝에 따른 추가 보상 계산 (선형 보상)
                m = -1000 / (3 * self.target_step)
                additional_reward = 100 + m * (self.steps - (2 / 5) * self.target_step)
                reward += additional_reward

        if self.steps >= self.max_steps:
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )
            # 스텝에 따른 추가 보상 계산 (선형 보상)
            m = -1000 / (3 * self.target_step)
            additional_reward = 100 + m * (self.steps - (2 / 5) * self.target_step)
            reward += additional_reward

        # 성공 종료 조건
        terminated = self.steps >= self.max_steps or self.psnr_sustained_steps >= self.T_steps
        truncated = self.steps >= self.max_steps

        return obs, reward, terminated, truncated, {}
