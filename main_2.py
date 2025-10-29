import torch
import json
import subprocess
import numpy as np
import os
import time

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_model

from performance import J_high # J_low는 이 예제에서 사용하지 않음


D = 1
VAR_NAME = 'displacement'
PHYSICAL_BOUNDS = (0.1, 0.3) # 단위: m

# Torch에서 사용할 수 있도록 정규화된 경계값 텐서 생성 ([0, 1] 범위)
bounds_normalized = torch.tensor([[0.0], [1.0]], dtype=torch.double)

def denormalize_scalar(normalized_value):
    """[0, 1] 스칼라를 물리적 값으로 변환"""
    p_min, p_max = PHYSICAL_BOUNDS
    return normalized_value * (p_max - p_min) + p_min

def run_abaqus_and_evaluate_hf(displacement_value):
    """단일 설계 변수 'displacement'를 받아 Abaqus HF 해석을 실행하고 평가"""
    design_params = {VAR_NAME: displacement_value}
    params_str = json.dumps(design_params)
    
    command = ['abaqus', 'python', 'abaqus_runner.py', '--params', params_str, '--fidelity', 'high']
    
    try:
        print(f"  Running Abaqus for displacement = {displacement_value:.4f} m...")
        start_time = time.time()
        # Abaqus 해석 시간은 길 수 있으므로 타임아웃을 넉넉하게 설정
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600) # 1시간
        end_time = time.time()
        print(f"  ...Abaqus finished in {end_time - start_time:.2f} seconds.")
        
        odb_path = result.stdout.strip().split('\n')[-1]
        
        if not odb_path or not odb_path.endswith('.odb'):
             print(f"  Error: Invalid ODB path received: {odb_path}")
             return None

        return J_high(odb_path)

    except Exception as e:
        print(f"  !!! Abaqus run or evaluation failed: {e} !!!")
        return None

def objective_function_sf(normalized_x: torch.Tensor):
    """
    성능 손실(1 - J'_High)을 반환하여, 이 값을 최대화하는 것을 목표로 함.
    """
    displacement_normalized = normalized_x[0].item()
    
    displacement_physical = denormalize_scalar(displacement_normalized)
    
    j_prime_high = run_abaqus_and_evaluate_hf(displacement_physical)
    
    if j_prime_high is None:
        return torch.tensor([[-1e6]], dtype=torch.double)
        
    performance_loss = 1.0 - j_prime_high
    
    return torch.tensor([[performance_loss]], dtype=torch.double)

if __name__ == "__main__":
    
    INITIAL_POINTS = 5  # 초기 샘플링 개수
    NUM_ITERATIONS = 15 # BO 반복 횟수
    
    print("="*60)
    print("Standard Bayesian Optimization Test for Wrinkle Analysis")
    print("="*60)
    print(f"Goal: Find displacement that MAXIMIZES performance loss (1 - J'_High)")
    print(f"Generating {INITIAL_POINTS} initial data points...")

    # 초기 데이터 생성 (랜덤 샘플링)
    train_X_normalized = torch.rand(INITIAL_POINTS, D, dtype=torch.double)
    train_Y = torch.zeros(INITIAL_POINTS, 1, dtype=torch.double)

    for i in range(INITIAL_POINTS):
        print(f"\n--- Initial Point {i+1}/{INITIAL_POINTS} ---")
        train_Y[i] = objective_function_sf(train_X_normalized[i])

    print("\nInitial data generation complete.")

    # --- 2. 최적화 루프 ---
    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- BO Iteration {iteration+1}/{NUM_ITERATIONS} ---")
        
        # 모델 학습
        model = SingleTaskGP(train_X_normalized, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        
        acq_func = ExpectedImprovement(
            model=model,
            best_f=train_Y.max().item(), # 현재까지의 최댓값
            maximize=True # 우리는 성능 손실을 '최대화'할 것임
        )
        
        new_point_normalized, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_normalized,
            q=1, # 한 번에 한 포인트씩 추천
            num_restarts=5,
            raw_samples=512,
        )
        
        new_y = objective_function_sf(new_point_normalized)
        
        train_X_normalized = torch.cat([train_X_normalized, new_point_normalized], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)
        
        best_loss, best_idx = torch.max(train_Y, dim=0)
        print(f"Best performance loss found so far: {best_loss.item():.6f}")

    best_loss, best_idx = torch.max(train_Y, dim=0)
    best_design_normalized = train_X_normalized[best_idx]
    best_design_physical = denormalize_scalar(best_design_normalized.item())
    
    final_j_prime_high = 1.0 - best_loss.item()

    print("\n" + "="*60)
    print("Optimization Finished")
    print("="*60)
    print(f"Optimal displacement found: {best_design_physical:.4f} m")
    print(f"Maximum performance loss: {best_loss.item():.6f} ({(best_loss.item()*100):.2f}%)")
    print(f"Corresponding J'_High at this point: {final_j_prime_high:.6f}")