import torch
import json
import subprocess
import numpy as np
import os
import time

from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
import gpytorch.mlls.exact_marginal_log_likelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.fit import fit_gpytorch_model

from performance import J_high, J_low


D = 5 
VAR_ORDER = ['slit_x', 'slit_y', 'slit_len', 'slit_angle', 'clamp_r']
DESIGN_SPACE_BOUNDS = {
    'slit_x': (0.2, 0.8),
    'slit_y': (0.2, 0.8),
    'slit_len': (0.5, 2.0),
    'slit_angle': (0.0, 90.0),
    'clamp_r': (0.5, 1.5)
}
bounds = torch.tensor([[DESIGN_SPACE_BOUNDS[var][0] for var in VAR_ORDER],
                       [DESIGN_SPACE_BOUNDS[var][1] for var in VAR_ORDER]], dtype=torch.double)
bounds_normalized = torch.zeros_like(bounds)
bounds_normalized[1] = 1.0


def denormalize_vector(normalized_vector):
    """[0, 1] 벡터를 물리적 값의 딕셔너리로 변환"""
    physical_dict = {}
    for i, var_name in enumerate(VAR_ORDER):
        n_val = normalized_vector[i]
        p_min, p_max = DESIGN_SPACE_BOUNDS[var_name]
        p_val = n_val * (p_max - p_min) + p_min
        physical_dict[var_name] = p_val
    return physical_dict

def abaqus_evaluate(design_params, fidelity_level):
    params_str = json.dumps(design_params)
    fidelity_str = 'high' if fidelity_level == 1 else 'low'
    
    command = ['abaqus', 'python', 'abaqus_runner.py', '--params', params_str, '--fidelity', fidelity_str]
    
    try:
        print(f"  Running Abaqus for fidelity '{fidelity_str}'...")
        start_time = time.time()
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800) # 30분 타임아웃
        end_time = time.time()
        print(f"  ...Abaqus finished in {end_time - start_time:.2f} seconds.")
        
        odb_path = result.stdout.strip().split('\n')[-1]
        
        if not odb_path or not odb_path.endswith('.odb'):
             print(f"  Error: Invalid ODB path received: {odb_path}")
             return -1e6 if fidelity_level == 1 else 1e6 # 벌점

        # 성능 평가
        if fidelity_level == 1:
            return J_high(odb_path)
        else:
            return -J_low(odb_path) # LF는 최소화 문제이므로 부호 반전

    except subprocess.TimeoutExpired:
        print(f"  !!! Abaqus job timed out for fidelity '{fidelity_str}' !!!")
        return -1e6 if fidelity_level == 1 else 1e6
    except subprocess.CalledProcessError as e:
        print(f"  !!! Abaqus runner script failed for fidelity '{fidelity_str}' !!!")
        # print(f"  Stderr: {e.stderr}") # 디버깅 시 주석 해제
        return -1e6 if fidelity_level == 1 else 1e6

def objective_function_mf(normalized_x: torch.Tensor):
    """
    BoTorch가 호출하는 메인 목적 함수. 각 행에 대해 Abaqus 실행 및 평가를 수행.
    """
    output = torch.zeros(normalized_x.shape[0], 1, dtype=torch.double)

    for i, x in enumerate(normalized_x):
        design_vars_normalized = x[:-1]
        fidelity_level = int(x[-1].item())
        
        design_params = denormalize_vector(design_vars_normalized.numpy())
        
        print(f"Evaluating point {i+1}/{normalized_x.shape[0]}: Fidelity {fidelity_level}, Params: {design_params}")
        
        performance_score = abaqus_evaluate(design_params, fidelity_level)
        output[i] = performance_score
        
    return output

if __name__ == "__main__":
    
    INITIAL_POINTS = 10
    NUM_ITERATIONS = 20
    
    print("="*60)
    print("Multi-Fidelity Bayesian Optimization for Solar Sail Wrinkle Engineering")
    print("="*60)
    print(f"Generating {INITIAL_POINTS} initial data points...")

    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=D)
        train_x_normalized = torch.tensor(sampler.random(n=INITIAL_POINTS), dtype=torch.double)
    except ImportError:
        print("SciPy not found, using random sampling for initial points.")
        train_x_normalized = torch.rand(INITIAL_POINTS, D, dtype=torch.double)

    # 초기 포인트에 대해 HF와 LF를 모두 평가
    train_x_hf = torch.cat([train_x_normalized, torch.full((INITIAL_POINTS, 1), 1.0, dtype=torch.double)], dim=1)
    train_x_lf = torch.cat([train_x_normalized, torch.full((INITIAL_POINTS, 1), 0.0, dtype=torch.double)], dim=1)

    train_y_hf = objective_function_mf(train_x_hf)
    train_y_lf = objective_function_mf(train_x_lf)

    # 데이터 합치기
    train_X = torch.cat([train_x_hf, train_x_lf], dim=0)
    train_Y = torch.cat([train_y_hf, train_y_lf], dim=0)
    
    print("\nInitial data generation complete.")

    # --- 최적화 루프 ---
    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration+1}/{NUM_ITERATIONS} ---")
        
        # --- 모델 학습 ---
        model = SingleTaskMultiFidelityGP(
            train_X, 
            train_Y, 
            outcome_transform=Standardize(m=1),
            data_fidelity_feature=-1
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        print("Model fitting complete.")

        # --- 다음 탐색 지점 결정 ---
        cost_model = {0: 1.0, 1: 20.0} 
        
        # 현재까지의 HF 결과 중 최상의 값
        current_best_hf_value = train_Y[train_X[:, -1] == 1].max().item()
        
        acq_func = qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=128,
            current_value=current_best_hf_value,
            cost_aware=True,
            target_fidelities={-1: 1} # 최종 목표는 HF 성능
        )
        
        print("Optimizing acquisition function to find next point...")
        acqf_bounds = torch.cat([bounds_normalized, torch.tensor([[0.0],[1.0]], dtype=torch.double)], dim=1)

        new_point_normalized, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=acqf_bounds.T,
            q=1,
            num_restarts=10,
            raw_samples=1024,
        )
        print("Next point selected.")

        new_y = objective_function_mf(new_point_normalized)
        
        train_X = torch.cat([train_X, new_point_normalized], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)
        
        best_value_so_far = train_Y[train_X[:, -1] == 1].max().item()
        print(f"Best J'_High found so far: {best_value_so_far:.6f}")

    best_idx = torch.argmax(train_Y[train_X[:, -1] == 1])
    # HF 데이터 포인트들만 필터링
    hf_points = train_X[train_X[:, -1] == 1]
    
    best_design_normalized = hf_points[best_idx, :-1]
    best_design_physical = denormalize_vector(best_design_normalized.numpy())

    print("\n" + "="*60)
    print("Optimization Finished")
    print("="*60)
    print(f"Best design (normalized): {best_design_normalized.numpy()}")
    print(f"Best design (physical): {best_design_physical}")
    print(f"Best performance (J'_High): {best_value_so_far:.6f}")
