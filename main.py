# main_optimizer.py
import torch
import json
import subprocess
import numpy as np
# ... (BoTorch 관련 import는 이전과 동일) ...
from utils import denormalize_vector
from performance_calculator import calculate_J_prime_high_from_odb, calculate_J_prime_low_from_odb

d = 5 # 설계 변수 차원

def evaluate_design_via_subprocess(design_params, fidelity_level):
    """
    외부 Python 스크립트를 호출하여 Abaqus를 실행하고 결과를 파싱합니다.
    """
    # 설계 변수를 JSON 문자열로 변환하여 명령행으로 전달
    params_str = json.dumps(design_params)
    
    fidelity_str = 'high' if fidelity_level == 1 else 'low'
    
    command = [
        'abaqus', 'python', 'abaqus_runner.py', # Abaqus 환경의 Python으로 실행
        '--params', params_str,
        '--fidelity', fidelity_str
    ]
    
    try:
        # subprocess 실행 및 표준 출력 캡처
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # abaqus_runner.py가 출력한 마지막 줄(odb 경로)을 파싱
        odb_path = result.stdout.strip().split('\n')[-1]
        
        if not odb_path or not odb_path.endswith('.odb'):
             print("Error: ODB path not received from runner.")
             return None

        print(f"Successfully received ODB path: {odb_path}")
        return odb_path

    except subprocess.CalledProcessError as e:
        print(f"!!! Abaqus runner script failed for fidelity '{fidelity_str}' !!!")
        print(f"Stderr: {e.stderr}")
        return None

def objective_function_mf(normalized_x: torch.Tensor):
    # ... (이전과 동일한 루프 구조) ...
    for i, x in enumerate(normalized_x):
        # ... (design_params 생성까지 동일) ...
        
        # Abaqus 실행 명령
        odb_path = evaluate_design_via_subprocess(design_params, fidelity_level)

        if odb_path is None: # 해석 실패 시
            if fidelity_level == 0: # LF
                output[i] = -1e6 # 최소화 문제의 벌점 -> 최대화 문제에서는 큰 음수
            else: # HF
                output[i] = -1e6
            continue

        # 결과 파일로부터 성능 계산
        if fidelity_level == 0: # LF
            j_prime_low = calculate_J_prime_low_from_odb(odb_path)
            output[i] = -j_prime_low # 최소화 -> 최대화
        else: # HF
            j_prime_high = calculate_J_prime_high_from_odb(odb_path)
            output[i] = j_prime_high
            
    return output

# --- 메인 최적화 루프 ---
# ... (이전 코드와 동일: 초기 데이터 생성, 루프, 모델 학습, 획득 함수, 결과 분석 등) ...
if __name__ == "__main__":
    # 이 부분에 BoTorch 최적화 루프 코드를 넣습니다.
    # ...
    pass