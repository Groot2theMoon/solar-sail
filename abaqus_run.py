# abaqus_runner.py
import os
import sys
import json
import argparse
import subprocess

def create_inp_file(template_path, params, job_name):
    """입력 파일 템플릿으로부터 새로운 .inp 파일 생성"""
    with open(template_path, 'r') as f:
        template = f.read()
    inp_content = template.format(**params)
    inp_filename = f"{job_name}.inp"
    with open(inp_filename, 'w') as f:
        f.write(inp_content)
    return inp_filename

def main():
    # 1. 명령행 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True, help='Design parameters as a JSON string')
    parser.add_argument('--fidelity', type=str, required=True, choices=['low', 'high'], help='Fidelity level')
    args = parser.parse_args()
    
    design_params = json.loads(args.params)
    
    # 2. 잡 이름 및 파일 경로 설정
    # 고유한 잡 이름을 위해 ID나 해시값 사용 추천
    job_id = hash(args.params) 
    job_name = f"sail_{args.fidelity}_{job_id}"
    
    template_path = f"template_{args.fidelity}.inp" # LF/HF용 템플릿 분리
    
    # 3. .inp 파일 생성
    inp_file = create_inp_file(template_path, design_params, job_name)
    print(f"Generated {inp_file} for fidelity {args.fidelity}")

    # 4. Abaqus 커맨드 실행
    # 이 스크립트는 이미 Abaqus Python으로 실행되므로, 직접 abaqus 명령을 호출
    # (주의: 시스템 환경에 따라 경로를 명시해야 할 수 있음)
    command = ['abaqus', 'job=' + job_name, 'interactive']
    
    try:
        # Abaqus 해석 실행
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Abaqus job '{job_name}' completed successfully.")
        
        # 5. 성공 시, .odb 파일 경로를 표준 출력으로 전달
        odb_path = f"{job_name}.odb"
        print("RESULT_PATH_START") # 파싱을 위한 구분자
        print(os.path.abspath(odb_path))
        print("RESULT_PATH_END")

    except subprocess.CalledProcessError as e:
        print(f"!!! Abaqus job '{job_name}' FAILED !!!", file=sys.stderr)
        print(f"Error: {e.stderr}", file=sys.stderr)
        sys.exit(1) # 실패 코드로 종료

if __name__ == '__main__':
    main()