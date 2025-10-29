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
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True, help='Design parameters as a JSON string')
    parser.add_argument('--fidelity', type=str, required=True, choices=['low', 'high'], help='Fidelity level')
    args = parser.parse_args()
    
    design_params = json.loads(args.params)
    
    job_id = hash(args.params) 
    job_name = f"sail_{args.fidelity}_{job_id}"
    
    template_path = f"template_{args.fidelity}.inp" # LF/HF용 템플릿 분리
    
    inp_file = create_inp_file(template_path, design_params, job_name)
    print(f"Generated {inp_file} for fidelity {args.fidelity}")

    command = ['abaqus', 'job=' + job_name, 'interactive']
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Abaqus job '{job_name}' completed successfully.")
        
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