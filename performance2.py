from odbAccess import openOdb, OdbError
from abaqusConstants import INTEGRATION_POINT, ELEMENT_NODAL
import numpy as np
import os
import matplotlib.pyplot as plt

np.set_printoptions(precision=16, suppress=True)

YIELD_STRENGTH = 70.0E6  # Pa

R0 = 0.926256
A0 = 0.073744

#INITIAL_A = 0.025625   # m^2

J_ideal = (2 * R0 + A0 ) # / sigma0

U_SUN = np.array([0.0, 0.0, -1.0])
N0 = np.array([0.0, 0.0, 1.0])

def J_low(odb_path):
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-1']
        frame = step.frames[-1]
        stress_field = frame.fieldOutputs['S']
        stress_subset = stress_field.getSubset(position=INTEGRATION_POINT)
        
        instance = odb.rootAssembly.instances['PART-1-1']
        total_element_count = len(instance.elements)

        if total_element_count == 0:
            print("Warning: No elements found in the instance.")
            odb.close()
            return 0.0

        compressed_element_labels = set()

        for value in stress_subset.values:
            stress_component = value.minInPlanePrincipal
            # stress_component = value.data[1] # S22
            if stress_component < 0.0:
                compressed_element_labels.add(value.elementLabel)
        odb.close()

        compressed_element_count = len(compressed_element_labels)
        compressive_ratio = float(compressed_element_count) / total_element_count
        
        return compressive_ratio

    except OdbError as e:
        print(f"Abaqus ODB Error: {e}")
        return -1.0
    except KeyError as e:
        print(f"Key Error: Could not find '{e}' in the ODB. Check step, instance, or field output names.")
        return -1.0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1.0

def J_high(odb_path):
    """
    비선형 포스트버클링 .odb의 모든 프레임에 대해 성능효율을 계산, list 로 반환
    """
    performance_efficiency_list = []
    area = []
    n = []
    sfi_list = []

    # 이상적인 돛(완전 평면, 태양광에 수직)의 기준 성능 J_ideal 계산
    # n=(0,0,1), cos_v=1, area_factor=1 일 때의 최대 추력 성분
    n_ideal = np.array([0.0, 0.0, 1.0])
    J_ideal_vector = 2 * R0 * n_ideal + A0 * U_SUN
    J_ideal = np.dot(J_ideal_vector, -U_SUN) # J_ideal = 2*R0 + A0

    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-1']  # 스텝 이름 확인 필요

        instance = odb.rootAssembly.instances['PART-1-1'] # 인스턴스 이름 확인 필요
        nodes = instance.nodes
        elements = instance.elements
        initial_coords = np.array([node.coordinates for node in nodes])
        connectivity = np.array([elem.connectivity for elem in elements]) - 1
        node_map = {node.label: idx for idx, node in enumerate(nodes)}

        print(f"Analyzing {len(step.frames)} frames from '{odb_path}' using SFI method...")

        for frame in step.frames:
            # --- 1. 변형 후 노드 좌표 계산 ---
            displacements = np.zeros_like(initial_coords)
            displacement_field = frame.fieldOutputs['U']
            for disp_value in displacement_field.values:
                node_label = disp_value.nodeLabel
                if node_label in node_map:
                    displacements[node_map[node_label]] = disp_value.data
            final_coords = initial_coords + displacements
            
            # --- 2. 면적 및 법선 벡터 계산을 위한 정점 좌표 준비 ---
            v0 = final_coords[connectivity[:, 0]]
            v1 = final_coords[connectivity[:, 1]]
            v2 = final_coords[connectivity[:, 2]]
            v3 = final_coords[connectivity[:, 3]]
            
            # 각 요소의 3D 면적 벡터 계산 (방향과 크기 모두 포함)
            cross_product1 = np.cross(v1 - v0, v2 - v0)
            cross_product2 = np.cross(v2 - v0, v3 - v0)
            area_3d_vec = 0.5 * (cross_product1 + cross_product2)

            # --- 3. SFI 계산 ---
            # 실제 3D 표면적 (A_wrinkled): 각 면적 벡터의 크기를 합산
            A_wrinkled = np.sum(np.linalg.norm(area_3d_vec, axis=1))
            
            # xy 평면 투영 면적 (A_projected): 각 면적 벡터의 z성분을 합산
            A_projected = np.sum(area_3d_vec, axis=0)[2]

            # SFI (Surface Flatness Index) 계산
            if A_wrinkled > 1e-12:
                sfi = A_projected / A_wrinkled
            else:
                sfi = 0.0
            sfi_list.append(sfi)
            
            # --- 4. 방향성 계산 ---
            # 유효 법선 벡터 (n_eff): 전체 면적 벡터 합을 실제 표면적으로 나눔
            n_eff = np.sum(area_3d_vec, axis=0) / A_wrinkled
            
            # 유효 입사각의 코사인 값
            cos_v_eff = np.abs(np.dot(n_eff, U_SUN))

            # --- 5. 최종 성능 지표 계산 ---
            # 추력의 벡터 부분 계산
            L_thrust_vec_part = 2 * R0 * cos_v_eff * n_eff + A0 * U_SUN
            
            # SFI를 면적 효율 계수로 사용하여 최종 추력 벡터 계산
            L_w = sfi * L_thrust_vec_part
            
            # 추력 방향(-U_SUN)으로의 성분 크기 (J_wrinkled) 계산
            J_wrinkled = np.dot(L_w, -U_SUN)

            # 이상적인 성능 대비 효율 계산
            performance_efficiency = J_wrinkled / J_ideal
            performance_efficiency_list.append(performance_efficiency)
            n.append(cos_v_eff)
            area.append(sfi)

        odb.close()
        print("Analysis complete.")
        return performance_efficiency_list, n, area

    except Exception as e:
        print(f"An error occurred in J_high_SFI: {e}")
        if 'odb' in locals() and odb.isopen:
            odb.close()
        return [], []

if __name__ == "__main__":

    test_odb_path_lf = 'Job-1-1.odb'
    test_odb_path_hf = 'postbuckle.odb'

    print(f"\n[1] Testing Low-Fidelity Calculation...")
    print(f"   - Target ODB: {test_odb_path_lf}")
    
    j_low = J_low(test_odb_path_lf)
    
    if j_low < 1e5: 
        print(f"\n   >>> RESULT: J_Low  = {j_low:.6f}")
        print(f"   >>> Meaning: {j_low*100:.2f}% of the area is under compression.")
    else:
        print("\n   >>> LF calculation failed.")

    print("-"*50)
    print(f"\n[2] Testing High-Fidelity  Calculation...")
    print(f"   - Target ODB: {test_odb_path_hf}")

    j_high, n_high, area_high = J_high(test_odb_path_hf)

    if j_high is not None and len(j_high) > 0:
        for i in area_high:
            print(i)

        plt.figure(figsize=(10, 5))
        plt.plot(j_high, marker='o', linestyle='-', color='b')
        plt.title("High-Fidelity Performance Efficiency")
        plt.xlabel("Frame Index")
        plt.ylabel("Performance Efficiency (J'_High)")
        #plt.ylim(0.9,1)
        plt.grid()
        #plt.axhline(0, color='red', linestyle='--')  # 0선 추가
        plt.xticks(range(len(j_high)))  # x축 눈금 설정

        plt.figure(figsize=(10, 5))
        plt.plot(n_high, marker='o', linestyle='-', color='b')
        plt.title("High-Fidelity Normal Vector")
        plt.xlabel("Frame Index")
        plt.ylabel("Normal Vector (N_High)")
        plt.grid()
        plt.xticks(range(len(n_high)))  # x축 눈금 설정

        plt.figure(figsize=(10, 5))
        plt.plot(area_high, marker='o', linestyle='-', color='b')
        plt.title("High-Fidelity Area Ratio")
        plt.xlabel("Frame Index")
        plt.ylabel("Area Ratio (A'_High)")
        plt.grid()
        plt.xticks(range(len(area_high)))  # x축 눈금 설정

        plt.show()
    else:
        print("\n   >>> HF calculation failed (or constraint violated).")
        
    print("="*50)