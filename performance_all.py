from odbAccess import openOdb, OdbError
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Constants ---

YIELD_STRENGTH = 70.0E6  # Pa

R0 = 0.926256
A0 = 0.073744

INITIAL_A = 0.025    # m^2

J_ideal = (2 * R0 + A0 ) # / sigma0

U_SUN = np.array([0.0, 0.0, -1.0])
N0 = np.array([0.0, 0.0, 1.0])

def J_low(odb_path):
    """
    선형 해석 .odb 로부터 압축 영역 면적 비율 계산
    """
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-2']
        frame = step.frames[-1]

        stress_field = frame.fieldOutputs['S']
        area_field = frame.fieldOutputs['AREA']

        instance = odb.rootAssembly.instances['PART-1-1']

        total_area = 0.0
        compressed_area = 0.0

        for elem_area, elem_stress in zip(area_field.values, stress_field.values):
            s22 = elem_stress.data[1]  # S22 성분

            total_area += elem_area.data

            if s22 < 0: # 압축응력
                compressed_area += elem_area.data
        
        odb.close()

        if total_area == 0:
            return 1.0
        
        car = compressed_area / total_area
        return car

    except Exception as e:
        print(f"Error reading ODB file: {e}")
        return 1e6

def J_high(odb_path):
    """
    비선형 포스트버클링 .odb의 모든 프레임에 대해 성능효율을 계산, list 로 반환
    """
    performance = []
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-1']

        instance = odb.rootAssembly.instances['PART-1-1']
        nodes = instance.nodes
        elements = instance.elements

        initial_coords = np.array([node.coordinates for node in nodes])
        connectivity = np.array([elem.connectivity for elem in elements]) - 1

        node_map = {node.label: idx for idx, node in enumerate(nodes)}

        for frame in step.frames:
            displacements = np.zeros_like(initial_coords)
            displacement_field = frame.fieldOutputs['U']

            for disp_value in displacement_field.values:
                node_label = disp_value.nodeLabel
                if node_label in node_map:
                    node_index = node_map[node_label]
                    displacements[node_index] = disp_value.data

            final_coords = initial_coords + displacements


            v0 = final_coords[connectivity[:, 0]]
            v1 = final_coords[connectivity[:, 1]]
            v2 = final_coords[connectivity[:, 2]]
            
            area_3d = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            A_wrinked = np.sum(area_3d)
            delta_A_ratio = (A_wrinked - INITIAL_A) / INITIAL_A

            normal = np.cross(v1 - v0, v2 - v0)
            z_sign = np.sign(normal[:, 2])
            z_sign[z_sign == 0] = 1
            normal *= z_sign[:, np.newaxis]
            normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]

            cos = np.dot(normal, U_SUN)
            weights = cos**2
            n_eff = np.mean(normal * weights[:, np.newaxis], axis=0)
            cos_v_eff = np.dot(n_eff, U_SUN)
            cos_v_eff = np.clip(cos_v_eff, -1.0, 1.0)

            L_w = (1 + delta_A_ratio) * (2 * R0 * (cos_v_eff**2) * n_eff + A0 * cos_v_eff * U_SUN)

            j_frame = -np.dot(L_w, U_SUN)/J_ideal
            performance.append(j_frame)

        odb.close()
        return performance
    
    except Exception as e:
        print(f"Error reading ODB file: {e}")
        return []

if __name__ == "__main__":

    test_odb_path_lf = 'Job-1.odb'
    test_odb_path_hf = 'Job-2.odb'

    print("="*50)
    print("PERFORMANCE CALCULATOR - STANDALONE TEST")
    print("="*50)

    print(f"\n[1] Testing Low-Fidelity (CAR) Calculation...")
    print(f"   - Target ODB: {test_odb_path_lf}")
    
    j_low = J_low(test_odb_path_lf)
    
    if j_low < 1e5: 
        print(f"\n   >>> RESULT: J'_Low (Compressive Area Ratio) = {j_low:.6f}")
        print(f"   >>> Meaning: {j_low*100:.2f}% of the area is under compression.")
    else:
        print("\n   >>> LF calculation failed.")

    print("-"*50)


    print(f"\n[2] Testing High-Fidelity (Performance Efficiency) Calculation...")
    print(f"   - Target ODB: {test_odb_path_hf}")

    j_high = J_high(test_odb_path_hf)

    if j_high is not None and len(j_high) > 0:
        print(f"\n   >>> RESULT: J'_High (Performance Efficiency) = \n")
        for i in j_high:
            print(f"{i*100:.3f}")
        plt.figure(figsize=(10, 5))
        plt.plot(j_high, marker='o', linestyle='-', color='b')
        plt.title("High-Fidelity Performance Efficiency")
        plt.xlabel("Frame Index")
        plt.ylabel("Performance Efficiency (J'_High)")
        plt.grid()
        plt.axhline(0, color='red', linestyle='--')  # 0선 추가
        plt.xticks(range(len(j_high)))  # x축 눈금 설정
        plt.show()
    else:
        print("\n   >>> HF calculation failed (or constraint violated).")
        
    print("="*50)