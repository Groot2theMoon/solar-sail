from odbAccess import openOdb, OdbError
from abaqusConstants import INTEGRATION_POINT, ELEMENT_NODAL
import numpy as np
import os
import matplotlib.pyplot as plt

np.set_printoptions(precision=16, suppress=True)

R0 = 0.926256
A0 = 0.073744

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
    performance = []
    n = []
    area = []
    angle = []
    net_area_changes = []
    stretch_ratios = []
    flatness_factors = []

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

            v0_3d = final_coords[connectivity[:, 0]]
            v1_3d = final_coords[connectivity[:, 1]]
            v2_3d = final_coords[connectivity[:, 2]]
            v3_3d = final_coords[connectivity[:, 3]]

            area_3d_vec = 0.5 * (np.cross(v1_3d - v0_3d, v2_3d - v0_3d) + np.cross(v2_3d - v0_3d, v3_3d - v0_3d))

            # 각 요소의 단위 법선 벡터(unit_normals)와 실제 면적(elem_areas) 계산
            # 각도 정의: angle = arcsin(|n_z| / |n|)
            elem_area = np.linalg.norm(area_3d_vec, axis=1)
            nonzero_mask = elem_area > 1e-16
            unit_normals = np.zeros_like(area_3d_vec)
            if np.any(nonzero_mask):
                z_negative_mask = area_3d_vec[:, 2] < 0.0
                area_3d_vec[z_negative_mask] *= -1.0
                unit_normals[nonzero_mask] = area_3d_vec[nonzero_mask] / elem_area[nonzero_mask, np.newaxis]

            elem_angles_rad = np.arcsin(np.clip(np.abs(unit_normals[:, 2]), 0.0, 1.0))
            elem_angles_deg = np.degrees(elem_angles_rad)

            # elem_angles_deg: 각 요소별 수평면 대비 각도(도). 필요하면 프레임별로 수집하여 반환하거나 저장하세요.

            INITIAL_A = 0.02  # m^2

            # 각 요소와 태양광 벡터(U_SUN) 사이의 각도(cos) 계산
            # abs를 사용하여 법선 벡터 방향에 관계없이 입사각을 0~90도로 간주
            cos_theta = np.dot(unit_normals, -U_SUN)
            cos_theta = np.clip(cos_theta, 0.0, 1.0)  # 음수 방지

            # 각 요소가 발생하는 미소 추력벡터 계산
            # np.newaxis를 사용하여 (N,) 배열을 (N,1)로 만들어 (N,3) 배열과 브로드캐스팅

            # 흡수 성분: F_abs ∝ A0 * cos(θ) * dA * U_SUN
            # 힘의 방향이 U_SUN 방향이므로, 최종 추력은 -U_SUN 방향임
            force_abs_vectors = A0 * cos_theta[:, np.newaxis] * elem_area[:, np.newaxis] * U_SUN
            
            # 반사 성분: F_refl ∝ 2 * R0 * cos²(θ) * dA * n
            force_refl_vectors = 2 * R0 * (cos_theta**2)[:, np.newaxis] * elem_area[:, np.newaxis] * unit_normals

            # 모든 요소의 힘 벡터를 합산하여 돛 전체의 총 힘 벡터를 구함
            total_force_vector = np.sum(force_abs_vectors + force_refl_vectors, axis=0)

            A_projected = np.sum(area_3d_vec, axis=0)[2]
            A_wrinkled = np.sum(np.linalg.norm(area_3d_vec, axis=1))
            # 초기면적으로 나눠서 정규화
            L_eff = total_force_vector / A_projected
            
            # performance 리스트에는 이제 3D 벡터가 저장됨
            performance.append(L_eff)
            
        
            net_change = (A_projected - INITIAL_A) / INITIAL_A
            net_area_changes.append(net_change)

            stretch = (A_wrinkled - INITIAL_A) / INITIAL_A
            stretch_ratios.append(stretch)
            
            flatness = A_projected / A_wrinkled if A_wrinkled > 1e-9 else 1.0
            flatness_factors.append(flatness)
            n_eff = np.sum(area_3d_vec, axis=0) / A_wrinkled
            cos_v_eff = np.abs(np.dot(n_eff, U_SUN))
            v_deg = np.degrees(np.arccos(np.clip(cos_v_eff, -1.0, 1.0)))
            n.append(v_deg)

            A_projected = np.sum(area_3d_vec, axis=0)[2]
            area.append((A_projected-INITIAL_A) / INITIAL_A)
            
            angle.append(elem_angles_deg)

        odb.close()
        return performance, n, area, angle, net_area_changes, stretch_ratios, flatness_factors
    
    except Exception as e:
        print(f"Error reading ODB file: {e}")
        return [], [], [], [], [], [], []
    
def analyze_equivalent_normal(odb_path, target_frame_index=-1):
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-1']
        
        if not (-len(step.frames) <= target_frame_index < len(step.frames)):
            print(f"Error: Frame index {target_frame_index} is out of bounds.")
            odb.close()
            return None

        frame = step.frames[target_frame_index]
        
        instance = odb.rootAssembly.instances['PART-1-1']
        nodes = instance.nodes
        elements = instance.elements

        initial_coords = np.array([node.coordinates for node in nodes])
        connectivity = np.array([elem.connectivity for elem in elements]) - 1
        node_map = {node.label: idx for idx, node in enumerate(nodes)}

        displacements = np.zeros_like(initial_coords)
        displacement_field = frame.fieldOutputs['U']
        for disp_value in displacement_field.values:
            if disp_value.nodeLabel in node_map:
                displacements[node_map[disp_value.nodeLabel]] = disp_value.data
        
        final_coords = initial_coords + displacements

        v0 = final_coords[connectivity[:, 0]]
        v1 = final_coords[connectivity[:, 1]]
        v2 = final_coords[connectivity[:, 2]]
        v3 = final_coords[connectivity[:, 3]]
        area_3d_vec = 0.5 * (np.cross(v1 - v0, v2 - v0) + np.cross(v2 - v0, v3 - v0))
        
        elem_areas = np.linalg.norm(area_3d_vec, axis=1)
        valid_mask = elem_areas > 1e-16
        unit_normals = np.zeros_like(area_3d_vec)
        if np.any(valid_mask):
            z_negative_mask = area_3d_vec[:, 2] < 0
            area_3d_vec[z_negative_mask] *= -1.0
            unit_normals[valid_mask] = area_3d_vec[valid_mask] / elem_areas[valid_mask, np.newaxis]

        thetas_rad = np.arccos(np.clip(unit_normals[:, 2], -1.0, 1.0))
        phis_rad = np.arctan2(unit_normals[:, 1], unit_normals[:, 0])

        thetas_deg = np.degrees(thetas_rad)
        phis_deg = np.degrees(phis_rad)
        phis_deg[phis_deg < 0] += 360  # 범위를 [0, 360]으로 변환
        weights = unit_normals[:, 2]**2
        
        weighted_normals = weights[:, np.newaxis] * unit_normals
        m_equiv = np.sum(weighted_normals, axis=0)
        m_equiv /= np.linalg.norm(m_equiv) # 최종 방향을 얻기 위해 단위 벡터로 정규화

        final_theta_deg = np.degrees(np.arccos(m_equiv[2]))
        final_phi_deg = np.degrees(np.arctan2(m_equiv[1], m_equiv[0]))
        if final_phi_deg < 0:
            final_phi_deg += 360

        odb.close()
        return thetas_deg, phis_deg, final_theta_deg, final_phi_deg

    except Exception as e:
        print(f"An unexpected error occurred in analyze_equivalent_normal: {e}")
        return None    

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

    L_eff_vector, n_high, area_high, angle_high, net_changes, stretches, flatness = J_high(test_odb_path_hf)

    if L_eff_vector is not None and len(L_eff_vector) > 0:
        
        L_eff_data = np.array(L_eff_vector)

        plt.figure(figsize=(12, 6))
        #plt.plot(L_eff_data[:, 0], marker='o', linestyle='-', label='L_eff_X (Side Force)')
        plt.plot(L_eff_data[:, 1], marker='s', linestyle='-', label='L_eff_Y (Side Force)')
        #plt.plot(L_eff_data[:, 2], marker='^', linestyle='-', label='L_eff_Z (Main Thrust Magnitude)')
        plt.title("High-Fidelity: Effective Lightness Vector Components")
        plt.xlabel("Frame Index")
        plt.ylabel("Lightness Vector Component")
        plt.legend()
        plt.grid()
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.xticks(range(len(L_eff_vector)))

        L_eff_magnitudes = np.linalg.norm(L_eff_data, axis=1)
        plt.figure(figsize=(12, 6))
        plt.plot(L_eff_magnitudes, marker='o', linestyle='-', color='r', label="Magnitude of Wrinkled Sail (||L_eff||)")
        
        # 이상적인 돛의 추력 크기 
        ideal_magnitude = 2 * R0 - A0 
        plt.axhline(ideal_magnitude, color='g', linestyle='--', label=f'Ideal Flat Sail Magnitude ({ideal_magnitude:.4f})')
        
        plt.title("High-Fidelity: Magnitude of Effective Lightness Vector")
        plt.xlabel("Frame Index")
        plt.ylabel("Magnitude")
        plt.ylim(bottom=0) # 크기는 항상 0 이상
        plt.legend()
        plt.grid()
        plt.xticks(range(len(L_eff_vector)))

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

        plt.figure(figsize=(12, 7))
        plt.plot(net_changes, marker='o', linestyle='-', label='Net Change (Tension - Wrinkle)')
        plt.plot(stretches, marker='s', linestyle='--', label='Stretch (Tension Only)')
        #plt.plot(flatness, marker='^', linestyle=':', label='Flatness (Wrinkle Only)')
        
        plt.title("Decomposition of Area Change Effects")
        plt.xlabel("Frame Index")
        plt.ylabel("Ratio / Factor")
        plt.legend()
        plt.grid()
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8) # 0 기준선
        plt.axhline(1, color='grey', linestyle=':', linewidth=0.8)  # 1 기준선 (평탄도)
        plt.xticks(range(len(net_changes)))

        if angle_high and -len(angle_high) <= 50 < len(angle_high):
            angles_for_frame = angle_high[50]
            actual_frame_number = 50 if 50 >= 0 else len(angle_high) + 50
            plt.figure(figsize=(12, 6))
            plt.hist(angles_for_frame, bins=100, color='skyblue', edgecolor='black')
            plt.title(f"Distribution of Element Angles for Frame {actual_frame_number}", fontsize=16)
            plt.xlabel("Angle (degrees from horizontal plane)", fontsize=12)
            plt.ylabel("Number of Elements (Frequency)", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            print(f"\n[3] Generated a histogram for element angles of frame {actual_frame_number}.")
        else:
            print("\n   >>> Could not generate histogram. Invalid frame index or no angle data available.")

    else:
        print("\n   >>> HF calculation failed (or constraint violated).")

        print("\n" + "="*50)
    print("[3] Analyzing Equivalent Normal Vector Distribution...")
    print(f"    - Target ODB: {test_odb_path_hf}")

    analysis_result = analyze_equivalent_normal(test_odb_path_hf, target_frame_index=-1)
    
    if analysis_result:
        all_thetas, all_phis, final_theta, final_phi = analysis_result

        plt.figure(figsize=(12, 6))
        plt.hist(all_thetas, bins=1000, range=(0, 90), color='skyblue', edgecolor='black', label='Element Distribution')
        plt.axvline(final_theta, color='r', linestyle='--', linewidth=2, label=f'Equivalent Normal θ = {final_theta:.3f}°')
        plt.title("Distribution of Element Zenithal Angles (θ)")
        plt.xlabel("Zenithal Angle θ (degrees from Z-axis)")
        plt.ylabel("Number of Elements")
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(12, 6))
        plt.hist(all_phis, bins=1000, range=(0, 360), color='lightgreen', edgecolor='black', label='Element Distribution')
        plt.axvline(final_phi, color='r', linestyle='--', linewidth=2, label=f'Equivalent Normal φ = {final_phi:.3f}°')
        plt.title("Distribution of Element Azimuthal Angles (φ)")
        plt.xlabel("Azimuthal Angle φ (degrees from X-axis)")
        plt.ylabel("Number of Elements")
        plt.legend()
        plt.grid(True)


        print(f"Equivalent Normal Vector Direction (Vulpetti's Method):")
        print(f"  > Zenithal Angle (θ): {final_theta:.4f}°")
        print(f"  > Azimuthal Angle (φ): {final_phi:.4f}°")

    plt.show()