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
    bulkDataBlocks를 사용하여 속도를 획기적으로 개선한 J_high 함수
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

        # [Pre-computation] 
        # 노드 좌표와 라벨을 Numpy 배열로 추출
        # instance.nodes를 순서대로 추출하면 initial_coords와 labels의 인덱스가 일치합니다.
        initial_coords = np.array([node.coordinates for node in nodes], dtype=np.float32) # float32로 메모리 절약 가능
        all_node_labels = np.array([node.label for node in nodes], dtype=np.int32)
        
        # searchsorted를 위해 노드 라벨이 정렬되어 있는지 확인 및 정렬
        # (대부분 정렬되어 있지만, 안전장치)
        sort_idx = np.argsort(all_node_labels)
        all_node_labels = all_node_labels[sort_idx]
        initial_coords = initial_coords[sort_idx]
        
        # Element connectivity도 노드 라벨 기준이므로, 이를 위에서 정렬한 인덱스로 변환해야 함
        # 하지만 Abaqus connectivity는 'Node Label'을 반환하지 않고 'Node Object'를 참조하거나
        # 단순히 순서대로 연결된 경우가 많으므로, 여기서는 기존 로직(인덱스-1)을 유지하되
        # connectivity가 '노드 인덱스(0부터 시작)'를 가리키도록 주의해야 합니다.
        # *가장 안전한 방법*: Connectivity에 있는 노드 라벨을 가져와서 searchsorted로 인덱스 변환
        
        raw_connectivity = np.array([elem.connectivity for elem in elements]) # 이것은 Node Label의 튜플임
        # Node Label -> Array Index 변환 (Vectorized)
        # raw_connectivity의 shape은 (N_elem, 4)
        connectivity = np.searchsorted(all_node_labels, raw_connectivity)

        INITIAL_A = 0.02  # m^2

        for frame in step.frames:
            displacements = np.zeros_like(initial_coords)
            
            # 'U' 필드 가져오기
            u_field = frame.fieldOutputs['U']
            
            # [Bulk Data Block 적용]
            for block in u_field.bulkDataBlocks:
                # block.data는 메모리 뷰이므로 copy 불필요, 바로 가져옴
                block_node_labels = block.nodeLabels
                block_disp_data = block.data
                
                # 전체 노드 배열에서 현재 블록 노드들의 인덱스 찾기
                # all_node_labels는 정렬되어 있으므로 searchsorted 사용 가능 (매우 빠름)
                # 주의: block_node_labels에 있는 라벨이 all_node_labels에 반드시 존재한다고 가정
                idx_locs = np.searchsorted(all_node_labels, block_node_labels)
                
                # 데이터 일괄 대입
                displacements[idx_locs] = block_disp_data

            # 이후 로직은 기존과 동일 (Numpy Vectorization 유지)
            final_coords = initial_coords + displacements

            v0_3d = final_coords[connectivity[:, 0]]
            v1_3d = final_coords[connectivity[:, 1]]
            v2_3d = final_coords[connectivity[:, 2]]
            v3_3d = final_coords[connectivity[:, 3]]

            # --- 이하 기존 코드와 동일 ---
            area_3d_vec = 0.5 * (np.cross(v1_3d - v0_3d, v2_3d - v0_3d) + np.cross(v2_3d - v0_3d, v3_3d - v0_3d))

            elem_area = np.linalg.norm(area_3d_vec, axis=1)
            nonzero_mask = elem_area > 1e-16
            unit_normals = np.zeros_like(area_3d_vec)
            
            # ... (중략: 기존 벡터 연산 로직) ...
            if np.any(nonzero_mask):
                z_negative_mask = area_3d_vec[:, 2] < 0.0
                area_3d_vec[z_negative_mask] *= -1.0
                unit_normals[nonzero_mask] = area_3d_vec[nonzero_mask] / elem_area[nonzero_mask, np.newaxis]

            elem_angles_rad = np.arcsin(np.clip(np.abs(unit_normals[:, 2]), 0.0, 1.0))
            elem_angles_deg = np.degrees(elem_angles_rad)

            cos_theta = np.dot(unit_normals, -U_SUN)
            cos_theta = np.clip(cos_theta, 0.0, 1.0)

            force_abs_vectors = A0 * cos_theta[:, np.newaxis] * elem_area[:, np.newaxis] * U_SUN
            force_refl_vectors = 2 * R0 * (cos_theta**2)[:, np.newaxis] * elem_area[:, np.newaxis] * unit_normals

            total_force_vector = np.sum(force_abs_vectors + force_refl_vectors, axis=0)

            A_projected = np.sum(area_3d_vec, axis=0)[2]
            A_wrinkled = np.sum(np.linalg.norm(area_3d_vec, axis=1))
            
            L_eff = total_force_vector / A_projected
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

            area.append(net_change) # net_change로 통일
            angle.append(elem_angles_deg)

        odb.close()
        return performance, n, area, angle, net_area_changes, stretch_ratios, flatness_factors
    
    except Exception as e:
        print(f"Error reading ODB file: {e}")
        import traceback
        traceback.print_exc()
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