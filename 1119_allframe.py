from odbAccess import openOdb, OdbError
from abaqusConstants import INTEGRATION_POINT
import numpy as np
import matplotlib.pyplot as plt  # 시각화를 위해 추가

# --- Global Constants ---
R0 = 0.926256
A0 = 0.073744
U_SUN = np.array([0.0, 0.0, -1.0])

def J_low(odb_path):
    """
    Low-Fidelity: 압축 응력 요소 비율 계산
    """
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        frame = odb.steps['Step-1'].frames[-1]
        stress_field = frame.fieldOutputs['S']
        stress_subset = stress_field.getSubset(position=INTEGRATION_POINT)
        
        instance = odb.rootAssembly.instances['PART-1-1']
        total_element_count = len(instance.elements)

        if total_element_count == 0:
            odb.close()
            return 0.0

        compressed_count = 0
        for value in stress_subset.values:
            if value.minInPlanePrincipal < 0.0:
                compressed_count += 1
        odb.close()
        return float(compressed_count) / total_element_count

    except Exception:
        return 1.0 

def J_high(odb_path):
    """
    High-Fidelity: 모든 프레임에 대해 추력 성능 및 효율 계산
    Returns: (times, magnitudes, flatness_factors, efficiencies)
    """
    # 결과 저장용 리스트
    time_steps = []
    magnitudes = []
    flatness_factors = []
    efficiencies = []

    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-1']
        instance = odb.rootAssembly.instances['PART-1-1']
        
        nodes = instance.nodes
        elements = instance.elements

        # --- [Pre-computation] 초기 형상 정보 추출 ---
        # 매 프레임 반복되지 않도록 루프 밖에서 처리
        initial_coords = np.array([node.coordinates for node in nodes], dtype=np.float64)
        all_node_labels = np.array([node.label for node in nodes], dtype=np.int32)
        
        # 정렬 (searchsorted 최적화)
        sort_idx = np.argsort(all_node_labels)
        all_node_labels = all_node_labels[sort_idx]
        initial_coords = initial_coords[sort_idx]

        # Connectivity 변환
        raw_connectivity = np.array([elem.connectivity for elem in elements])
        connectivity = np.searchsorted(all_node_labels, raw_connectivity)

        # --- 초기 면적(A0) 동적 계산 ---
        v0_i = initial_coords[connectivity[:, 0]]
        v1_i = initial_coords[connectivity[:, 1]]
        v2_i = initial_coords[connectivity[:, 2]]
        v3_i = initial_coords[connectivity[:, 3]]

        init_area_vec = 0.5 * (np.cross(v1_i - v0_i, v2_i - v0_i) + np.cross(v2_i - v0_i, v3_i - v0_i))
        initial_area = np.sum(np.linalg.norm(init_area_vec, axis=1))

        if initial_area < 1e-12:
            print("Warning: Initial area is zero.")
            initial_area = 1.0 
        
        print(f"Initial Area (A0): {initial_area:.6f}")

        # 이상적인 Lightness Vector 한계치 (효율 계산용 분모)
        # L_ideal = 2*R0 + A0
        L_ideal_material_limit = 2 * R0 + A0

        # --- [Main Loop] 모든 프레임 순회 ---
        print(f"Processing {len(step.frames)} frames...")
        
        for frame in step.frames:
            current_time = frame.frameValue
            
            # Bulk Data로 변위 추출
            displacements = np.zeros_like(initial_coords)
            u_field = frame.fieldOutputs['U']
            
            for block in u_field.bulkDataBlocks:
                block_labels = block.nodeLabels
                block_data = block.data
                idx_locs = np.searchsorted(all_node_labels, block_labels)
                displacements[idx_locs] = block_data

            # 변형 후 좌표
            final_coords = initial_coords + displacements

            v0 = final_coords[connectivity[:, 0]]
            v1 = final_coords[connectivity[:, 1]]
            v2 = final_coords[connectivity[:, 2]]
            v3 = final_coords[connectivity[:, 3]]

            # 면적 벡터 및 법선 계산
            area_3d_vec = 0.5 * (np.cross(v1 - v0, v2 - v0) + np.cross(v2 - v0, v3 - v0))
            elem_area = np.linalg.norm(area_3d_vec, axis=1)
            
            valid_mask = elem_area > 1e-16
            unit_normals = np.zeros_like(area_3d_vec)
            
            if np.any(valid_mask):
                z_neg_mask = area_3d_vec[:, 2] < 0.0
                area_3d_vec[z_neg_mask] *= -1.0
                unit_normals[valid_mask] = area_3d_vec[valid_mask] / elem_area[valid_mask, np.newaxis]

            # 추력 계산
            cos_theta = np.dot(unit_normals, -U_SUN)
            cos_theta = np.clip(cos_theta, 0.0, 1.0)

            force_abs = A0 * cos_theta[:, np.newaxis] * elem_area[:, np.newaxis] * U_SUN
            force_refl = 2 * R0 * (cos_theta**2)[:, np.newaxis] * elem_area[:, np.newaxis] * unit_normals
            
            total_force_vector = np.sum(force_abs + force_refl, axis=0)

            A_projected = np.sum(area_3d_vec, axis=0)[2]
            A_wrinkled = np.sum(elem_area)

            # 1. Thrust Magnitude
            if A_projected > 1e-9:
                L_eff_vec = total_force_vector / A_projected
                thrust_val = np.linalg.norm(L_eff_vec)
            else:
                thrust_val = 0.0

            # 2. Flatness Factor
            flatness_val = A_projected / A_wrinkled if A_wrinkled > 1e-9 else 0.0

            # 3. Thrust Efficiency (%)
            eff_val = (thrust_val / L_ideal_material_limit) * 100.0 if L_ideal_material_limit > 0 else 0.0

            # 리스트에 저장
            time_steps.append(current_time)
            magnitudes.append(thrust_val)
            flatness_factors.append(flatness_val)
            efficiencies.append(eff_val)

        odb.close()
        return time_steps, magnitudes, flatness_factors, efficiencies

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], []

if __name__ == "__main__":
    target_odb = 'postbuckle.odb'
    
    print(f"Analyzing ODB: {target_odb}")
    
    # J_high 함수 호출
    times, mags, flats, effs = J_high(target_odb)
    
    if len(times) > 0:
        print("\nAnalysis Complete.")
        print(f"Final Thrust Magnitude: {mags[-1]:.6f}")
        print(f"Final Efficiency:       {effs[-1]:.2f} %")
        print(f"Final Flatness:         {flats[-1]:.6f}")

        # 프레임 인덱스 생성 (0, 1, 2, ...)
        frames = np.arange(len(times))

        # 시각화 (Matplotlib)
        plt.style.use('seaborn-darkgrid')
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # 1. Thrust Magnitude Graph
        axes[0].plot(frames, mags, 'r-o', linewidth=1.5, markersize=4)
        axes[0].set_ylabel('Thrust Magnitude (L_eff)', fontsize=12)
        axes[0].set_title('Solar Sail Performance Analysis', fontsize=14, fontweight='bold')
        axes[0].grid(True)

        # 2. Efficiency Graph
        axes[1].plot(frames, effs, 'g-s', linewidth=1.5, markersize=4)
        axes[1].set_ylabel('Thrust Efficiency (%)', fontsize=12)
        axes[1].grid(True)
        
        # 3. Flatness Graph
        axes[2].plot(frames, flats, 'b-^', linewidth=1.5, markersize=4)
        axes[2].set_ylabel('Flatness Factor', fontsize=12)
        axes[2].set_xlabel('Frame Index', fontsize=12) # X축 라벨 변경
        axes[2].grid(True)

        # X축 눈금을 정수로 설정 (프레임이므로)
        # 프레임이 너무 많으면 알아서 간격 조정됨
        # axes[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show()
        
    else:
        print("Failed to extract data from ODB.")