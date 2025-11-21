from odbAccess import openOdb, OdbError
from abaqusConstants import CENTROID, WHOLE_ELEMENT
import numpy as np
import matplotlib.pyplot as plt  # 시각화를 위해 추가

R0 = 0.926256
A0 = 0.073744
U_SUN = np.array([0.0, 0.0, -1.0])
L_ideal_material_limit = 2 * R0 + A0

def J_low(odb_path):
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        frame = odb.steps['Step-1'].frames[-1]
        
        stress_field = frame.fieldOutputs['S']
        stress_subset = stress_field.getSubset(position=CENTROID)
        
        try:
            evol_field = frame.fieldOutputs['EVOL']
        except KeyError:
            print("Error: 'EVOL' field output is missing in ODB. Please request EVOL in Abaqus.")
            odb.close()
            return 0.0
            
        evol_subset = evol_field.getSubset(position=WHOLE_ELEMENT)
        
        elem_areas = {}
        total_area = 0.0
        
        for val in evol_subset.values:
            label = val.elementLabel
            area = val.data  # EVOL 값
            elem_areas[label] = area
            total_area += area

        if total_area == 0:
            print("Warning: Total area is zero.")
            odb.close()
            return 0.0

        compressed_area = 0.0
        
        for val in stress_subset.values:
            if val.minInPlanePrincipal < 0.0:
                label = val.elementLabel
                if label in elem_areas:
                    compressed_area += elem_areas[label]
        
        odb.close()
        
        return float(compressed_area) / total_area

    except OdbError as e:
        print(f"Abaqus ODB Error: {e}")
        return -1.0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1.0

def J_high(odb_path):
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        
        # 마지막 프레임 (변형 후 상태)
        frame = odb.steps['Step-1'].frames[-1] 
        instance = odb.rootAssembly.instances['PART-1-1']
        
        elements = instance.elements
        nodes = instance.nodes

        # ---------------------------------------------------------
        # [1] Connectivity 데이터 전처리 (불규칙 메쉬 대응)
        # ---------------------------------------------------------
        # 목표: 모든 요소를 4개의 노드 인덱스로 통일 (Tri는 마지막 노드 반복)
        # Abaqus ODB의 connectivity는 튜플 형태임.
        
        raw_conn = [e.connectivity for e in elements]
        max_nodes_per_elem = 4 # 멤브레인/쉘은 최대 4절점
        
        padded_conn = []
        for c in raw_conn:
            if len(c) == 4:
                padded_conn.append(c)
            elif len(c) == 3:
                # 삼각형이면 (0, 1, 2) -> (0, 1, 2, 2)로 패딩
                padded_conn.append((c[0], c[1], c[2], c[2]))
            else:
                # 예외 처리 (Line 요소 등)
                pass
                
        connectivity = np.array(padded_conn, dtype=np.int32)

        # ---------------------------------------------------------
        # [2] COORD(변형 후 좌표) 데이터 추출 (Bulk Data)
        # ---------------------------------------------------------
        try:
            coord_field = frame.fieldOutputs['COORD']
        except KeyError:
            print("Error: 'COORD' field output is missing.")
            return 0.0, 0.0, 0.0

        # Node Label을 Index로 사용하는 좌표 배열 생성
        # (노드 라벨이 1부터 시작하거나, 중간에 비어있을 수 있으므로 max label 사용)
        max_node_label = nodes[-1].label 
        # 만약 정렬 안됐다면: max_node_label = max(n.label for n in nodes)
        
        node_coords = np.zeros((max_node_label + 1, 3), dtype=np.float64)

        # Bulk Data로 좌표 채우기
        for block in coord_field.bulkDataBlocks:
            node_coords[block.nodeLabels] = block.data

        # ---------------------------------------------------------
        # [3] 요소 기하 계산 (Vectorized) - 대각선 외적법
        # ---------------------------------------------------------
        # 4개 노드의 좌표 추출
        # connectivity가 Node Label을 담고 있으므로 바로 인덱싱
        v0 = node_coords[connectivity[:, 0]]
        v1 = node_coords[connectivity[:, 1]]
        v2 = node_coords[connectivity[:, 2]]
        v3 = node_coords[connectivity[:, 3]]

        # 대각선 벡터 계산 (General Quad & Triangle Approach)
        # Quad: 대각선 1, 2
        # Tri (v0,v1,v2,v2): (v2-v0) 와 (v2-v1) -> 삼각형의 두 변이 됨
        diag1 = v2 - v0
        diag2 = v3 - v1

        # 외적 (Cross Product) -> 법선 방향 * (면적 * 2)
        cross_prod = np.cross(diag1, diag2)
        
        # 요소별 법선 벡터 크기 (= 면적 * 2)
        norms = np.linalg.norm(cross_prod, axis=1)
        
        # 면적 (dA)
        # 대각선 외적의 크기는 사각형 면적의 약 2배 (평면일 때 정확히 2배)
        elem_area = 0.5 * norms
        
        # 단위 법선 벡터 (n)
        valid_mask = norms > 1e-12
        unit_normals = np.zeros_like(cross_prod)
        # 0으로 나누기 방지
        unit_normals[valid_mask] = cross_prod[valid_mask] / norms[valid_mask, np.newaxis]

        # 법선 방향 보정 (태양을 바라보는 방향 +Z로 정렬)
        # 만약 모델링 상 요소의 Top/Bottom이 뒤집혀 있다면 여기서 보정
        # 여기서는 Z성분이 음수면 뒤집어서 태양(+Z)을 보게 함
        z_neg_mask = unit_normals[:, 2] < 0.0
        unit_normals[z_neg_mask] *= -1.0

        # ---------------------------------------------------------
        # [4] 추력 계산 (Physics)
        # ---------------------------------------------------------
        # 입사각 (cos theta)
        # U_SUN = [0,0,-1]. n = [0,0,1]. dot(n, -U_SUN) = 1
        cos_theta = np.dot(unit_normals, -U_SUN)
        cos_theta = np.clip(cos_theta, 0.0, 1.0) # 음수(뒷면) 방지

        # 힘 벡터 계산
        # Abs: 미는 힘 (U_SUN 방향, -Z)
        force_abs = A0 * cos_theta[:, np.newaxis] * elem_area[:, np.newaxis] * U_SUN
        
        # Refl: 미는 힘 (법선 반대 방향 -n, -Z)
        # 중요: unit_normals가 +Z를 향하므로, 밀리는 힘은 -unit_normals 방향
        force_refl = 2 * R0 * (cos_theta**2)[:, np.newaxis] * elem_area[:, np.newaxis] * (-unit_normals)
        
        total_force_vector = np.sum(force_abs + force_refl, axis=0)

        # ---------------------------------------------------------
        # [5] 결과 산출
        # ---------------------------------------------------------
        # 투영 면적 (Z축)
        A_projected = np.sum(elem_area * unit_normals[:, 2])
        A_wrinkled = np.sum(elem_area)

        if A_projected > 1e-9:
            L_eff_vec = total_force_vector / A_projected
            # 추진 방향 유효 성분 (-U_SUN 방향 투영)
            thrust_val = np.dot(L_eff_vec, -U_SUN)
        else:
            thrust_val = 0.0

        flatness_factor = A_projected / A_wrinkled if A_wrinkled > 0 else 0.0
        thrust_efficiency = thrust_val / L_ideal_material_limit if L_ideal_material_limit > 0 else 0.0

        odb.close()
        return thrust_val, flatness_factor, thrust_efficiency

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0

if __name__ == "__main__":
    target_hf = 'postbuckle.odb'
    target_lf = 'Job-1.odb'
    
    car = J_low(target_lf)
    print(f"Compressed Area Ratio (J_low): {car:.6f}")

    times, mags, flats, effs = J_high(target_hf)
    
    if len(times) > 0:
        print("\nAnalysis Complete.")
        print(f"Final Thrust Magnitude: {mags[-1]:.6f}")
        print(f"Final Efficiency:       {effs[-1]:.6f} %")
        print(f"Final Flatness:         {flats[-1]:.6f}")

        frames = np.arange(len(times))

        for i in range(len(times)):
            print(f"Frame {i}: Time={times[i]:.4f}, Thrust={mags[i]:.6f}, Efficiency={effs[i]:.6f}%, Flatness={flats[i]:.6f}")

        plt.style.use('default')
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

        plt.tight_layout()
        plt.show()
        
    else:
        print("Failed to extract data from ODB.")