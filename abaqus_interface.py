from odbAccess import onpenOdb
import numpy as np

# --- Constants ---

YIELD_STRENGTH = 70.0E6  # Pa

R0 = 0.926256
A0 = 0.073744

INITIAL_A = 50.0    # m^2

J_ideal = ( 1 + R0 ) # / sigma0

U_SUN = np.array([0.0, 0.0, -1.0])
N0 = np.array([0.0, 0.0, 1.0])

def J_low(odb_path):
    """
    선형 해석 .odb 로부터 압축 영역 면적 비율 계산
    """
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Job-1']
        frame = step.frames[-1]

        stress_field = frame.fieldOutputs['S']
        area_field = frame.fieldOutputs['AREA']

        instance = odb.rootAssembly.instances['Sail_1']

        total_area = 0.0
        compressed_area = 0.0

        for elem_area, elem_stress in zip(area_field.values, stress_field.values):
            s22 = elem_stress.data[1]  # S22 성분

            total_area += elem_area.data

            if s22 < 0: # 압축응력
                cpmressed_area += elem_area.data
        
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
    비선형 포스트버클링 .odb 로부터 성능효율 계산
    """
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Job-2']
        frame = step.frames[-1]

        instance = odb.rootAssembly.instances['Sail_1']

        nodes = instance.nodes
        elements = instance.elements
        displacement_field = frame.fieldOutputs['U']

        initial_coords = np.array([node.coordinates for node in nodes])
        displacements = np.array([disp.data for disp in displacement_field.getSubset(region=instance).values])
        final_coords = initial_coords + displacements

        connectivity = np.array([elem.connectivity for elem in elements]) - 1

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
        n_eff = np.mean(normal * cos**2[:, np.newaxis], axis=0)
        cos_v_eff = np.dot(n_eff, U_SUN)
        cos_v_eff = np.clip(cos_v_eff, -1.0, 1.0)

        L_w = (1 + delta_A_ratio) * (2 * R0 * (cos_v_eff**2) * n_eff + A0 * cos_v_eff * U_SUN)

        odb.close()
        return (-np.dot(L_w, U_SUN))/J_ideal
    
    except Exception as e:
        print(f"Error reading ODB file: {e}")
        return -1e6

