<<<<<<< HEAD
from odbAccess import openOdb, OdbError
import numpy as np
import os

# --- Constants ---

YIELD_STRENGTH = 70.0E6  # Pa

R0 = 0.926256
A0 = 0.073744

INITIAL_A = 50.0    # m^2

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
    비선형 포스트버클링 .odb 로부터 성능효율 계산
    """
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-1']
        frame = step.frames[-1]

        instance = odb.rootAssembly.instances['PART-1-1']

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
        weights = cos**2
        n_eff = np.mean(normal * weights[:, np.newaxis], axis=0)
        cos_v_eff = np.dot(n_eff, U_SUN)
        cos_v_eff = np.clip(cos_v_eff, -1.0, 1.0)

        L_w = (1 + delta_A_ratio) * (2 * R0 * (cos_v_eff**2) * n_eff + A0 * cos_v_eff * U_SUN)

        odb.close()
        return -np.dot(L_w, U_SUN)/J_ideal
    
    except Exception as e:
        print(f"Error reading ODB file: {e}")
        return -1e6

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

    if j_high > -1e5: 
        print(f"\n   >>> RESULT: J'_High (Performance Efficiency) = {j_high:.6f}")
        print(f"   >>> Meaning: Achieved {j_high*100:.2f}% of the ideal performance.")
    else:
        print("\n   >>> HF calculation failed (or constraint violated).")
        
    print("="*50)
=======
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

>>>>>>> a730cf014da51ad52454f373b09ea4f3ee084433
