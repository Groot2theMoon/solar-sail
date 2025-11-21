from odbAccess import openOdb, OdbError
from abaqusConstants import CENTROID, INTEGRATION_POINT, DOUBLE_PRECISION, WHOLE_ELEMENT
import numpy as np
import matplotlib.pyplot as plt

R0 = 0.926256
A0 = 0.073744
U_SUN = np.array([0.0, 0.0, -1.0])
L_ideal = 2 * R0 + A0

def J_low(odb_path):
    odb = openOdb(path=odb_path, readOnly=True)
    frame = odb.steps['Step-1'].frames[-1]
    
    evol_field = frame.fieldOutputs['EVOL'].getSubset(position=WHOLE_ELEMENT)

    elem_areas = {}
    total_area = 0.0
    
    for val in evol_field.values:
        label = val.elementLabel
        area = val.dataDouble
        elem_areas[label] = area
        total_area += area

    stress_field = frame.fieldOutputs['S'].getSubset(position=INTEGRATION_POINT)
    
    compressed_elements = set()
    
    for val in stress_field.values:
        if val.minInPlanePrincipal < 0.0:
            compressed_elements.add(val.elementLabel)
    
    compressed_area = 0.0
    for label in compressed_elements:
        if label in elem_areas:
            compressed_area += elem_areas[label]
    
    odb.close()
    return float(compressed_area) / total_area

def J_high(odb_path):
    odb = openOdb(path=odb_path, readOnly=True)
    step = odb.steps['Step-1']
    instance = odb.rootAssembly.instances['PART-1-1']
    
    elements = instance.elements
    nodes = instance.nodes
    
    raw_conn = [e.connectivity for e in elements]
    padded_conn = []
    for c in raw_conn:
        padded_conn.append((c[0], c[1], c[2], c[3])) 
        
    connectivity = np.array(padded_conn, dtype=np.int32)
    max_node_label = nodes[-1].label + 10000

    print(f"Processing {len(step.frames)} frames for {len(connectivity)} valid elements...")

    time_steps = []
    magnitudes = []
    flatness_factors = []
    efficiencies = []

    for frame in step.frames:
        current_time = frame.frameValue
        coord_field = frame.fieldOutputs['COORD']
        node_coords = np.zeros((max_node_label + 1, 3), dtype=np.float64)

        for block in coord_field.bulkDataBlocks:
            labels = block.nodeLabels
            
            if labels is None or len(labels) == 0:
                continue
                
            data = block.data
            current_max_label = np.max(labels)

            if current_max_label >= node_coords.shape[0]:
                new_size = current_max_label + 50000
                new_coords = np.zeros((new_size, 3), dtype=np.float64)
                new_coords[:node_coords.shape[0]] = node_coords
                node_coords = new_coords
            
            node_coords[labels] = data

        v0 = node_coords[connectivity[:, 0]]
        v1 = node_coords[connectivity[:, 1]]
        v2 = node_coords[connectivity[:, 2]]
        v3 = node_coords[connectivity[:, 3]]

        diag1 = v2 - v0
        diag2 = v3 - v1
        cross_prod = np.cross(diag1, diag2)
        
        norms = np.linalg.norm(cross_prod, axis=1)
        elem_area = 0.5 * norms
        
        valid_mask = norms > 1e-12
        unit_normals = np.zeros_like(cross_prod)
        unit_normals[valid_mask] = cross_prod[valid_mask] / norms[valid_mask, np.newaxis]

        z_neg_mask = unit_normals[:, 2] < 0.0
        unit_normals[z_neg_mask] *= -1.0

        cos_theta = np.dot(unit_normals, -U_SUN)
        cos_theta = np.clip(cos_theta, 0.0, 1.0)

        force_abs = A0 * cos_theta[:, np.newaxis] * elem_area[:, np.newaxis] * U_SUN
        force_refl = 2 * R0 * (cos_theta**2)[:, np.newaxis] * elem_area[:, np.newaxis] * (-unit_normals)

        L_eff_vec = np.sum(force_abs + force_refl, axis=0) / np.sum(elem_area * unit_normals[:, 2])
        thrust_val = np.dot(L_eff_vec, U_SUN)

        flatness_val = np.sum(elem_area * unit_normals[:, 2]) / np.sum(elem_area)
        eff_val = (thrust_val / L_ideal) * 100.0

        time_steps.append(current_time)
        magnitudes.append(thrust_val)
        flatness_factors.append(flatness_val)
        efficiencies.append(eff_val)

    odb.close()
    return time_steps, magnitudes, flatness_factors, efficiencies

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