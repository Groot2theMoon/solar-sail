from odbAccess import openOdb, OdbError
from abaqusConstants import INTEGRATION_POINT
import numpy as np

R0 = 0.926256
A0 = 0.073744
U_SUN = np.array([0.0, 0.0, -1.0])

def J_low(odb_path):
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

    try:
        odb = openOdb(path=odb_path, readOnly=True)
        frame = odb.steps['Step-1'].frames[-1] 
        instance = odb.rootAssembly.instances['PART-1-1']
        
        nodes = instance.nodes
        elements = instance.elements

        initial_coords = np.array([node.coordinates for node in nodes], dtype=np.float64)
        all_node_labels = np.array([node.label for node in nodes], dtype=np.int32)
        
        sort_idx = np.argsort(all_node_labels)
        all_node_labels = all_node_labels[sort_idx]
        initial_coords = initial_coords[sort_idx]

        raw_connectivity = np.array([elem.connectivity for elem in elements])
        connectivity = np.searchsorted(all_node_labels, raw_connectivity)

        displacements = np.zeros_like(initial_coords)
        u_field = frame.fieldOutputs['U']
        
        for block in u_field.bulkDataBlocks:
            block_labels = block.nodeLabels
            block_data = block.data
            
            idx_locs = np.searchsorted(all_node_labels, block_labels)
            displacements[idx_locs] = block_data

        final_coords = initial_coords + displacements

        v0 = final_coords[connectivity[:, 0]]
        v1 = final_coords[connectivity[:, 1]]
        v2 = final_coords[connectivity[:, 2]]
        v3 = final_coords[connectivity[:, 3]]

        area_3d_vec = 0.5 * (np.cross(v1 - v0, v2 - v0) + np.cross(v2 - v0, v3 - v0))

        elem_area = np.linalg.norm(area_3d_vec, axis=1)
        
        valid_mask = elem_area > 1e-16
        unit_normals = np.zeros_like(area_3d_vec)
        
        if np.any(valid_mask):
            z_neg_mask = area_3d_vec[:, 2] < 0.0
            area_3d_vec[z_neg_mask] *= -1.0
            unit_normals[valid_mask] = area_3d_vec[valid_mask] / elem_area[valid_mask, np.newaxis]

        cos_theta = np.dot(unit_normals, -U_SUN)
        cos_theta = np.clip(cos_theta, 0.0, 1.0)

        force_abs = A0 * cos_theta[:, np.newaxis] * elem_area[:, np.newaxis] * U_SUN
        force_refl = 2 * R0 * (cos_theta**2)[:, np.newaxis] * elem_area[:, np.newaxis] * unit_normals
        
        total_force_vector = np.sum(force_abs + force_refl, axis=0)

        A_projected = np.sum(area_3d_vec, axis=0)[2] # Z축 투영 면적
        A_wrinkled = np.sum(elem_area)

        if A_projected > 1e-9:
            L_eff_vec = total_force_vector / A_projected
            thrust_magnitude = np.linalg.norm(L_eff_vec)
        else:
            thrust_magnitude = 0.0

        flatness_factor = A_projected / A_wrinkled if A_wrinkled > 1e-9 else 0.0

        odb.close()
        
        return thrust_magnitude, flatness_factor

    except Exception:
        return 0.0, 0.0
