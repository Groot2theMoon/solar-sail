from odbAccess import *
import csv
import os

odb_path = 'Job-1.odb' 

step_name = 'Step-1'

output_csv_file = 'negative_min_principal_stresses.csv'

try:
    odb = openOdb(path=odb_path, readOnly=True)
except Exception as e:
    print(f"Error opening ODB file: {e}")
    exit()

try:
    
    step = odb.steps[step_name]
    frame = step.frames[-1]
    

    stress_field = frame.fieldOutputs['S']
    

    stress_field_at_ip = stress_field.getSubset(position=INTEGRATION_POINT)


    negative_stress_data = []

    negative_stress_data.append(['Element_ID', 'Integration_Point', 'Min_InPlane_Principal_Stress (Pa)'])


    for value in stress_field_at_ip.values:

        stress_val = value.minInPlanePrincipal
        

        if stress_val < 0.0:
           
            negative_stress_data.append([value.elementLabel, 
                                         value.integrationPoint, 
                                         stress_val])

    odb.close()

 
    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(negative_stress_data)
        
    print(f"Success! Negative stress data written to: {os.path.abspath(output_csv_file)}")

except KeyError:
    print(f"Error: Could not find Step '{step_name}' or Field 'S' in the ODB.")
    odb.close()
except Exception as e:
    print(f"An error occurred: {e}")
    odb.close()
