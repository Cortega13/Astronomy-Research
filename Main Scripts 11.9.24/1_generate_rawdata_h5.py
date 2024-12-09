"""
Compressing the raw csv data into a single h5 file. 
INPUT: modelfiles_5D
OUTPUT: uclchem_rawdata.h5
"""
import pandas as pd
import numpy as np
import os

work_path = "/work2/09338/carlos9/frontera/"
data_folder_path = os.path.join(work_path, "modelfiles_5D")

h5_store_path = os.path.join(work_path, "uclchem_rawdata.h5")
h5_store = pd.HDFStore(h5_store_path)

def rename_columns(columns):
    return [col.replace('#', 'Num').replace('+', 'Plus').replace('@', 'At').replace('-', '_minus').replace(' ', '') for col in columns]

files_list_in_data_folder = os.listdir(data_folder_path)

batch_size = 1000
batch_data = []

for i, file in enumerate(files_list_in_data_folder):
    if i % 100 == 0:
        print(f"Currently on Model: {i}")

    file_path = os.path.join(data_folder_path, file)
    
    with open(file_path, 'r') as f:
        header1 = f.readline().strip()
        header2 = f.readline().strip()

    header_parameters = header1 + " " + header2
    temp = "".join(header_parameters.split(":")).split()[2:]
    
    single_model_data = pd.read_csv(file_path, header=2)
    single_model_data["Radfield"] = np.float32(float(temp[-1]))
    single_model_data["Model"] = i

    single_model_data.columns = rename_columns(single_model_data.columns)

    single_model_data = single_model_data.drop(columns=["zeta", "E_minus"], errors='ignore')
    single_model_data = single_model_data.astype(np.float32)
    single_model_data["Model"] = single_model_data["Model"].astype(int)
    
    single_model_data = single_model_data.drop(columns=["point"])

    batch_data.append(single_model_data)
    
    if (i + 1) % batch_size == 0 or (i + 1) == len(files_list_in_data_folder):
        combined_data = pd.concat(batch_data)
        h5_store.append('models', combined_data, format='table')
        batch_data = []

print("Raw Data Saving Completed")
h5_store.close()
