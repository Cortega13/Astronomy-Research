"""
Obtaining the MinMax for every chemical species and physical parameter in the dataset in order to normalize the data.
INPUT: uclchem_rawdata.h5
OUTPUT: abundances.scalers
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from joblib import dump, load
np.random.seed(13)

WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"
METADATA = ["Time", "Model"]
PHYSICAL_PARAMETERS = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/physical_parameters.txt"), dtype=str, delimiter=" ").tolist()
GAS_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/gas_species.txt"), dtype=str, delimiter=" ").tolist()
BULK_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/bulk_species.txt"), dtype=str, delimiter=" ").tolist()
SURFACE_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/surface_species.txt"), dtype=str, delimiter=" ").tolist()
CALCULATION_DIVISIONS = 10

### Getting Dataset Size
with pd.HDFStore(os.path.join(WORKING_PATH, "datasets/uclchem_rawdata.h5"), mode='r') as store:
    dataset_size = int(store.get_storer('models').nrows)
print(f"-=+=- Dataset Size: {dataset_size} -=+=-")

global_min = pd.Series(dtype='float32')
global_max = pd.Series(dtype='float32')

try:
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
except:
    scalers = {}
for i in tqdm(range(CALCULATION_DIVISIONS), desc="Calculating Min-Max Division"):
    start_idx = int((i/CALCULATION_DIVISIONS)*dataset_size)
    stop_idx = int(((i+1)/CALCULATION_DIVISIONS)*dataset_size)
    chunk = pd.read_hdf(os.path.join(WORKING_PATH, "datasets/uclchem_rawdata.h5"), "models", start=start_idx, stop=stop_idx)
    chunk = chunk.drop(columns=METADATA)
    chunk = np.log10(chunk)

    current_min = chunk.min()
    current_max = chunk.max()
    
    global_min = pd.concat([global_min, current_min], axis=1).min(axis=1)
    global_max = pd.concat([global_max, current_max], axis=1).max(axis=1)
    del chunk
min_max = pd.DataFrame([global_min, global_max], index=['Min', 'Max'], dtype='float32')

gas_species_min_max = min_max[GAS_SPECIES]
bulk_species_min_max = min_max[BULK_SPECIES]
surface_species_min_max = min_max[SURFACE_SPECIES]

for physical_parm in PHYSICAL_PARAMETERS:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = min_max[physical_parm].values.reshape(-1, 1).astype(np.float32)
    scaler.fit(data)
    
    scaler.data_min_ = scaler.data_min_.astype(np.float32)
    scaler.data_max_ = scaler.data_max_.astype(np.float32)
    scaler.scale_ = scaler.scale_.astype(np.float32)
    scaler.min_ = scaler.min_.astype(np.float32)
    
    scalers[physical_parm] = scaler

gas_species_min = np.float32(-20)
gas_species_max = np.float32(gas_species_min_max.loc["Max"].max())

bulk_species_min = np.float32(-20)
bulk_species_max = np.float32(bulk_species_min_max.loc["Max"].max())

surface_species_min = np.float32(-20)
surface_species_max = np.float32(surface_species_min_max.loc["Max"].max())


scalers["gas_species"] = (
    gas_species_min,
    gas_species_max
)

scalers["bulk_species"] = (
    bulk_species_min,
    bulk_species_max
)

scalers["surface_species"] = (
    surface_species_min,
    surface_species_max
)

print(scalers)

dump(scalers, os.path.join(WORKING_PATH, "Datasets/scalers.plk"))

