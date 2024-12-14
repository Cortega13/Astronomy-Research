"""
Lower clipping the dataset, removing rounded duplicates, keeping constant time diff for emulator dataset, and creating the training/validation split.
INPUT: uclchem_rawdata.h5
OUTPUT: training.h5, validation.h5
"""

import pandas as pd
import numpy as np
import os
from math import log10, floor
from numba import njit
np.random.seed(13)

TRAINING_SPLIT = 0.7
LOWER_CLIPPING_THRESHOLD = np.float32(1E-20)
WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/datasets/"
METADATA = ["Time", "Model"]
TOTAL_SPECIES = np.loadtxt("C:/Users/carlo/Projects/Astronomy Research/Main Scripts 11.9.24/utils/species.txt", dtype=str, delimiter=" ").tolist()

df = pd.read_hdf(os.path.join(WORKING_PATH, "uclchem_rawdata.h5"), "models", start=0, dtype=np.float32)
df = df.astype(np.float32)
df.reset_index(drop=True, inplace=True)
df.sort_values(by=['Model', 'Time'], inplace=True)
print("-=+=- Dataset Loaded -=+=-")
print(f"Original Total Dataset Size: {len(df)}")

### Apply clipping to entire dataset.
df = df.clip(lower=LOWER_CLIPPING_THRESHOLD)
df.infer_objects(copy=False)
print("-=+=- Dataset Clipped -=+=-")

### Generate the training and validation model identifiers.
model_numbers_list = df['Model'].unique()
print(f"Total Number of Models: {len(model_numbers_list)}")
shuffled_model_numbers = np.random.permutation(model_numbers_list)
training_validation_split = int(len(shuffled_model_numbers) * TRAINING_SPLIT)

training_model_ids = shuffled_model_numbers[:training_validation_split]
validation_model_ids = shuffled_model_numbers[training_validation_split:]
print("-=+=- Training and Validation Model Identifiers Split -=+=-")


######### Autoencoder #########
@njit
def round_sig(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

@njit
def round_array(arr, sig=3):
    result = np.empty_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i, j] = round_sig(arr[i, j], sig)
    return result

### Remove rounded duplicates.
species_only = df[METADATA + TOTAL_SPECIES]
species_only_np = species_only.to_numpy()
rounded_species_np = round_array(species_only_np, sig=3)
rounded_species = pd.DataFrame(rounded_species_np, columns=species_only.columns, dtype=np.float32)
duplicates_mask = rounded_species.duplicated(keep="first")
df_rmduplicates = species_only[~duplicates_mask]
del species_only, species_only_np, rounded_species_np, rounded_species, duplicates_mask
print("-=+=- Autoencoder | Duplicates Removed -=+=-")

### Filter autoencoder training set and validation set.
autoencoder_training_models = df_rmduplicates[df_rmduplicates['Model'].isin(training_model_ids)]
autoencoder_validation_models = df_rmduplicates[df_rmduplicates['Model'].isin(validation_model_ids)]
print("-=+=- Autoencoder | Training & Validation Split -=+=-")

print(f"Autoencoder Total Dataset Size: {len(df_rmduplicates)} | Percentage: {len(df_rmduplicates) / len(df) * 100:.2f}%")
print("training shape", autoencoder_training_models.shape)
print("validation shape", autoencoder_validation_models.shape)

### Reset Index
autoencoder_training_models.reset_index(drop=True, inplace=True)
autoencoder_validation_models.reset_index(drop=True, inplace=True)

### Save the autoencoder data to h5 files. 
autoencoder_training_models.to_hdf(os.path.join(WORKING_PATH, "training.h5"), key="autoencoder", mode="a")
autoencoder_validation_models.to_hdf(os.path.join(WORKING_PATH, "validation.h5"), key="autoencoder", mode="a")

del df_rmduplicates, autoencoder_training_models, autoencoder_validation_models



######### Emulator #########
### Emulator Training-Validation split.

### Adjust for constant timestep differences per model.
def filter_constant_timesteps(df, timestep=1000):
    df['diffs'] = df['Time'].diff().fillna(timestep)
    df['is_new_group'] = df['diffs'] != timestep
    df['temp_group'] = df['is_new_group'].cumsum()

    group_sizes = df.groupby('temp_group').size()
    max_group = group_sizes.idxmax()

    group_indices = df[df['temp_group'] == max_group].index
    start_index = group_indices[0]
    end_index = group_indices[-1]
    filtered_df = df.loc[start_index:end_index].drop(columns=['diffs', 'is_new_group', 'temp_group'])
    return filtered_df

df_constant_dt = df.groupby('Model').apply(filter_constant_timesteps).reset_index(drop=True)


### Filter emulator training set and validation set.
emulator_training_models = df_constant_dt[df_constant_dt["Model"].isin(training_model_ids)]
emulator_validation_models = df_constant_dt[df_constant_dt["Model"].isin(validation_model_ids)]

print(f"Emulator Total Dataset Size: {len(df_constant_dt)} | Percentage: {len(df_constant_dt) / len(df) * 100:.2f}%")
print("training shape", emulator_training_models.shape)
print("validation shape", emulator_validation_models.shape)

### Reset Index
emulator_training_models.reset_index(drop=True, inplace=True)
emulator_validation_models.reset_index(drop=True, inplace=True)

### Save the emulator data to h5 files. 
emulator_training_models.to_hdf(os.path.join(WORKING_PATH, "training.h5"), key="emulator", mode="a")
emulator_validation_models.to_hdf(os.path.join(WORKING_PATH, "validation.h5"), key="emulator", mode="a")

# Autoencoder Total Dataset Size: 11188405 | Percentage: 98.25%
# training shape (7831606, 334)
# validation shape (3356799, 334)
# Emulator Total Dataset Size: 9932426 | Percentage: 87.22%
# training shape (6953053, 340)
# validation shape (2979373, 340)