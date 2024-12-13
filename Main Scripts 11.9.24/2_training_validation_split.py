"""
Lower clipping the dataset, removing rounded duplicates, keeping constant time diff for emulator dataset, and creating the training/validation split.
INPUT: uclchem_rawdata.h5
OUTPUT: training.h5, validation.h5
"""

import pandas as pd
import numpy as np
import os
np.random.seed(13)

FRACTION_OF_TOTAL_DATASET = 0.8
TRAINING_SPLIT = 0.7
LOWER_CLIPPING_THRESHOLD = np.float32(1E-20)
WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/datasets/"

df = pd.read_hdf(os.path.join(WORKING_PATH, "uclchem_rawdata.h5"), "models")
df.sort_values(by=['Model', 'Time'], inplace=True)
print("-=+=- Dataset Loaded -=+=-")

### Apply clipping to entire dataset.
df = df.clip(lower=LOWER_CLIPPING_THRESHOLD)
df.infer_objects(copy=False)
print("-=+=- Dataset Clipped -=+=-")

### Generate the training and validation model identifiers.
model_numbers_list = df['Model'].unique()
print(f"Total Number of Models: {len(model_numbers_list)}")
shuffled_model_numbers = np.random.permutation(model_numbers_list)[:int(FRACTION_OF_TOTAL_DATASET*len(model_numbers_list))]
training_validation_split = int(len(shuffled_model_numbers) * TRAINING_SPLIT)

training_model_ids = shuffled_model_numbers[:training_validation_split]
validation_model_ids = shuffled_model_numbers[training_validation_split:]
print("-=+=- Training and Validation Model Identifiers Split -=+=-")


######### Autoencoder #########
### Remove rounded duplicates.
duplicates_mask = df.round(3).duplicated(keep=False)
df_rmduplicates = df[~duplicates_mask]
print("-=+=- Autoencoder | Duplicates Removed -=+=-")


### Filter autoencoder training set and validation set.
autoencoder_training_models = df_rmduplicates[df_rmduplicates['Model'].isin(training_model_ids)]
autoencoder_validation_models = df_rmduplicates[df_rmduplicates['Model'].isin(validation_model_ids)]
print("-=+=- Autoencoder | Training & Validation Split -=+=-")

print("training shape", autoencoder_training_models.shape)
print("validation shape", autoencoder_validation_models.shape)

### Reset Index
autoencoder_training_models.reset_index(drop=True, inplace=True)
autoencoder_validation_models.reset_index(drop=True, inplace=True)

### Save the autoencoder data to h5 files. 
# autoencoder_training_models.to_hdf(os.path.join(WORKING_PATH, "training.h5"), key="autoencoder", mode="a")
# autoencoder_validation_models.to_hdf(os.path.join(WORKING_PATH, "validation.h5"), key="autoencoder", mode="a")


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

print("training shape", emulator_training_models.shape)
print("validation shape", emulator_validation_models.shape)

### Reset Index
emulator_training_models.reset_index(drop=True, inplace=True)
emulator_validation_models.reset_index(drop=True, inplace=True)

### Save the emulator data to h5 files. 
# emulator_training_models.to_hdf(os.path.join(WORKING_PATH, "training.h5"), key="emulator", mode="a")
# emulator_validation_models.to_hdf(os.path.join(WORKING_PATH, "validation.h5"), key="emulator", mode="a")


### Print Lines
print(f"Original Total Dataset Size: {len(df)}")
print(f"Autoencoder Total Dataset Size: {len(df_rmduplicates)} | Percentage: {len(df_rmduplicates) / len(df) * 100:.2f}%")
print(f"Emulator Total Dataset Size: {len(df_constant_dt)} | Percentage: {len(df_constant_dt) / len(df) * 100:.2f}%")

# Original Total Dataset Size: 11387737
# Autoencoder Total Dataset Size: 10845164 | Percentage: 95.24%
# Emulator Total Dataset Size: 9932426 | Percentage: 87.22%