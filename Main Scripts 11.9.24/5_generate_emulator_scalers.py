"""

INPUT: training.h5, validation.h5, abundances.scalers
OUTPUT: autoencoder.pth
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import load, dump
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Configurations
WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"
vaes_HP = {
    "gas_encoded_dims": 6,
    "gas_hidden_dims": 400,
    "bulk_encoded_dims": 4,
    "bulk_hidden_dims": 200,
    "surface_encoded_dims": 4,
    "surface_hidden_dims": 600,
    "batch_size": 1024
}

METADATA = ["Time", "Model"] 
PHYSICAL_PARAMETERS = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/physical_parameters.txt"), dtype=str, delimiter=" ").tolist()

GAS_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/gas_species.txt"), dtype=str, delimiter=" ").tolist()
BULK_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/bulk_species.txt"), dtype=str, delimiter=" ").tolist()
SURFACE_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/surface_species.txt"), dtype=str, delimiter=" ").tolist()

TOTAL_COMPONENTS = [f"Component_{i}" for i in range(1, vaes_HP["gas_encoded_dims"]+vaes_HP["bulk_encoded_dims"]+vaes_HP["surface_encoded_dims"]+1)]
GAS_COMPONENTS = TOTAL_COMPONENTS[:vaes_HP["gas_encoded_dims"]]
BULK_COMPONENTS = TOTAL_COMPONENTS[vaes_HP["gas_encoded_dims"]:vaes_HP["gas_encoded_dims"]+vaes_HP["bulk_encoded_dims"]]
SURFACE_COMPONENTS = TOTAL_COMPONENTS[vaes_HP["gas_encoded_dims"]+vaes_HP["bulk_encoded_dims"]:]

### Data Processing Functions
def load_datasets(path):
    training_dataset_path = os.path.join(path, "Datasets/training.h5")
    validation_dataset_path = os.path.join(path, "Datasets/validation.h5")

    training_dataset = pd.read_hdf(training_dataset_path, "emulator", start=0).astype(np.float32).reset_index(drop=True)
    validation_dataset = pd.read_hdf(validation_dataset_path, "emulator", start=0).astype(np.float32).reset_index(drop=True)
    
    return training_dataset, validation_dataset


def autoencoder_preprocessing(abundances_features, phase_type):
    ### Preprocesses the data for the autoencoder training. Returns a dataloader.
    print("Starting Encoder Preprocessing.")

    # Created using only the training data.
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
    abundances_min, abundances_max = scalers[phase_type]

    # Log10 Scale Abundances and then MinMax scale.
    abundances_features = np.log10(abundances_features, dtype=np.float32)

    # Minmax scale all the abundances_features.
    abundances_features = (abundances_features - abundances_min) / (abundances_max - abundances_min)

    #Convert the abundances_features dataframe to tensor format for inferencing.
    abundances_features_tensor = torch.tensor(abundances_features.to_numpy(), dtype=torch.float32)
    
    abundances_features_dataloader = DataLoader(TensorDataset(abundances_features_tensor), batch_size=vaes_HP["batch_size"], shuffle=False)

    return abundances_features_dataloader


def reconstruct_dataset_dataframe(encoded_dataset, dataset):    
    encoded_dataset = encoded_dataset.cpu().detach().numpy()
    encoded_dataset = pd.DataFrame(encoded_dataset, columns=TOTAL_COMPONENTS)
    
    encoded_dataset = pd.concat([dataset[PHYSICAL_PARAMETERS], encoded_dataset], axis=1)
    encoded_dataset = pd.concat([dataset[METADATA], encoded_dataset], axis=1)
    
    return encoded_dataset


def create_emulator_dataset(df, timesteps=1):
    print("-=+=- Creating Emulator Dataset (input, output) -=+=-")
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
    gas_encoded_min, gas_encoded_max = scalers["gas_species"]
    bulk_encoded_min, bulk_encoded_max = scalers["bulk_species"]
    surface_encoded_min, surface_encoded_max = scalers["surface_species"]

    df[PHYSICAL_PARAMETERS] = np.log10(df[PHYSICAL_PARAMETERS])
    for parameter in PHYSICAL_PARAMETERS:
        df[parameter] = scalers[parameter].transform(df[parameter].values.reshape(-1, 1))

    df[GAS_COMPONENTS] = (df[GAS_COMPONENTS] - gas_encoded_min) / (gas_encoded_max - gas_encoded_min)
    df[BULK_COMPONENTS] = (df[BULK_COMPONENTS] - bulk_encoded_min) / (bulk_encoded_max - bulk_encoded_min)
    df[SURFACE_COMPONENTS] = (df[SURFACE_COMPONENTS] - surface_encoded_min) / (surface_encoded_max - surface_encoded_min)
    
    inputs, outputs = [], []
    
    for _, sub_df in df.groupby('Model'):
        differences = (sub_df["Time"].diff()).dropna()
        if not (differences == 1000).all():
            print("Time differences not equal to 1000.")
            continue
        sub_array = sub_df[PHYSICAL_PARAMETERS + TOTAL_COMPONENTS].to_numpy()
        
        num_rows = len(sub_array)
        if num_rows > timesteps:
            input_window = sub_array[:-timesteps]
            output_window = sub_array[timesteps:, -len(TOTAL_COMPONENTS):]
            if timesteps == 0:
                input_window = sub_array
                output_window = sub_array[:, -len(TOTAL_COMPONENTS):]
            
            inputs.append(input_window)
            outputs.append(output_window)

    inputs = torch.tensor(np.vstack(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.vstack(outputs), dtype=torch.float32)
    return inputs, outputs



### Defining the Variational Autoencoder.
class VariationalAutoencoder(nn.Module):
    def __init__(self, num_features, encoded_dimensions=0, hidden_layer=0):
        super(VariationalAutoencoder, self).__init__()        
        self.layer1 = nn.Linear(num_features, hidden_layer)
        self.fc_mu = nn.Linear(hidden_layer, encoded_dimensions)
        self.fc_logvar = nn.Linear(hidden_layer, encoded_dimensions)
        self.layer3 = nn.Linear(encoded_dimensions, hidden_layer)
        self.layer4 = nn.Linear(hidden_layer, num_features)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm1 = nn.BatchNorm1d(hidden_layer)

    def encode(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z1 = self.layer3(z)
        z1 = F.relu(z1)
        z2 = self.layer4(z1)
        z3 = self.sigmoid(z2)
        return z3

    def forward(self, x, is_inference=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if is_inference:
            return self.decode(mu)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar


def encode_dataset(dataset):
    print("-=+=- Encoding Dataset -=+=-")
    gas_autoencoder = VariationalAutoencoder(
        num_features=len(GAS_SPECIES),
        encoded_dimensions=vaes_HP["gas_encoded_dims"],
        hidden_layer=vaes_HP["gas_hidden_dims"],
    ).to(device)
    gas_autoencoder.load_state_dict(torch.load(os.path.join(WORKING_PATH, "Weights/gas_species_vae.pth")))
    gas_autoencoder.eval()
    
    bulk_autoencoder = VariationalAutoencoder(
        num_features=len(BULK_SPECIES),
        encoded_dimensions=vaes_HP["bulk_encoded_dims"],
        hidden_layer=vaes_HP["bulk_hidden_dims"],
    ).to(device)
    bulk_autoencoder.load_state_dict(torch.load(os.path.join(WORKING_PATH, "Weights/bulk_species_vae.pth")))
    bulk_autoencoder.eval()
    
    surface_autoencoder = VariationalAutoencoder(
        num_features=len(SURFACE_SPECIES),
        encoded_dimensions=vaes_HP["surface_encoded_dims"],
        hidden_layer=vaes_HP["surface_hidden_dims"],
    ).to(device)
    surface_autoencoder.load_state_dict(torch.load(os.path.join(WORKING_PATH, "Weights/surface_species_vae.pth")))
    surface_autoencoder.eval()

    gas_species_dataloader = autoencoder_preprocessing(dataset[GAS_SPECIES], "gas_species")
    bulk_species_dataloader = autoencoder_preprocessing(dataset[BULK_SPECIES], "bulk_species")
    surface_species_dataloader = autoencoder_preprocessing(dataset[SURFACE_SPECIES], "surface_species")

    gas_components = []
    bulk_components = []
    surface_components = []
    
    with torch.no_grad():
        for batch in gas_species_dataloader:
            batch_tensor = batch[0].to(device)
            encoded_batch, _ = gas_autoencoder.encode(batch_tensor)
            gas_components.append(encoded_batch.cpu())
        
        for batch in bulk_species_dataloader:
            batch_tensor = batch[0].to(device)
            encoded_batch, _ = bulk_autoencoder.encode(batch_tensor)
            bulk_components.append(encoded_batch.cpu())
        
        for batch in surface_species_dataloader:
            batch_tensor = batch[0].to(device)
            encoded_batch, _ = surface_autoencoder.encode(batch_tensor)
            surface_components.append(encoded_batch.cpu())

    gas_components = torch.cat(gas_components, dim=0)
    bulk_components = torch.cat(bulk_components, dim=0)
    surface_components = torch.cat(surface_components, dim=0)
    
    encoded_combined_dataset = torch.cat([gas_components, bulk_components, surface_components], dim=1)
        
    encoded_dataset = reconstruct_dataset_dataframe(encoded_combined_dataset, dataset)
    
    return encoded_dataset


def calculate_encoded_scalers(training_dataset, validation_dataset):
    training_min = training_dataset.min().min()
    training_max = training_dataset.max().max()
    
    validation_min = validation_dataset.min().min()
    validation_max = validation_dataset.max().max()
    
    encoded_min = min(training_min, validation_min)
    encoded_max = max(training_max, validation_max)
    
    return np.float32(encoded_min), np.float32(encoded_max)



if __name__ == "__main__":
    if torch.cuda.is_available():        
        start_time = datetime.now()
        WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"
        
        scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))

        training_dataset, validation_dataset = load_datasets(WORKING_PATH)
        encoded_training = encode_dataset(training_dataset)
        encoded_validation = encode_dataset(validation_dataset)
        del training_dataset, validation_dataset

        gas_components_min, gas_components_max = calculate_encoded_scalers(encoded_training[GAS_COMPONENTS], encoded_validation[GAS_COMPONENTS])
        bulk_components_min, bulk_components_max = calculate_encoded_scalers(encoded_training[BULK_COMPONENTS], encoded_validation[BULK_COMPONENTS])
        surface_encoded_min, surface_components_max = calculate_encoded_scalers(encoded_training[SURFACE_COMPONENTS], encoded_validation[SURFACE_COMPONENTS])
        
        scalers["gas_components"] = (
            gas_components_min,
            gas_components_max
        )
        scalers["bulk_components"] = (
            bulk_components_min,
            bulk_components_max
        )
        scalers["surface_components"] = (
            surface_encoded_min,
            surface_components_max
        )
        
        print()
        print(f"Gas Components Min/Max: {gas_components_min} / {gas_components_max}")
        print(f"Bulk Components Min/Max: {bulk_components_min} / {bulk_components_max}")
        print(f"Surface Components Min/Max: {surface_encoded_min} / {surface_components_max}")
        print()
        print(scalers)
        
        dump(scalers, os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
        
        print("Time Elapsed: ", datetime.now() - start_time)
    else:
        print("CUDA is not available.")