"""

INPUT: training.h5, validation.h5, abundances.scalers
OUTPUT: autoencoder.pth
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import load
import torch
from torch import nn, optim, multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import torch.nn.functional as F
import math
import gc

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
}
emulator_HP = {
    "layer_sizes": [20, 256, 256, 14],
}

METADATA = ["Time", "Model"] 
PHYSICAL_PARAMETERS = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/physical_parameters.txt"), dtype=str, delimiter=" ").tolist()

GAS_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/gas_species.txt"), dtype=str, delimiter=" ").tolist()
BULK_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/bulk_species.txt"), dtype=str, delimiter=" ").tolist()
SURFACE_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/surface_species.txt"), dtype=str, delimiter=" ").tolist()
TOTAL_SPECIES = GAS_SPECIES + BULK_SPECIES + SURFACE_SPECIES


TOTAL_COMPONENTS = [f"Component_{i}" for i in range(1, vaes_HP["gas_encoded_dims"]+vaes_HP["bulk_encoded_dims"]+vaes_HP["surface_encoded_dims"]+1)]
GAS_COMPONENTS = TOTAL_COMPONENTS[:vaes_HP["gas_encoded_dims"]]
BULK_COMPONENTS = TOTAL_COMPONENTS[vaes_HP["gas_encoded_dims"]:vaes_HP["gas_encoded_dims"]+vaes_HP["bulk_encoded_dims"]]
SURFACE_COMPONENTS = TOTAL_COMPONENTS[vaes_HP["gas_encoded_dims"]+vaes_HP["bulk_encoded_dims"]:]

### Data Processing Functions
def load_datasets(path):
    validation_dataset_path = os.path.join(path, "Datasets/validation.h5")
    validation_dataset = pd.read_hdf(validation_dataset_path, "emulator", start=0, stop=50).astype(np.float32).reset_index(drop=True)
    validation_dataset = validation_dataset[METADATA + PHYSICAL_PARAMETERS + TOTAL_SPECIES]
    
    return validation_dataset


def autoencoder_preprocessing(scalers, abundances_features, phase_type):
    print("Starting Encoder Preprocessing.")

    abundances_min, abundances_max = scalers[phase_type]

    abundances_features = np.log10(abundances_features, dtype=np.float32)

    abundances_features = (abundances_features - abundances_min) / (abundances_max - abundances_min)

    abundances_features_tensor = torch.tensor(abundances_features.to_numpy(), dtype=torch.float32)
    
    return abundances_features_tensor


def autoencoder_postprocessing(scalers, features, phase_type, species_columns):
    ### Postprocesses the data for the autoencoder training. Returns a dataloader.
    print("Starting Decoder Postprocessing.")

    # Created using only the training data.
    abundances_min, abundances_max = scalers[phase_type]

    features = features.cpu().detach().numpy()
    features = pd.DataFrame(features, columns=species_columns)
    features = features * (abundances_max - abundances_min) + abundances_min
    features = np.power(10, features, dtype=np.float32)

    return features


def emulator_preprocessing(scalers, encoded_dataset, dataset):
    gas_encoded_min, gas_encoded_max = scalers["gas_species"]
    bulk_encoded_min, bulk_encoded_max = scalers["bulk_species"]
    surface_encoded_min, surface_encoded_max = scalers["surface_species"]
    
    encoded_dataset = encoded_dataset.cpu().detach().numpy()
    encoded_dataset = pd.DataFrame(encoded_dataset, columns=TOTAL_COMPONENTS)
    
    encoded_dataset = pd.concat([dataset[PHYSICAL_PARAMETERS], encoded_dataset], axis=1)
    
    encoded_dataset[GAS_COMPONENTS] = (encoded_dataset[GAS_COMPONENTS] - gas_encoded_min) / (gas_encoded_max - gas_encoded_min)
    encoded_dataset[BULK_COMPONENTS] = (encoded_dataset[BULK_COMPONENTS] - bulk_encoded_min) / (bulk_encoded_max - bulk_encoded_min)
    encoded_dataset[SURFACE_COMPONENTS] = (encoded_dataset[SURFACE_COMPONENTS] - surface_encoded_min) / (surface_encoded_max - surface_encoded_min)
    
    encoded_dataset = encoded_dataset[PHYSICAL_PARAMETERS + TOTAL_COMPONENTS]  
    
    return torch.tensor(encoded_dataset.to_numpy(), dtype=torch.float32)


def emulator_postprocessing(scalers, emulator_outputs):
    gas_encoded_min, gas_encoded_max = scalers["gas_species"]
    bulk_encoded_min, bulk_encoded_max = scalers["bulk_species"]
    surface_encoded_min, surface_encoded_max = scalers["surface_species"]
    
    emulator_outputs = emulator_outputs.cpu().detach().numpy()
    emulator_outputs = pd.DataFrame(emulator_outputs, columns=TOTAL_COMPONENTS)
    
    emulator_outputs[GAS_COMPONENTS] = emulator_outputs[GAS_COMPONENTS] * (gas_encoded_max - gas_encoded_min) + gas_encoded_min
    emulator_outputs[BULK_COMPONENTS] = emulator_outputs[BULK_COMPONENTS] * (bulk_encoded_max - bulk_encoded_min) + bulk_encoded_min
    emulator_outputs[SURFACE_COMPONENTS] = emulator_outputs[SURFACE_COMPONENTS] * (surface_encoded_max - surface_encoded_min) + surface_encoded_min    
    
    return torch.tensor(emulator_outputs.to_numpy(), dtype=torch.float32)  


def create_emulator_dataset(scalers, df, timesteps):
    df = df.copy()
    inputs, outputs = [], []

    df[PHYSICAL_PARAMETERS] = np.log10(df[PHYSICAL_PARAMETERS])

    for parameter in PHYSICAL_PARAMETERS:
        df[parameter] = scalers[parameter].transform(df[parameter].values.reshape(-1, 1))

    for _, sub_df in df.groupby('Model'):
        sub_array = sub_df[PHYSICAL_PARAMETERS + TOTAL_SPECIES].to_numpy()
        
        num_rows = len(sub_array)
        if num_rows > timesteps:
            input_window = sub_array[:-timesteps]
            output_window = sub_array[timesteps:, -len(TOTAL_SPECIES):]
            if timesteps == 0:
                input_window = sub_array
                output_window = sub_array[:, -len(TOTAL_SPECIES):]
            
            inputs.append(input_window)
            outputs.append(output_window)

    inputs = pd.DataFrame(np.vstack(inputs), columns=PHYSICAL_PARAMETERS + TOTAL_SPECIES)
    outputs = pd.DataFrame(np.vstack(outputs), columns=TOTAL_SPECIES)
    return inputs, outputs


### Defining the Variational Autoencoder.
class VariationalAutoencoder(nn.Module):
    def __init__(self, num_features, encoded_dimensions=11, hidden_layer=128, dropout_rate=0.2):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.layer1 = nn.Linear(num_features, hidden_layer)
        self.batch_norm1 = nn.BatchNorm1d(hidden_layer)
        self.dropout1 = nn.Dropout(p=dropout_rate)  # Dropout in the encoder
        self.fc_mu = nn.Linear(hidden_layer, encoded_dimensions)
        self.fc_logvar = nn.Linear(hidden_layer, encoded_dimensions)
        
        # Decoder
        self.layer3 = nn.Linear(encoded_dimensions, hidden_layer)
        self.dropout2 = nn.Dropout(p=dropout_rate)  # Dropout in the decoder
        self.layer4 = nn.Linear(hidden_layer, num_features)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)  # Apply dropout in the encoder
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
        z1 = self.dropout2(z1)  # Apply dropout in the decoder
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


### Defining the Emulator.
class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x


class Emulator(nn.Module):
    def __init__(self, layer_sizes, noise_mean=0, noise_std=0):
        super(Emulator, self).__init__()
        self.gaussiannoise = GaussianNoise(mean=noise_mean, std=noise_std)
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()  # Constrain outputs to [0, 1]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                if i == 0:
                    x = self.gaussiannoise(layer(x))
                else:
                    x = layer(x)
                x = F.relu(x)
            else:
                x = layer(x)
                x = self.final_activation(x)
        return x


def load_objects():
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
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
    
    emulator = Emulator(
        layer_sizes=emulator_HP["layer_sizes"],
    ).to(device)
    emulator.load_state_dict(torch.load(os.path.join(WORKING_PATH, f"Weights/emulator.pth")))
    emulator.eval()
    
    return scalers, gas_autoencoder, bulk_autoencoder, surface_autoencoder, emulator


def main(timestep_multiplier, validation_dataset, batch_size=4192):
    timesteps = 0
    scalers, gas_autoencoder, bulk_autoencoder, surface_autoencoder, emulator = load_objects()
    inputs, outputs = create_emulator_dataset(scalers, validation_dataset, timesteps*timestep_multiplier)
    
    with torch.no_grad():
        preencoded_gas_inputs = autoencoder_preprocessing(scalers, inputs[GAS_SPECIES], "gas_species")
        preencoded_bulk_inputs = autoencoder_preprocessing(scalers, inputs[BULK_SPECIES], "bulk_species")
        preencoded_surface_inputs = autoencoder_preprocessing(scalers, inputs[SURFACE_SPECIES], "surface_species")
        
        ### Encoding Step
        gas_encoded_inputs, bulk_encoded_inputs, surface_encoded_inputs = [], [], []
        for batch_start in range(0, len(preencoded_gas_inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(preencoded_gas_inputs))
            batch = preencoded_gas_inputs[batch_start:batch_end]
            batch = batch.to(device)
            batch_encoded, _ = gas_autoencoder.encode(batch)
            gas_encoded_inputs.append(batch_encoded)
        
        for batch_start in range(0, len(preencoded_bulk_inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(preencoded_bulk_inputs))
            batch = preencoded_bulk_inputs[batch_start:batch_end]
            batch = batch.to(device)
            batch_encoded, _ = bulk_autoencoder.encode(batch)
            bulk_encoded_inputs.append(batch_encoded)
        
        for batch_start in range(0, len(preencoded_surface_inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(preencoded_surface_inputs))
            batch = preencoded_surface_inputs[batch_start:batch_end]
            batch = batch.to(device)
            batch_encoded, _ = surface_autoencoder.encode(batch)
            surface_encoded_inputs.append(batch_encoded)
        
        gas_encoded_inputs = torch.cat(gas_encoded_inputs, dim=0)
        bulk_encoded_inputs = torch.cat(bulk_encoded_inputs, dim=0)
        surface_encoded_inputs = torch.cat(surface_encoded_inputs, dim=0)
        
        total_encoded_inputs = torch.cat([gas_encoded_inputs, bulk_encoded_inputs, surface_encoded_inputs], dim=1)
        
        ### Emulator Step
        for _ in range(timesteps):
            encoded_inputs = emulator_preprocessing(scalers, total_encoded_inputs, inputs)
            
            emulator_outputs = []
            for batch_start in range(0, len(encoded_inputs), batch_size):
                batch_end = min(batch_start + batch_size, len(encoded_inputs))
                batch = encoded_inputs[batch_start:batch_end]
                batch = batch.to(device)
                batch_outputs = emulator(batch)
                emulator_outputs.append(batch_outputs)
            emulator_outputs = torch.cat(emulator_outputs, dim=0)
            encoded_inputs = emulator_postprocessing(scalers, emulator_outputs)

        ### Decoding Step
        gas_outputs = encoded_inputs[:, :vaes_HP["gas_encoded_dims"]]
        bulk_outputs = encoded_inputs[:, vaes_HP["gas_encoded_dims"]:vaes_HP["gas_encoded_dims"]+vaes_HP["bulk_encoded_dims"]]
        surface_outputs = encoded_inputs[:, vaes_HP["gas_encoded_dims"]+vaes_HP["bulk_encoded_dims"]:]
        
        gas_decoded_outputs, bulk_decoded_outputs, surface_decoded_outputs = [], [], []
        for batch_start in range(0, len(gas_outputs), batch_size):
            batch_end = min(batch_start + batch_size, len(gas_outputs))
            batch = gas_outputs[batch_start:batch_end]
            batch = batch.to(device)
            batch_decoded = gas_autoencoder.decode(batch)
            gas_decoded_outputs.append(batch_decoded)
        
        for batch_start in range(0, len(bulk_outputs), batch_size):
            batch_end = min(batch_start + batch_size, len(bulk_outputs))
            batch = bulk_outputs[batch_start:batch_end]
            batch = batch.to(device)
            batch_decoded = bulk_autoencoder.decode(batch)
            bulk_decoded_outputs.append(batch_decoded)
        
        for batch_start in range(0, len(surface_outputs), batch_size):
            batch_end = min(batch_start + batch_size, len(surface_outputs))
            batch = surface_outputs[batch_start:batch_end]
            batch = batch.to(device)
            batch_decoded = surface_autoencoder.decode(batch)
            surface_decoded_outputs.append(batch_decoded)            
        
        gas_decoded_outputs = torch.cat(gas_decoded_outputs, dim=0)
        bulk_decoded_outputs = torch.cat(bulk_decoded_outputs, dim=0)
        surface_decoded_outputs = torch.cat(surface_decoded_outputs, dim=0)
        
        gas_decoded_outputs = autoencoder_postprocessing(scalers, gas_decoded_outputs, "gas_species", GAS_SPECIES)
        bulk_decoded_outputs = autoencoder_postprocessing(scalers, bulk_decoded_outputs, "bulk_species", BULK_SPECIES)
        surface_decoded_outputs = autoencoder_postprocessing(scalers, surface_decoded_outputs, "surface_species", SURFACE_SPECIES)
        
        final_outputs = pd.concat([gas_decoded_outputs, bulk_decoded_outputs, surface_decoded_outputs], axis=1)


    ### Error Analysis
    percent_error = ((abs(outputs[TOTAL_SPECIES] - final_outputs[TOTAL_SPECIES])) / outputs[TOTAL_SPECIES])
    print()
    print(percent_error.mean().sort_values(ascending=False))
    print(f"Average Error: {percent_error.mean().mean():.4e}")
    print(f"STD Error: {percent_error.mean().std():.4e}")
    gc.collect()

if __name__ == "__main__":
    if torch.cuda.is_available():
        start_time = datetime.now()
        WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"

        validation_dataset = load_datasets(WORKING_PATH)
        main(1, validation_dataset)
        print("Time Elapsed: ", datetime.now() - start_time)
    else:
        print("CUDA is not available.")
