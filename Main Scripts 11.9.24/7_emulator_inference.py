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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### Configurations
WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"
HP = {
    "encoded_dimensions": 11,
    "layer_sizes": [15, 256, 256, 11],
    "hidden_layer": 200
}

METADATA = ["Time", "Model"] 
PHYSICAL_PARAMETERS = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/physical_parameters.txt"), dtype=str, delimiter=" ").tolist()
TOTAL_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/species.txt"), dtype=str, delimiter=" ").tolist()
COMPONENTS = [f"Component_{i}" for i in range(1, HP["encoded_dimensions"]+1)]

### Data Processing Functions
def load_datasets(path):
    validation_dataset_path = os.path.join(path, "Datasets/validation.h5")
    validation_dataset = pd.read_hdf(validation_dataset_path, "emulator", start=0, stop=5000).astype(np.float32).reset_index(drop=True)
    validation_dataset = validation_dataset[METADATA + PHYSICAL_PARAMETERS + TOTAL_SPECIES]
    
    return validation_dataset


def autoencoder_preprocessing(scalers, abundances_features):
    ### Preprocesses the data for the autoencoder training. Returns a dataloader.
    print("Starting Encoder Preprocessing.")

    # Created using only the training data.
    abundances_min, abundances_max = scalers["total_species"]

    # Log10 Scale Abundances and then MinMax scale.
    abundances_features = np.log10(abundances_features, dtype=np.float32)

    # Minmax scale all the abundances_features.
    abundances_features = (abundances_features - abundances_min) / (abundances_max - abundances_min)

    #Convert the abundances_features dataframe to tensor format for inferencing.
    abundances_features_tensor = torch.tensor(abundances_features.to_numpy(), dtype=torch.float32)

    return abundances_features_tensor


def autoencoder_postprocessing(scalers, features):
    ### Postprocesses the data for the autoencoder training. Returns a dataloader.
    print("Starting Decoder Postprocessing.")

    # Created using only the training data.
    abundances_min, abundances_max = scalers["total_species"]

    features = features.cpu().detach().numpy()
    features = pd.DataFrame(features, columns=TOTAL_SPECIES)
    features = features * (abundances_max - abundances_min) + abundances_min
    features = np.power(10, features, dtype=np.float32)

    return features


def emulator_preprocessing(scalers, encoded_dataset, dataset):
    encoded_min, encoded_max = scalers["encoded_components"]
    encoded_dataset = encoded_dataset.cpu().detach().numpy()
    encoded_dataset = pd.DataFrame(encoded_dataset, columns=COMPONENTS)
    
    encoded_dataset = pd.concat([dataset[PHYSICAL_PARAMETERS], encoded_dataset], axis=1)
    
    encoded_dataset[COMPONENTS] = (encoded_dataset[COMPONENTS] - encoded_min) / (encoded_max - encoded_min)
    encoded_dataset = encoded_dataset[PHYSICAL_PARAMETERS + COMPONENTS]  
    
    return torch.tensor(encoded_dataset.to_numpy(), dtype=torch.float32)


def emulator_postprocessing(scalers, emulator_outputs):
    encoded_min, encoded_max = scalers["encoded_components"]
    
    emulator_outputs = emulator_outputs.cpu().detach().numpy()
    emulator_outputs = pd.DataFrame(emulator_outputs, columns=COMPONENTS)
    emulator_outputs = emulator_outputs * (encoded_max - encoded_min) + encoded_min
    return torch.tensor(emulator_outputs.to_numpy(), dtype=torch.float32)  


def create_emulator_dataset(scalers, df, timesteps):
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
    autoencoder = VariationalAutoencoder(
        num_features=len(TOTAL_SPECIES),
        encoded_dimensions=HP["encoded_dimensions"],
        hidden_layer=HP["hidden_layer"]
    ).to(device)
    autoencoder.load_state_dict(torch.load(os.path.join(WORKING_PATH, "Weights/vae.pth")))
    autoencoder.eval()
    
    emulator = Emulator(
        layer_sizes=HP["layer_sizes"],
    ).to(device)
    emulator.load_state_dict(torch.load(os.path.join(WORKING_PATH, "Weights/emulator.pth")))
    emulator.eval()
    
    return autoencoder, emulator, scalers


def main(validation_dataset, batch_size=4192):
    timesteps = 1
    autoencoder, emulator, scalers = load_objects()
    inputs, validation = create_emulator_dataset(scalers, validation_dataset, timesteps)
    
    with torch.no_grad():
        preencoded_inputs = autoencoder_preprocessing(scalers, inputs[TOTAL_SPECIES])
        encoded_inputs = []
        for batch_start in range(0, len(preencoded_inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(preencoded_inputs))
            batch = preencoded_inputs[batch_start:batch_end]
            batch = batch.to(device)
            batch_encoded, _ = autoencoder.encode(batch)
            #batch_encoded = batch+encoded + torch.randn_like(batch_encoded) * 0.1
            encoded_inputs.append(batch_encoded)
        encoded_inputs = torch.cat(encoded_inputs, dim=0)
        
        for _ in range(timesteps):
            encoded_inputs = emulator_preprocessing(scalers, encoded_inputs, inputs)
            
            emulator_outputs = []
            for batch_start in range(0, len(encoded_inputs), batch_size):
                batch_end = min(batch_start + batch_size, len(encoded_inputs))
                batch = encoded_inputs[batch_start:batch_end]
                batch = batch.to(device)
                batch_outputs = emulator(batch)
                emulator_outputs.append(batch_outputs)
            emulator_outputs = torch.cat(emulator_outputs, dim=0)
            encoded_inputs = emulator_postprocessing(scalers, emulator_outputs)

        decoded_outputs = []
        for batch_start in range(0, len(encoded_inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(encoded_inputs))
            batch = encoded_inputs[batch_start:batch_end]
            batch = batch.to(device)
            batch_decoded = autoencoder.decode(batch)
            decoded_outputs.append(batch_decoded)
        decoded_outputs = torch.cat(decoded_outputs, dim=0)

        decoded_outputs = autoencoder_postprocessing(scalers, decoded_outputs)

    percent_error = ((abs(validation[TOTAL_SPECIES] - decoded_outputs[TOTAL_SPECIES])) / validation[TOTAL_SPECIES])
    print("Error", percent_error.mean().sort_values(ascending=True).iloc[-40:20])
    print(validation["NH"])
    print(decoded_outputs["NH"])
    print(f"Average Error: {percent_error.mean().mean():.4e}")
    print(f"STD Error: {percent_error.mean().std():.4e}")


if __name__ == "__main__":
    if torch.cuda.is_available():        
        start_time = datetime.now()
        WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"

        validation_dataset = load_datasets(WORKING_PATH)
        main(validation_dataset)
        print("Time Elapsed: ", datetime.now() - start_time)
    else:
        print("CUDA is not available.")
