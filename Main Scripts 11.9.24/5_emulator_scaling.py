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
HP = {
    "encoded_dimensions": 11,
    "learning_rate": 0.001,
    "weight_decay": 0,
    "batch_size": 512,
    "shuffle": True,
    "early_stopping_tolerance": 7,
    "max_epochs": 999999,
    "gaussian_noise_std": 0,
    "layer_sizes": [15, 40, 40, 40, 11],
    "hidden_layer": 1800
}

METADATA = ["Time", "Model"] 
PHYSICAL_PARAMETERS = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/physical_parameters.txt"), dtype=str, delimiter=" ").tolist()
TOTAL_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/species.txt"), dtype=str, delimiter=" ").tolist()
COMPONENTS = [f"Component_{i}" for i in range(1, HP["encoded_dimensions"]+1)]

### Data Processing Functions
def load_datasets(path):
    training_dataset_path = os.path.join(path, "Datasets/training.h5")
    validation_dataset_path = os.path.join(path, "Datasets/validation.h5")

    training_dataset = pd.read_hdf(training_dataset_path, "emulator").astype(np.float32).reset_index(drop=True)
    validation_dataset = pd.read_hdf(validation_dataset_path, "emulator").astype(np.float32).reset_index(drop=True)
    
    return training_dataset, validation_dataset


def autoencoder_preprocessing(abundances_features):
    ### Preprocesses the data for the autoencoder training. Returns a dataloader.
    print("Starting Encoder Preprocessing.")

    # Created using only the training data.
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
    abundances_min, abundances_max = scalers["total_species"]

    # Log10 Scale Abundances and then MinMax scale.
    abundances_features = np.log10(abundances_features, dtype=np.float32)

    # Minmax scale all the abundances_features.
    abundances_features = (abundances_features - abundances_min) / (abundances_max - abundances_min)

    #Convert the abundances_features dataframe to tensor format for inferencing.
    abundances_features_tensor = torch.tensor(abundances_features.to_numpy(), dtype=torch.float32)

    return abundances_features_tensor


def reconstruct_dataset_dataframe(encoded_dataset, dataset):
    component_columns = [f"Component_{i}" for i in range(1, HP["encoded_dimensions"]+1)]
    
    encoded_dataset = encoded_dataset.cpu().detach().numpy()
    encoded_dataset = pd.DataFrame(encoded_dataset, columns=component_columns)
    
    encoded_dataset = pd.concat([dataset[PHYSICAL_PARAMETERS], encoded_dataset], axis=1)
    encoded_dataset = pd.concat([dataset[METADATA], encoded_dataset], axis=1)
    
    return encoded_dataset


def create_emulator_dataset(df):
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
    inputs = []
    outputs = []
    def create_timestep_pairs(df):
        df.drop(columns=["Model"], inplace=True)
        for parameter in PHYSICAL_PARAMETERS:
            df[parameter] = np.log10(df[parameter].values)
            df[parameter] = scalers[parameter].transform(df[parameter].values.reshape(-1, 1))
        print(df.head())
        pairs = [(df.iloc[i], df.iloc[i+1]) for i in range(len(df)-1)]
        inputs.extend(pair[0].to_numpy() for pair in pairs)
        outputs.extend(pair[1][COMPONENTS].to_numpy() for pair in pairs)
        #print(pairs[0][0].to_numpy(), pairs[0][1][COMPONENTS].to_numpy())
    df.groupby('Model').apply(create_timestep_pairs).reset_index(drop=True)
    
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.array(outputs), dtype=torch.float32)
    return inputs, outputs


### Defining the Variational Autoencoder.
class VariationalAutoencoder(nn.Module):
    def __init__(self, num_features, encoded_dimensions=11, hidden_layer=HP["hidden_layer"]):
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
    autoencoder = VariationalAutoencoder(
        num_features=len(TOTAL_SPECIES),
        encoded_dimensions=HP["encoded_dimensions"],
    ).to(device)
    autoencoder.load_state_dict(torch.load(os.path.join(WORKING_PATH, "Weights/vae.pth")))
    autoencoder.eval()

    dataset_tensor = autoencoder_preprocessing(dataset[TOTAL_SPECIES])
    dataset_loader = DataLoader(TensorDataset(dataset_tensor), batch_size=HP["batch_size"], shuffle=False)
    
    encoded_batches = []
    
    with torch.no_grad():
        for batch in dataset_loader:
            batch_tensor = batch[0].to(device)
            encoded_batch, _ = autoencoder.encode(batch_tensor)
            encoded_batches.append(encoded_batch.cpu())

    encoded_dataset = torch.cat(encoded_batches, dim=0)
    encoded_dataset = reconstruct_dataset_dataframe(encoded_dataset, dataset)
    
    return encoded_dataset


def calculate_encoded_scalers(training_dataset, validation_dataset):
    training_dataset = training_dataset.drop(columns=METADATA + PHYSICAL_PARAMETERS)
    validation_dataset = validation_dataset.drop(columns=METADATA + PHYSICAL_PARAMETERS)
    
    training_min = training_dataset.min().min()
    training_max = training_dataset.max().max()
    
    validation_min = validation_dataset.min().min()
    validation_max = validation_dataset.max().max()
    
    encoded_min = min(training_min, validation_min)
    encoded_max = max(training_max, validation_max)
    
    return encoded_min, encoded_max




if __name__ == "__main__":
    if torch.cuda.is_available():        
        start_time = datetime.now()
        WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"
        
        scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))

        training_dataset, validation_dataset = load_datasets(WORKING_PATH)
        encoded_training = encode_dataset(training_dataset)
        encoded_validation = encode_dataset(validation_dataset)
        del training_dataset, validation_dataset

        encoded_min, encoded_max = calculate_encoded_scalers(encoded_training, encoded_validation)
        
        scalers["encoded_components"] = (
            encoded_min,
            encoded_max
        )
        print("Encoded Min: ", encoded_min)
        print("Encoded max: ", encoded_max)
        
        print(scalers)
        
        dump(scalers, os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
        print("Time Elapsed: ", datetime.now() - start_time)
    else:
        print("CUDA is not available.")