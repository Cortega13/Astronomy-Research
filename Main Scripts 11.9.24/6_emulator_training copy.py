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
    "learning_rate": 0.001,
    "weight_decay": 0,
    "batch_size": 512,
    "shuffle": True,
    "early_stopping_tolerance": 12,
    "max_epochs": 999999,
    "gaussian_noise_std": 0.005,
    "layer_sizes": [20, 256, 256, 14],
    "max_clipping": 4,
}

TIMESTEP_DIFFS = 1
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
        
    training_dataset.sort_values(by=['Model', 'Time'], inplace=True)
    validation_dataset.sort_values(by=['Model', 'Time'], inplace=True)
    
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
    
    abundances_features_dataloader = DataLoader(TensorDataset(abundances_features_tensor), batch_size=emulator_HP["batch_size"], shuffle=False)

    return abundances_features_dataloader


def reconstruct_dataset_dataframe(encoded_dataset, dataset):
    encoded_dataset = encoded_dataset.cpu().detach().numpy()
    encoded_dataset = pd.DataFrame(encoded_dataset, columns=TOTAL_COMPONENTS)
    
    encoded_dataset = pd.concat([dataset[PHYSICAL_PARAMETERS], encoded_dataset], axis=1)
    encoded_dataset = pd.concat([dataset[METADATA], encoded_dataset], axis=1)
    
    return encoded_dataset


def create_emulator_dataset(df, timesteps=TIMESTEP_DIFFS):
    print("-=+=- Creating Emulator Dataset (input, output) -=+=-")
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
    gas_components_min, gas_components_max = scalers["gas_components"]
    bulk_components_min, bulk_components_max = scalers["bulk_components"]
    surface_components_min, surface_components_max = scalers["surface_components"]

    df[PHYSICAL_PARAMETERS] = np.log10(df[PHYSICAL_PARAMETERS])
    for parameter in PHYSICAL_PARAMETERS:
        df[parameter] = scalers[parameter].transform(df[parameter].values.reshape(-1, 1))

    df[GAS_COMPONENTS] = (df[GAS_COMPONENTS] - gas_components_min) / (gas_components_max - gas_components_min)
    df[BULK_COMPONENTS] = (df[BULK_COMPONENTS] - bulk_components_min) / (bulk_components_max - bulk_components_min)
    df[SURFACE_COMPONENTS] = (df[SURFACE_COMPONENTS] - surface_components_min) / (surface_components_max - surface_components_min)
    
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
    
    inputs, outputs = create_emulator_dataset(encoded_dataset)
    
    tensor_dataset = TensorDataset(inputs, outputs)
    return tensor_dataset


def tensor_to_dataloader(tensor_dataset, rank, world_size):
    sampler_train = DistributedSampler(
        tensor_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        drop_last=False,
        shuffle=emulator_HP["shuffle"]
    )
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=emulator_HP["batch_size"],
        pin_memory=True,
        shuffle=False,
        sampler=sampler_train,
        num_workers=4
    )
    return dataloader


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
        self.final_activation = nn.Sigmoid()

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


### Training Functions in Trainer Class
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        ampscaler: GradScaler,
        criterion: nn.Module,
        gpu_id: int,
        scalers: dict,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.ampscaler = ampscaler
        self.criterion = criterion
        self.minimum_loss = np.inf
        self.total_loss = 0
        self.epochs_without_improvement = 0
        self.scalers = scalers        

    def _save_checkpoint(self):
        print(f"Saving model with new minimum loss: {self.minimum_loss}.")
        checkpoint = self.model.module.state_dict()
        PATH = os.path.join(WORKING_PATH, f"Weights/emulator.pth")
        torch.save(checkpoint, PATH)


    def _check_early_stopping(self):
        if self.epochs_without_improvement >= emulator_HP["early_stopping_tolerance"]:
            print("Early Stopping.")
            return True
        return False
    

    def _check_minimum_loss(self):
        average_validation_loss = self.total_loss / len(self.validation_dataloader)
        if average_validation_loss < self.minimum_loss:
            self.minimum_loss = average_validation_loss
            self.epochs_without_improvement = 0
            if self.gpu_id == 0:
                self._save_checkpoint()
        else:
            self.epochs_without_improvement += 1
            print(f"Total stagnant epochs: {self.epochs_without_improvement}")
        self.total_loss = 0


    def _run_training_batch(self, features, targets):
        self.optimizer.zero_grad()  
        with autocast():
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
        
        self.ampscaler.scale(loss).backward()
        self.ampscaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), emulator_HP["max_clipping"])
        self.ampscaler.step(self.optimizer)
        self.ampscaler.update()


    def _run_validation_batch(self, features, targets):
        with torch.no_grad():
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            self.total_loss += loss.item()


    def _run_epoch(self, epoch):
        self.training_dataloader.sampler.set_epoch(epoch)
        self.model.train()
        for features, targets in self.training_dataloader:
            features = features.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)
            self._run_training_batch(features, targets)

        self.validation_dataloader.sampler.set_epoch(epoch)
        self.model.eval()
        for features, targets in self.validation_dataloader:
            features = features.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)
            self._run_validation_batch(features, targets)


    def train(self):
        for epoch in range(emulator_HP["max_epochs"]):
            self._run_epoch(epoch)
            self._check_minimum_loss()
            if self._check_early_stopping():
                break
        print(f"Training Complete. Trial Results: {self.minimum_loss}")


### Main Setup Functions
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if world_size > 1:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def load_training_objects(rank):
    model = Emulator(
        layer_sizes=emulator_HP["layer_sizes"],
        noise_std = emulator_HP["gaussian_noise_std"],
        ).to(rank)
    
    model = DDP(
        model, 
        device_ids=[rank], 
        output_device=rank
        )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=emulator_HP["learning_rate"], 
        weight_decay=emulator_HP["weight_decay"]
        )
    
    ampscaler = GradScaler()
    
    criterion = nn.L1Loss()

    return model, optimizer, ampscaler, criterion


def main(rank, world_size, training_tensor, validation_tensor):
    ddp_setup(rank, world_size)
    model, optimizer, ampscaler, criterion = load_training_objects(rank)
    training_dataloader = tensor_to_dataloader(training_tensor, rank, world_size)
    validation_dataloader = tensor_to_dataloader(validation_tensor, rank, world_size)
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
    trainer = Trainer(
        model,
        training_dataloader,
        validation_dataloader,
        optimizer,
        ampscaler,
        criterion,
        rank,
        scalers,
        )
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.is_available():
        start_time = datetime.now()
        WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"

        training_dataset, validation_dataset = load_datasets(WORKING_PATH)
        training_tensor = encode_dataset(training_dataset)
        validation_tensor = encode_dataset(validation_dataset)
        del training_dataset, validation_dataset

        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, training_tensor, validation_tensor), nprocs=world_size)

        print("Time Elapsed: ", datetime.now() - start_time)
    else:
        print("CUDA is not available.")