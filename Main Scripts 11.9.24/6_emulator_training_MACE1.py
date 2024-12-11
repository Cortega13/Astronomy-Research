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
HP = {
    "encoded_dimensions": 11,
    "learning_rate": 0.001,
    "weight_decay": 0,
    "batch_size": 512,
    "shuffle": True,
    "early_stopping_tolerance": 12,
    "max_epochs": 999999,
    "hidden_layer": 200,
    "max_clipping": 4,
}

METADATA = ["Time", "Model"] 
PHYSICAL_PARAMETERS = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/physical_parameters.txt"), dtype=str, delimiter=" ").tolist()
TOTAL_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/species.txt"), dtype=str, delimiter=" ").tolist()
COMPONENTS = [f"Component_{i}" for i in range(1, HP["encoded_dimensions"]+1)]

### Data Processing Functions
def load_datasets(path):
    training_dataset_path = os.path.join(path, "Datasets/training.h5")
    validation_dataset_path = os.path.join(path, "Datasets/validation.h5")

    training_dataset = pd.read_hdf(training_dataset_path, "emulator", start=0).astype(np.float32).reset_index(drop=True)
    validation_dataset = pd.read_hdf(validation_dataset_path, "emulator", start=0).astype(np.float32).reset_index(drop=True)
    training_dataset.sort_values(by=['Model', 'Time'], inplace=True)
    validation_dataset.sort_values(by=['Model', 'Time'], inplace=True)
    
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
    encoded_dataset = encoded_dataset.cpu().detach().numpy()
    encoded_dataset = pd.DataFrame(encoded_dataset, columns=COMPONENTS)
    
    encoded_dataset = pd.concat([dataset[PHYSICAL_PARAMETERS], encoded_dataset], axis=1)
    encoded_dataset = pd.concat([dataset[METADATA], encoded_dataset], axis=1)
    
    return encoded_dataset


def create_emulator_dataset(df, timesteps=1):
    print("-=+=- Creating Emulator Dataset (input, output) -=+=-")
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
    encoded_min, encoded_max = scalers["encoded_components"]

    df[PHYSICAL_PARAMETERS] = np.log10(df[PHYSICAL_PARAMETERS])
    for parameter in PHYSICAL_PARAMETERS:
        df[parameter] = scalers[parameter].transform(df[parameter].values.reshape(-1, 1))

    df[COMPONENTS] = (df[COMPONENTS] - encoded_min) / (encoded_max - encoded_min)
    
    inputs, outputs = [], []
    
    for _, sub_df in df.groupby('Model'):
        sub_array = sub_df[PHYSICAL_PARAMETERS + COMPONENTS].to_numpy()
        
        num_rows = len(sub_array)
        if num_rows > timesteps:
            input_window = sub_array[:-timesteps]
            output_window = sub_array[timesteps:, :]
            if timesteps == 0:
                input_window = sub_array
                output_window = sub_array[:, :]
            
            inputs.append(input_window)
            outputs.append(output_window)

    inputs = torch.tensor(np.vstack(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.vstack(outputs), dtype=torch.float32)
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
    
    inputs, outputs = create_emulator_dataset(encoded_dataset)
    
    tensor_dataset = TensorDataset(inputs, outputs)
    return tensor_dataset


def tensor_to_dataloader(tensor_dataset, rank, world_size):
    sampler_train = DistributedSampler(
        tensor_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        drop_last=False,
        shuffle=HP["shuffle"]
    )
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=HP["batch_size"],
        pin_memory=True,
        shuffle=False,
        sampler=sampler_train,
        num_workers=4
    )
    return dataloader


class G(nn.Module):
    def __init__(self, z_dim=11):
        super(G, self).__init__()
        self.C = nn.Parameter(torch.randn(z_dim).requires_grad_(True))
        self.A = nn.Parameter(torch.randn(z_dim, z_dim).requires_grad_(True))
        self.B = nn.Parameter(torch.randn(z_dim, z_dim, z_dim).requires_grad_(True))

    def forward(self, z):
        return self.C + torch.einsum("ij, bj -> bi", self.A, z) + torch.einsum("ijk, bj, bk -> bi", self.B, z, z)

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
        encoded_min: float,
        encoded_max: float
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
        self.encoded_min = encoded_min
        self.encoded_max = encoded_max
        

    def _save_checkpoint(self):
        print(f"Saving model with new minimum loss: {self.minimum_loss}.")
        checkpoint = self.model.module.state_dict()
        PATH = os.path.join(WORKING_PATH, "Weights/Gemulator.pth")
        torch.save(checkpoint, PATH)


    def _check_early_stopping(self):
        if self.epochs_without_improvement >= HP["early_stopping_tolerance"]:
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), HP["max_clipping"])
        self.ampscaler.step(self.optimizer)
        self.ampscaler.update()


    def _run_validation_batch(self, features, targets):
        with torch.no_grad():
            outputs = self.model(features)
            unscaled_outputs = outputs * (self.encoded_max - self.encoded_min) + self.encoded_min
            unscaled_targets = targets * (self.encoded_max - self.encoded_min) + self.encoded_min
            loss = self.criterion(unscaled_outputs, unscaled_targets)
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
        for epoch in range(HP["max_epochs"]):
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
    model = G(
        z_dim=len(COMPONENTS)+len(PHYSICAL_PARAMETERS)
        ).to(rank)
    
    model = DDP(
        model, 
        device_ids=[rank], 
        output_device=rank
        )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=HP["learning_rate"], 
        weight_decay=HP["weight_decay"]
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
    encoded_min, encoded_max = scalers["encoded_components"]
    trainer = Trainer(
        model,
        training_dataloader,
        validation_dataloader,
        optimizer,
        ampscaler,
        criterion,
        rank,
        encoded_min,
        encoded_max
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
        

#  2x256
