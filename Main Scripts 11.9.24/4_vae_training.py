"""

INPUT: training.h5, validation.h5, abundances.scalers
OUTPUT: autoencoder.pth
"""

import os
import optuna
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import load
import torch
from torch import nn, optim, multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

### Configurations
def load_configurations(WORKING_PATH):
    HP = {
        "encoded_dimensions": 11,
        "learning_rate": 0.001,
        "weight_decay": 0,
        "batch_size": 2*4096,
        "shuffle": True,
        "early_stopping_tolerance": 12,
        "max_epochs": 999999,
        "hidden_layer": 200,
        "max_clipping": 6,
        "dropout_rate": 0.005,
        "exponential_coefficient": 18,
    }
    METADATA = ["Time", "Model"] 
    PHYSICAL_PARAMETERS = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/physical_parameters.txt"), dtype=str, delimiter=" ").tolist()
    TOTAL_SPECIES = np.loadtxt(os.path.join(WORKING_PATH, "Main Scripts 11.9.24/utils/species.txt"), dtype=str, delimiter=" ").tolist()

    return HP, METADATA, PHYSICAL_PARAMETERS, TOTAL_SPECIES

WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/"
HP, METADATA, PHYSICAL_PARAMETERS, TOTAL_SPECIES = load_configurations(WORKING_PATH)

### Data Processing Functions
def load_datasets(path):
    training_dataset_path = os.path.join(path, "Datasets/training.h5")
    validation_dataset_path = os.path.join(path, "Datasets/validation.h5")

    training_dataset = pd.read_hdf(training_dataset_path, "autoencoder", start=0).astype(np.float32)
    validation_dataset = pd.read_hdf(validation_dataset_path, "autoencoder", start=0).astype(np.float32)

    training_dataset = training_dataset[TOTAL_SPECIES]
    validation_dataset = validation_dataset[TOTAL_SPECIES]

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


def autoencoder_postprocessing(encoded_features):
    #print("Starting Decoder Postprocessing.")
    scalers = load(os.path.join(WORKING_PATH, "Datasets/scalers.plk"))
    abundances_min, abundances_max = scalers["total_species"]

    # Moving decoder_output to cpu and convert to numpy.
    encoded_features = encoded_features.cpu().detach().numpy()

    # Convert the decoded_abundances to pandas dataframe with column names.
    encoded_features = pd.DataFrame(encoded_features, columns=TOTAL_SPECIES)

    unscaled_features = encoded_features * (abundances_max - abundances_min) + abundances_min

    decoded_abundances = pd.DataFrame(10**unscaled_features, columns=TOTAL_SPECIES)

    return decoded_abundances


### Defining the Variational Autoencoder.
class VariationalAutoencoder(nn.Module):
    def __init__(self, num_features, encoded_dimensions=1, hidden_layer=1, dropout_rate=0):
        super(VariationalAutoencoder, self).__init__()
                
        # Encoder
        self.layer1 = nn.Linear(num_features, hidden_layer)
        self.batch_norm1 = nn.BatchNorm1d(hidden_layer)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc_mu = nn.Linear(hidden_layer, encoded_dimensions)
        self.fc_logvar = nn.Linear(hidden_layer, encoded_dimensions)
        
        # Decoder
        self.layer3 = nn.Linear(encoded_dimensions, hidden_layer)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.layer4 = nn.Linear(hidden_layer, num_features)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
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
        z1 = self.dropout2(z1)
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



### Defining the VAE Loss Function
def vae_loss(reconstructed_x, x, mu, logvar, beta=3):
    elementwise_loss = torch.abs(reconstructed_x - x)*((x+3e-2)**0.2)
    elementwise_loss = torch.pow(10, HP["exponential_coefficient"]*elementwise_loss)
    recon_loss = torch.sum(elementwise_loss)
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    total_loss *= 1e-6
    
    print(f"Recon Loss: {recon_loss:.4e} | KL Loss: {beta*kl_loss:.4e} | Total Loss: {total_loss:.4e}")
    return total_loss


### Training Functions in Trainer Class
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        ampscaler: GradScaler,
        criterion_train: callable,
        criterion_val: nn.Module,
        gpu_id: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.ampscaler = ampscaler
        self.criterion_train = criterion_train
        self.criterion_val = criterion_val
        self.minimum_loss = np.inf
        self.total_loss = 0
        self.epochs_without_improvement = 0
        

    def _save_checkpoint(self):
        checkpoint = self.model.module.state_dict()
        PATH = os.path.join(WORKING_PATH, "Weights/vae.pth")
        torch.save(checkpoint, PATH)


    def _check_early_stopping(self):
        if self.epochs_without_improvement >= HP["early_stopping_tolerance"]:
            print("Early Stopping.")
            return True
        return False
    

    def _check_minimum_loss(self):
        average_validation_loss = self.total_loss / len(self.validation_dataloader)
        if average_validation_loss < self.minimum_loss:
            if self.gpu_id == 0:
                print(f"New Minimum loss: {average_validation_loss:.4e}. Percent Improvement: {(average_validation_loss*100/self.minimum_loss):.3f}%")
                self._save_checkpoint()
            self.minimum_loss = average_validation_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            print(f"Total stagnant epochs: {self.epochs_without_improvement}")
        self.total_loss = 0


    def _run_training_batch(self, features):
        self.optimizer.zero_grad()  
        with autocast():
            reconstructed_x, mu, logvar = self.model(features, is_inference=False)
            loss = self.criterion_train(reconstructed_x, features, mu, logvar)
        
        self.ampscaler.scale(loss).backward()
        self.ampscaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), HP["max_clipping"])
        self.ampscaler.step(self.optimizer)
        self.ampscaler.update()


    def _run_validation_batch(self, features):
        with torch.no_grad():
            reconstructed_x = self.model(features, is_inference=True)

            unscaled_features = autoencoder_postprocessing(features)
            unscaled_predictions = autoencoder_postprocessing(reconstructed_x)

            unscaled_features_np = unscaled_features.to_numpy()
            unscaled_predictions_np = unscaled_predictions.to_numpy()

            unscaled_features_tensor = torch.tensor(unscaled_features_np, dtype=torch.float32)
            unscaled_predictions_tensor = torch.tensor(unscaled_predictions_np, dtype=torch.float32)
            
            #self.test = (torch.abs(unscaled_predictions_tensor - unscaled_features_tensor) / unscaled_features_tensor).mean()
            self.total_loss = (torch.abs(unscaled_features_tensor - unscaled_predictions_tensor) / unscaled_features_tensor).sum()
            # loss = self.criterion_val(unscaled_predictions_tensor, unscaled_features_tensor)
            # self.total_loss += loss.item()


    def _run_epoch(self, epoch):
        self.training_dataloader.sampler.set_epoch(epoch)
        self.model.train()
        for features in self.training_dataloader:
            features = features.to(self.gpu_id, non_blocking=True)
            self._run_training_batch(features)

        self.validation_dataloader.sampler.set_epoch(epoch)
        self.model.eval()
        for features in self.validation_dataloader:
            features = features.to(self.gpu_id, non_blocking=True)
            self._run_validation_batch(features)


    def train(self):
        for epoch in range(HP["max_epochs"]):
            self._run_epoch(epoch)
            self._check_minimum_loss()
            if self._check_early_stopping():
                break
        print(f"Training Complete. Trial Results: {self.minimum_loss:.4e}")


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
    model = VariationalAutoencoder(
        encoded_dimensions=HP["encoded_dimensions"],
        num_features=len(TOTAL_SPECIES),
        hidden_layer=HP["hidden_layer"],
        dropout_rate=HP["dropout_rate"]
        ).to(rank)
    
    model = DDP(
        model, 
        device_ids=[rank], 
        output_device=rank,
        )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=HP["learning_rate"], 
        weight_decay=HP["weight_decay"]
        )
    
    ampscaler = GradScaler()
    
    criterion_train = vae_loss
    criterion_val = nn.L1Loss()

    return model, optimizer, ampscaler, criterion_train, criterion_val


def prepare_dataloaders(training_tensor, validation_tensor, world_size, rank):
    sampler_train = DistributedSampler(training_tensor, num_replicas=world_size, rank=rank, drop_last=False)
    training_dataloader = DataLoader(
        training_tensor,
        batch_size=HP["batch_size"],
        pin_memory=True,
        shuffle=False,
        sampler=sampler_train,
        num_workers=4
    )

    sampler_val = DistributedSampler(validation_tensor, num_replicas=world_size, rank=rank, drop_last=False)
    validation_dataloader = DataLoader(
        validation_tensor,
        batch_size=HP["batch_size"],
        pin_memory=True,
        shuffle=False,
        sampler=sampler_val,
        num_workers=4
    )

    return training_dataloader, validation_dataloader


def main(rank, world_size, training_tensor, validation_tensor):
    ddp_setup(rank, world_size)
    model, optimizer, ampscaler, criterion_train, criterion_val = load_training_objects(rank)
    training_dataloader, validation_dataloader = prepare_dataloaders(training_tensor, validation_tensor, world_size, rank)
    trainer = Trainer(
        model,
        training_dataloader,
        validation_dataloader,
        optimizer,
        ampscaler,
        criterion_train,
        criterion_val,
        rank
        )
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.is_available():
        start_time = datetime.now()
        training_dataset, validation_dataset = load_datasets(WORKING_PATH)

        training_tensor = autoencoder_preprocessing(training_dataset)
        validation_tensor = autoencoder_preprocessing(validation_dataset)

        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, training_tensor, validation_tensor), nprocs=world_size)

        print("Time Elapsed: ", datetime.now() - start_time)
    else:
        print("CUDA is not available.")


#5.2406e+02

# 8.8672e+02

# 7.5997e+02