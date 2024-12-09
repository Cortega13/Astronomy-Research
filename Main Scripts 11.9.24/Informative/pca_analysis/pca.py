from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import gc
from sklearn.decomposition import PCA
from joblib import load


model_timesteps = 90
dataset_size = 108115 * model_timesteps
#dataset_size = 4
dataset_divisions = 1

start1 = datetime.now()

dataset_constant_column_names = ['Time', 'Model', 'point', 'gasTemp', 'zeta', 'av', 'Radfield', 'Density']
num_input_features = 343 - len(dataset_constant_column_names)
scalers = load("/home1/09338/carlos9/uclchem_emulator/abundances.scalers")

def load_data(partition):
    print("Dataset loading.")
    start_ind = int((partition / dataset_divisions)*dataset_size)
    end_ind = int(((partition+1) / dataset_divisions)*dataset_size)
    if end_ind > dataset_size:
        end_ind = dataset_size
    partial_dataset = pd.read_hdf('/work2/09338/carlos9/frontera/uclchem_data.h5', 'uclchem_models', start=start_ind, stop=end_ind).astype(np.float32)
    dataset = partial_dataset.drop(columns=dataset_constant_column_names)
    dataset_scaled = np.log10(dataset)
    for column in dataset_scaled.columns:
        dataset_scaled[column] = scalers[column].transform(dataset_scaled[column].values.reshape(-1, 1)).reshape(-1)

    gc.collect()

    return dataset_scaled

dataset_scaled = load_data(0)


end1 = datetime.now()
print(f"Time taken to import data: {end1 - start1}")

start2 = datetime.now()

pca = PCA(n_components=15)
data_reduced = pca.fit_transform(dataset_scaled)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

end2 = datetime.now()

print(f"Time taken to run PCA: {end2 - start2}")
print("Shape of the reduced dataset:", data_reduced.shape)
print("Cumulative Explained variance ratio of the components:", cumulative_explained_variance)
print(f"Total variance explained: {pca.explained_variance_ratio_.sum()}")

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='blue', marker='.', linestyle='-', linewidth=1, markersize=3, label='Individual Explained Variance')
plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, color='red', marker='.', linestyle='-', linewidth=1, markersize=3, label='Cumulative Explained Variance')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("/home1/09338/carlos9/uclchem_emulator/pca_analysis/pca_analysis15.png", dpi=300)
print("Image Saved.")

# [0.6235075  0.7852024  0.83862469 0.87330458 0.90348652 0.91748042
#  0.92626792 0.93456668 0.94218808 0.94750781 0.95222075 0.9566758
#  0.96012421 0.96321638 0.96568548 0.96810104 0.97028733 0.97228095
#  0.97393327 0.97552231 0.9769285  0.97827002 0.97958407 0.98071084
#  0.98171514 0.98264696 0.98348198 0.98425418 0.98500637 0.98570687
#  0.98636857 0.98694262 0.98748589 0.98797195 0.98842645 0.98886617
#  0.98928081 0.98968827 0.99005022 0.99040717 0.99075262 0.99108664
#  0.99140456 0.99170228 0.99199141 0.99226216 0.99251618 0.99275656
#  0.9929802  0.99319196 0.99340168 0.99360197 0.9937956  0.99397979
#  0.99415794 0.99433393 0.99450199 0.99465585 0.9948049  0.99494912
#  0.99508671 0.99522027 0.99534849 0.99546611 0.99558011 0.99568831
#  0.9957908  0.99589104 0.99598907 0.99608615 0.99617909 0.99627063
#  0.99636029 0.99644322 0.99652522 0.99660399 0.99668014 0.99675413
#  0.99682608 0.9968965  0.99696491 0.99703007 0.9970931  0.99715509
#  0.99721576 0.9972734  0.99733078 0.99738686 0.99744059 0.99749369
#  0.99754509 0.99759455 0.99764286 0.99768888 0.99773402 0.99777819
#  0.99782108 0.99786206 0.99790211 0.99794193]

#95.22% for n=11 components