Validation Loss Definition

loss = abs(actual - predicted) / actual

This loss is calculated for each species and then the mean is calculated. This is calculated for the entire validation set. The species abundances range from 1e-20 to 1. 

Current mean loss for variational autoencoder: 5.2250e-01
Current STD of loss for variational autoencoder: 4.5079e-01

Current mean loss for 1 timestep with FFN emulator: 1.6274e+03
Current std of loss for 1 timestep with FFN emulator: 2.8587e+04

Current mean loss for 10 timesteps with FFN emulator: 1.4071e+06
Current std of loss for 10 timesteps with FFN emulator: 1.9440e+07

Error of 6 Worst performing species after emulating 1 timestep:

NH             522.993042
H2O            701.716980
SIH4          1822.667236
OH            2060.281494
H2OPlus      12319.539062
H2          563971.562500


With 100% of the total dataset
1 Timestep
Mean: 1.6611e+02
STD: 2.8037e+03

10 Timesteps
Mean: 7.3099e+05
STD: 9.2439e+06


With 90% of the total dataset

1 Timestep
Mean: 2.2294e+02
STD: 3.8278e+03

10 Timesteps
Mean: 8.7011e+05
STD: 1.0980e+07


With 80% of the total dataset

1 Timestep
Mean: 1.8111e+02
STD: 3.0644e+03

10 Timesteps
Mean: 1.6373e+06
STD: 2.1271e+07


With 70% of the total dataset

1 Timestep
Mean: 2.0768e+02
STD: 3.5570e+03

10 Timesteps
Mean: 9.0304e+05
STD: 1.1814e+07


With 60% of the total dataset

1 Timestep
Mean: 2.7134e+02
STD: 4.7127e+03

10 Timesteps
Mean: 3.0176e+06
STD: 3.8885e+07


With 40% of the total dataset

1 Timestep
Mean: 2.1285e+02
STD: 3.5937e+03

10 Timesteps
Mean: 1.6386e+06
STD: 2.2049e+07