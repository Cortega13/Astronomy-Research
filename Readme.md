Validation Loss Definition

loss = abs(actual - predicted) / actual

This loss is calculated for each species and then the mean is calculated. This is calculated for the entire validation set. The species abundances range from 1e-20 to 1. 

Current mean loss for variational autoencoder: 5.2250e-01
Current STD of loss for variational autoencoder: 4.5079e-01

Current mean loss for 1 timestep with FFN emulator: 1.6274e+03
Current std of loss for 1 timestep with FFN emulator: 2.8587e+04

Current mean loss for 10 timesteps with FFN emulator: 1.4071e+06
Current std of loss for 10 timesteps with FFN emulator: 1.9440e+07

Current mean loss for 1 timestep with Mace ODEs: 3.7718e+03
Current std of loss for 1 timestep with Mace ODEs: 6.7567e+04

Error of 6 Worst performing species after emulating 1 timestep:

NH             522.993042
H2O            701.716980
SIH4          1822.667236
OH            2060.281494
H2OPlus      12319.539062
H2          563971.562500
