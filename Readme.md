Validation Loss Definition

loss = abs(actual - predicted) / actual

This loss is calculated for each species and then the mean is calculated. This is calculated for the entire validation set.

Current mean loss for variational autoencoder: 5.2204e-01
Current STD of loss for variational autoencoder: 4.2515e-01

Current mean loss for 1 timestep with emulator: 2.4885e+03
Current std of loss for 1 timestep with emulator: 4.4960e+04

20 Worst performing species after encoding and decoding:
H3OPlus     1.332775 \n
N2          1.341790
CLPlus      1.345040
C3Plus      1.366286
SIPlus      1.410562
HCN         1.435771
CL          1.482872
HCL         1.518985
AtHE        1.543168
HC3N        1.595797
N2HPlus     1.603085
S2Plus      1.668589
HCSPlus     1.715503
H2COPlus    1.757007
HCLPlus     1.870920
SIC2Plus    1.876659
HCO2Plus    1.885456
CH2CO       2.151337
H3Plus      2.158025
C2H2Plus    2.215540

20 Wost performing species after 1 timestep emulation:
C3H5Plus        16.898203
H2CLPlus        17.529638
NumH2           18.812281
NH              20.007870
NumH2SIO        25.829245
H3Plus          29.090912
H2S2            50.711880
NumH2S2         60.682934
O2HPlus         66.897209
CH2Plus         68.538132
CHPlus          81.680840
H2SIO           84.870277
HOCSPlus        99.133972
H2O            237.100571
SIH3           249.515396
H3OPlus        291.983459
SIH4          2014.740967
OH            2166.838623
H2OPlus       3592.217773
H2          821690.625000