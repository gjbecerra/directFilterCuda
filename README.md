# Direct Filter implementation using CUDA

[![DOI](https://zenodo.org/badge/367249382.svg)](https://zenodo.org/badge/latestdoi/367249382)

This repository contains the source code referenced in the paper **Learning-based Current Estimation in Power Converters Operating in Continuous and Discontinuous Conduction Modes** presented in the ELECTRIMACS 2022 Conference and submitted for review to the *Mathematics and Computers in Simulation* journal.

The repository also contains sample datasets that can be used to test the algorithm. These files are in Matlab `.mat` format and contain the measured signals from the SEPIC power converter.

For compiling the code, type the following on a terminal:

```console
make all
```

This will build the `directFilter` executable in the current directory. For executing the algorithm run the following command:

```console
./directFilter data/mXX_NYYYYY.mat result.mat
```

where `XX` corresponds to the regressor length and `YYYYY` corresponds to the dataset size. The file `result.mat` will contain the Direct Filter estimate. This file can be loaded in Matlab for inspecting and printing the results.

The platform used to test this code is the Nvidia Jetson TX2.
