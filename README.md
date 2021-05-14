# Direct Filter implementation using CUDA

This repository contains the source code referenced in the paper **Current Estimation in Power Converters Operating in Continuous and Discontinuous Conduction Mode** submitted for review on the *Control Engineering Practice* journal.

The repository also contains sample datasets that can be used to test the algorithm. These files are in Matlab `.mat` format and contain the measured signals from the SEPIC power converter.

For compiling the code, type the following on a terminal:

``
$ make all
``

This will build the `directFilter` executable in the current directory. For executing the algorithm run the following command:

``
$ ./directFilter data/mXX_NYYYYY.mat result.mat
``

where `XX` corresponds to the regressor length and `YYYYY` corresponds to the dataset size. The file `result.mat` will contain the Direct Filter estimate. This file can be loaded in Matlab for inspecting and printing the results.

The platform used to test this code is the Nvidia Jetson TX2.