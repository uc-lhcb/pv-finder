# PV Finder



## Directories:

1. `gen`: Code to generate events using toy Velo model.
2. `ana`: The code that generates the kernels.
3. `model`: The ML model definitions
4. `notebooks`: Jupyter notebooks that run the ML models

Other directories:

* `dat`: A few sample data files live here
* `binned_tracking`: Binned tracking toy code (CMake compile)


## Requirements:

#### For the generation:

You should have a working ROOT-Python environement, with Pythia8 available from Python. See readme in `gen`.

#### For the kernel generation:

You should have a working ROOT environment.  See readme in `ana`.

#### For the machine learning:

This should be run in Jupyter on Anaconda using Python 3.6, with PyTorch available. Otherwise, most of the requirements are simple.

* tqdm : Nice progress bar

 See readme in `model` and/or `notebooks`.