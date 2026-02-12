[[_TOC_]]
## Updates to pv-finder repo

This version of pv-finder is an attempt to merge pv-finder_v2 which vastly 
simplified and refactorized this repository with the original repository itself, 
which focused on CNN studies.

You can find pv-finder_v2 here: [https://gitlab.cern.ch/sakar/pv-finder_v2/-/tree/master?ref_type=heads](https://gitlab.cern.ch/sakar/pv-finder_v2/-/tree/master?ref_type=heads)

**Main improvements:**
- Clean the busy environment from pv-finder
- Streamline data preparation, model training, and model inference
    - Remove hard coded constants and fixed parameters that were previously defined inside different methods and group them into a single yaml configuration file 
    - Allow for a flexible usage of pv-finder between LHCb, ATLAS and CMS specific configurations using a single parameter `Experiment={EXP}` at the top of the configuration file
- Create even more complete and comprehensive documentation
    - User manual: Documentation for users to deploy pv-finder.
    - Developer manual: Documentation for developers to contribute on pv-finder
    - Inline comments: Include comments at the top of each file to give its 
    importance in the codebase. Include inline comments where possible for 
    better understanding. 
- Use latest version of underlying packages (awkward arrays, uproot, etc).
- Implement more sophisticated PyTorch techniques for improved performance and 
readability
- Integrate MLFlow and other machine learning tools for faster iteration, plot
generation, run tracking, and better development flow.
- "Re"-implement efficiency computation during training.

## Project status
- **pv-finder is currently being heavily refactorized and does not currently** 
**work**
- **Please report any issues to peters.michael@cern.ch**
- **Current version tested on: "sneezy"**
- **Last update: 5 February 2026**
- **To-do list**
    - **Documentation:**
        - [ ] File header comments
        - [ ] Developer manual
        - [ ] Inline comments
        - [ ] User manual
    - **Code:**
        - [ ] Simple, fully functioning pipeline from data -> final plots
            - [ ] Initial tracks ROOT files
            - [ ] Tracks-to-KDEs
            - [ ] KDEs-to-hists
            - [ ] Tracks-to-hists
            - [ ] Hists-to-PVs (or Tracks-to-PVs)
            - [ ] Plots
        - [ ] Advanced pytorch techniques: sophisticated data prepping, 
        data augmentation (opt.), robust datasets, updated model classes, 
        computation graphs, etc.
        - [ ] MLFlow integration
        - [ ] Model architecture studies
        - [ ] Pruning studies
        - [ ] Quantization studies: static, QAT

## Project installation

```bash
git clone git@github.com:uc-lhcb/pv-finder.git
```
## Project environement preparation

This step needs to be done once from a fresh git clone.

Start by going to the pv-finder base directory:

```bash
cd pv-finder
```
Then we need to create a conda environment based on the yaml file pv-finder_env.yml containing most of the necessary dependencies:

```bash
conda env create -f pv-finder_env.yml
```

Once done (typically several minutes), activate the environement:

```bash
conda activate pv-finder
```

PyTorch needs to be installed as well, using the suggested method from [https://pytorch.org/](https://pytorch.org/) (therefore not included in the yaml file)

Start by checing the installed CUDA version on the machine:

```bash
nvcc --version
```

```
which output should look like:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

Then run the appropriate install command (in the case using cuda_12):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Note that the PyTorch install might also take several minutes...

Some missing pieces (causing some conflicts with pytorch librairies if included in the yaml file, which I don't understand why) still need to be installed. 

To have the nice progress bars (widgets) in the jupyter notebooks: 

```bash
conda install -c conda-forge tqdm
conda install -n pv-finder -c conda-forge ipywidgets
```

**DO NOT USE! the following two command lines as they introduce conflicts...to be addressed!**
```bash
[DO NOT USE!] pip install -U jupyterlab-widgets==1.1.1
[DO NOT USE!] pip install -U ipywidgets==7.7.2
```

To be able to correctly set the kernel in the jupyter notebooks: 

```bash
pip install ipykernel
python -m ipykernel install --user --name pv-finder --display-name "pv-finder"
```

To get the latest uproot version (5): 

```bash
pip install uproot
```

In order to tell the environement about pv-finder dependencies, you can run the following command:

```bash
pip install -e .
```

## Activating project environement

The following commands needs to be done each time you log into the machine. It assumes you went through the package preparation step described above.

To run pv-finder: 

```bash
cd pv-finder
conda activate pv-finder
```

You can run `conda deactivate` to get out of the environment.

## Project directories

- `notebooks`: Jupyter notebooks that run the ML models
    - `notebooks/tutorials`: contains the tutorial notebooks 
- `configs`: Contains the yaml configuration files 
- `utils`: Contains the various methods used for NN models configuration, training, testing, plotting, efficiency evaluation...
- `scripts`: Scripts for data preprocessing
- `tools`: Contains the various methods used for data preprocessing
- `models`: Contains the NN models and loss definitions

## Configuration file

All configurable parameters (data preparation, NNs training, ...) are included in a unique file: `configs/configuration_default.yaml`.

The configurable parameter values are automatically propagated to all pv-finder scripts, from the data preprocessing scripts to the model training and inference.

The configuration file needs to be updated to set the files location (set in two places of the configuration file) to your user location: 
```
# ===================================================
data_processing_configs:
    [...]
    # path to the hdf5 (or tensor files) files created from the input root files
    # If using already processed files => no need to change this path
    # If creating new HDF5 files, this path should be changed to user home directory  
    output_files_path: /data/home/sakar/ML_dir/

# ===================================================
training_configs:
    [...]
    # path to the output of the models training (training weights ; training and testing plots)
    # This path should be changed to user home directory
    output_files_path: /data/home/sakar/ML_dir/

```

## Input data 

#### Input ROOT files

Original (LHCb) datasets are stored in ROOT format in `/share/lazy/pv-finder`. Currently four independent datasets are available:
- Minimum bias (magnet UP polarity): `pv_HLT1CPU_MinBiasMagUp`
- Minimum bias (magnet DOWN polarity): `pv_HLT1CPU_MinBiasMagDown`
- Bs -> Jpsi Phi  (magnet DOWN polarity): `pv_HLT1CPU_JpsiPhiMagDown`
- B+ -> D0 pi+  (magnet UP polarity): `pv_HLT1CPU_D0piMagUp`

Each ROOT file contains about 1000 events, and there are about 80 files for each case, resulting in about 80000 events for each of the four type of available data. 

A pre-processing of the original input data needs to be performed to:
- compute the variables used for training
- structure the data for the training

The input ROOT files contain: 
- track reconstructed POCA (Point of Closest Approach) parameter: 9 parameters
- POCA KDE (Kernel Density Estimator) histograms: 4000 bins in z
- Monte Carlo generator-level information on the primary vertexes: (x,y,z) position; number of originating tracks
- Monte Carlo generator-level information on the tracks: association to PV by unique key

#### HDF5 and Pytorch tensor files

The preprocessed files (see "Creation of HDF5 files from ROOT files") will be created in the `output_files_path` location defined in the configuration file.
Note that the full statistics of all four datasets have already been processed and are located in:

`/data/home/sakar/ML_dir/`

which has been made accessible for everyone on `sneezy`. 

The files name are:
- pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins.hdf5
- pv_HLT1CPU_MinBiasMagDown_AllFiles__validPVs_5tracks__Intervals_100bins.hdf5
- pv_HLT1CPU_JpsiPhiMagDown_AllFiles__validPVs_5tracks__Intervals_100bins.hdf5
- pv_HLT1CPU_D0piMagUp_AllFiles__validPVs_5tracks__Intervals_100bins.hdf5

and can be used directly for model training (see "Model training, inference and performances estimation")

## Data preprocessing

#### Creation of HDF5 files from ROOT files
Data preprocessing will create HDF5 files from the input ROOT files.
All the variables are stored in Awkward arrays, because of the irregular shapes of the data (varying number of tracks per event).

The preprocessing of the ROOT files consist of several steps:

1. **Event filtering**: Filter events with badly behaving tracks (nan, inf...) that will cause issues during the NN training. Typically this steps removes 1/1000 events
2. **Tracks sorting**: Tracks are sorted by increasing **z** values 
3. **Track-PV key update**: Key associating tracks and PVs are updated to be more easilly readable
4. **Valid PV definition**: PVs are declared valid given a minimum number of originating tracks (typically 5)
5. **Computing tracks' ellipsoid parameters**: Reduce the 9 POCA parameters to 6 POCA ellipsoid parameters for each track
6. **Target histograms creation**: For each event, a target histogram (4000 bins) is created from the list of valid PVs. Each PV is represented by a gaussian with the mean as the PV z position and width as a function of the number of originating tracks
7. **Intervals splitting**: Data is split in intervals of **z** (typically 100 intervals of 40 bins each). This is applied to the tracks, the KDE and target histograms. Note that when splitting the tracks in **z**, tracks with z values within sideband intervals (above and below a consdired interval) are also kept as input to a given interval. This way tracks falling near the boundary between two intervals will be used for both intervals.

To run the data preprocessing execute the following script:

```bash
cd pv-finder/
./scripts/Make_HDF5_files_from_ROOT_files.py -c configuration_default.yaml -o pv_HLT1CPU_MinBiasMagUp_1 -f pv_HLT1CPU_MinBiasMagUp_1.root
```

where `-c configuration_default.yaml` identifies the configuration file to be used, `-o pv_HLT1CPU_MinBiasMagUp_1` tags the output file name and `-f pv_HLT1CPU_MinBiasMagUp_1.root` allows to identify which input ROOT files to used. 

Note that the `.hdf5` extension will automatically be added to `pv_HLT1CPU_MinBiasMagUp_1`, so one should not include the extension in the command line. 

In this example, the output produced output HDF5 file will be:

```bash
pv_HLT1CPU_MinBiasMagUp_1__validPVs_5tracks__Intervals_100bins.hdf5
```

where `_validPVs_5tracks__Intervals_100bins` are tags automatically added to the output file name. In details `validPVs_5tracks` state that only PVs with at least 5 originating tracks are considered valid, and `Intervals_100bins` state that each event in the data is split in 40 intervals of 100 bins each (4000 total bins per event).

The output location of the HDF5 files is also defined in the `configuration_default.yaml` file:

```bash
[...]
data_processing_configs:
    [...]
    output_files_path: /data/home/sakar/ML_dir/
```
which needs to be changed to a dedicated `$USER` directory to create new HDF5 files.

Also it is possible to run over multiple input ROOT files. For instance:

```bash
./scripts/Make_HDF5_files_from_ROOT_files.py -c configuration_default.yaml -o pv_HLT1CPU_MinBias_ConfigA -f pv_HLT1CPU_MinBiasMagUp_1.root pv_HLT1CPU_MinBiasMagUp_2.root pv_HLT1CPU_MinBiasMagDown_3.root
```

will create the output file `pv_HLT1CPU_MinBias_ConfigA.hdf5` from the three input ROOT files.

It is also possible to parse many input ROOT files using the asterisk as:

```bash
./scripts/Make_HDF5_files_from_ROOT_files.py -c configuration_default.yaml -o pv_HLT1CPU_MinBiasMagUp_AllFiles -f pv_HLT1CPU_MinBiasMagUp_*.root
```

which will run over all `pv_HLT1CPU_MinBiasMagUp` ROOT files.

**Running over ~80 ROOT files takes about 15-20 minutes. The resulting output file may be as large as ~50 GB.**

#### Creation of Pytorch tensors from HDF5 file

To perform the training and inference of the models `DataLoader` objects needs to be created from the data.
Loading the data from the HDF5 file into `DataLoader` objects can take several minutes depending on the number of events you consider. 
In fact the data first needs to be transformed into Pytorch tensors, which is the step that takes longer.

To speed up the data loading when performing model training, it is possible to create the Pytorch tensors in advance from the HDF5 file.
These Pytorch tensors will then be used to create the `DataLoader` objects that will be used in the model training. The downside of this approach is that it duplicates the stored info on disk (typically tens of GB), which can be a limiting factor depending on the available storage ressources.

To build the Pytorch tensors from an existing HDF5 file, execute the following command:

```bash
./scripts/Build_Torch_Tensors.py -c configuration_default.yaml -i pv_HLT1CPU_MinBiasMagUp_AllFiles
```

where `-c configuration_default.yaml` identifies the configuration file to be used and `-i pv_HLT1CPU_MinBiasMagUp_AllFiles` tags the input HDF5 file name. Note that as for the HDF5 file creation (see "Creation of the HDF5 files from ROOT files" above) the `.hdf5` extension will automatically be added to `outputfileName`, so one should not include the extension in the command line.


Several output files will be created in the same location as the input HDF5 file. Using the default `configuration_default.yaml` file parameters,  the following files will be created in `/data/home/sakar/ML_dir/`:

- `pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins_tracksData_inter.pt`
- `pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins_poca_KDE_A_inter.pt`
- `pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins_poca_KDE_B_inter.pt`
- `pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins_targetHists_inter.pt`
- `pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins_poca_KDE_A_xMax_inter.pt `
- `pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins_poca_KDE_A_yMax_inter.pt `
- `pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins_poca_KDE_B_xMax_inter.pt `
- `pv_HLT1CPU_MinBiasMagUp_AllFiles__validPVs_5tracks__Intervals_100bins_poca_KDE_B_yMax_inter.pt `

where the suffix of the files indicates which variables are stored in the pytorch tensor:

- `tracksData_inter`: tracks parameters: {x,y,z} & {6 ellipsoid params}
- `poca_KDE_A`: reconstructed KDE in z from the SUM over tracks POCA params (split in 100 intervals)
- `poca_KDE_B`: reconstructed KDE in z from the SUM over "squared" tracks POCA params (split in 100 intervals)
- `targetHists_inter`: target histogram from PV truth information
- `poca_KDE_A_xMax`: maximum of X for which poca_KDE_A is maximum in Z (split in 100 intervals)
- `poca_KDE_A_yMax`: maximum of Y for which poca_KDE_A is maximum in Z (split in 100 intervals)
- `poca_KDE_B_xMax`: maximum of X for which poca_KDE_B is maximum in Z (split in 100 intervals)
- `poca_KDE_B_yMax`: maximum of Y for which poca_KDE_B is maximum in Z (split in 100 intervals)

Depending on the NN model configuration, these pytorch tensors can be used directly to construct the `DataLoader` objects used for training and validation.

## Data Manipulation

The training, validation and eventually testing datasets can be constructed either using the HDF5 files or the pytorch tensors as input.
The following tutorial (in `notebooks/tutorials/`) presents the different available methods to construct these datasets:

- `Tutorial_Data_Manipulation.ipynb`

Detailled documentation is available inside the tutorial. 

## Model training, inference and performances estimation

The following tutorials (in `notebooks/tutorials/`) introduce procedure to train and test the two classes of NN models:

- `Tutorial_training_tracks_to_KDE.ipynb`: using the `tracks-to-KDE` model
- `Tutorial_training_tracks_to_hist.ipynb`: using the `tracks-to-hist` model

Detailled documentation is available inside the tutorials. 
