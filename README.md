# PV Finder

To download this repository:

```bash
git clone git@gitlab.cern.ch:7999/LHCb-Reco-Dev/pv-finder.git
```

## Directories:

1. `gen`: Code to generate events using toy Velo model
2. `ana`: The code that generates the kernels
3. `model`: The ML model definitions
4. `notebooks`: Jupyter notebooks that run the ML models
5. `scripts`: Scripts for data processing

Other directories:

* `dat`: A few tiny sample data files live here
* `binned_tracking`: Binned tracking toy code (CMake compile)
* `tests`: Minimal testing for now

## Requirements:

#### For the generation:

You should have a working ROOT-Python environment, with Pythia8 available from Python. See readme in `gen`.

#### For the kernel generation:

You should have a working ROOT environment.  See readme in `ana`.

#### For the machine learning:

This should be run in Jupyter on Anaconda using Python 3.6, with PyTorch available. Otherwise, most of the requirements are simple. Look at the file Pipfile to see what the requirements are; the requirements beyond the base anaconda distribution are:

* pytorch : Machine learning package
* tqdm : Nice progress bar
* uproot : ROOT file and awkward array support
* plumbum : Used in scripts

 See readme in [`model`](./model) and/or [`notebooks`](./notebooks).

## Docker image

A CPU docker image is provided, mostly for the CI. To use:

```bash
docker run --rm -it gitlab-registry.cern.ch/lhcb-reco-dev/pv-finder:latest
```

To build it:

```bash
docker build -t gitlab-registry.cern.ch/lhcb-reco-dev/pv-finder .
```

To push it:

```bash
docker login gitlab-registry.cern.ch
docker push gitlab-registry.cern.ch/lhcb-reco-dev/pv-finder
```

