# PV Finder

To download this repository:

```bash
git clone git@gitlab.cern.ch:7999/LHCb-Reco-Dev/pv-finder.git
```

## Running

This is designed to run with Conda. You should prepare your environment:

```bash
conda env create -f environment.yml
```

(There is a GPU environment file there, too). Then whenever you want to work, run `conda activate pvfinder`. You can run `conda deactivate` to get out of the environment.

## Directories:

1. [`gen`](gen): Code to generate events using toy Velo model and produce files with hits and truth information.
2. [`ana`](ana): The code that generates the kernels, either from hits or from tracks. You can also convert hits to tracks using our proto tracking algorithm here.
3. [`model`](model): The ML model definitions
4. [`notebooks`](notebooks): Jupyter notebooks that run the ML models
5. [`scripts`](scripts): Scripts for data processing

Other directories:

* [`dat`](dat): A few tiny sample data files live here
* `binned_tracking`: Binned tracking toy code (classic code, CMake compile)
* [`tests`](tests): Minimal testing for now

## Requirements:

All requirements are now part of the `environment.yml` file, for all environments.

## Docker image (Developer info)

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

