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
* [`binned_tracking`](binned_tracking): Binned tracking toy code (classic code, CMake compile)
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

## Run PV Finder in the LHCb software stack ("the trigger")

Setup your own lhcb stack (in this way we can ensure that PVFinder can be compiled against LHCb software) https://gitlab.cern.ch/rmatev/lb-stack-setup. The stack setup works best on lxplus.

The lb-stack-setup README contains many useful informations for further development, here we only need
```sh
curl https://gitlab.cern.ch/rmatev/lb-stack-setup/raw/master/setup.py | python3 - stack
```
then open the config for editing
```sh
$EDITOR utils/config.json
```
and add the following:
```json
    "binaryTag": "x86_64-centos7-clang10-opt",
    "defaultProjects": [
      "Moore"
    ],
    "gitBranch": {
      "Detector": "v0-patches",
      "default": "CNNVertexFinder"
    },
    "lcgVersion": "97a",
    "gitGroup": {
      "Detector": "lhcb",
      "default": "mstahl"
    },
    "dataPackages": [
    ],
    "lbenvPath": "/cvmfs/lhcb.cern.ch/lib/var/lib/LbEnv/1110/stable/linux-64",
    "cmakeFlags": {
        "Moore": "-DLOKI_BUILD_FUNCTOR_CACHE=OFF"
    }
```
After that, you can type `make` and let the stack compile (takes about 1 to 2 hours if eveything works fine).
With that configuration, we make sure that a snapshot of the entire stack (taken from private forks under `mstahl`) is used, which we know runs. We have also freezed lcg and lbenv versions, as well as the binary tag to be able to work with torch-script.

After compiling, you should be able to run commands like 
```sh
Moore/run gaudirun.py Moore/Hlt/Moore/tests/options/default_input_and_conds_hlt1_FTv6.py Moore/Hlt/RecoConf/options/hlt1_reco_pvchecker.py 2>&1 | tee CNNVertexFinder.log
```
