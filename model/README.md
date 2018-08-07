# Model files

The following files are used to run the ML models for finding PVs. Make sure this directory is in your `PYTHONPATH` (see the jupyter notebooks).

Before using this, set up `device` for either GPU or CPU. For example:

```python
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## collectdata

Usage:

```python
from collectdata import collect_data

data = '/data/schreihf/PvFinder/July_31_75000.npz'
dataset_train, dataset_val, _ = collect_data(
    data, 55_000, 10_000,
    device=device)
```

## models.py

This stores model functions. Use like this:

```python
from models import SimpleCNN2Layer as Model
```

If you want to support multiple GPUs, this can be added:

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

Finally, you need to move the model onto the device (if GPU, but no harm done if this is a CPU):

```python
model = model.to(device)
```

Current models include `SimpleCNN2Layer`, `SimpleCNN3Layer`


## loss.py

```python
from loss import Loss
```

Currently, you simply pass a loss object `Loss()` to the `trainNet` function described next.

## training.py

```python
from training import trainNet

for results in trainNet(model, dataset_train, dataset_val, Loss(), batch, range(n_epochs), learning_rate=1e-3):
    pass
```

This will train on the dataset. . The return value is a named tuple, with per epoch lists `.time`, `.cost`, and `.val`. The latter two are the training and validation loss per epoch, respectively. The original model will be updated, so you should do:

```python
torch.save(model.state_dict(), f'{name}_final.pyt')
```

to save the parameters inside or after the loop.

In the actual code, a nice progress bar is added as a wrapper around the `n_epoch` iterator.

## RunModel.py

This file runs models directly, without using jupyter notebooks. Use `./RunModel.py -h` to view the usage. You *must* set `CUDA_VISIBLE_DEVICES` to use this script!

The notebook form may be perferred, in `notebooks/RunModel.ipynb`.


## Loading a model

To perform inference, just load a model and the state dict associated with it:

```python
from models import SimpleCNN2Layer as Model
model = Model().to(device) # optional

model.load_state_dict(torch.load(filename))

# Prepare for inference
model.eval()
```

See the plethora of notebooks that do this: 

* `PlotEachPV`
* `FalsePositives`
* `PlotModel` (only partially load the model)