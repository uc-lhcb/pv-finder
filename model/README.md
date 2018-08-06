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

results = trainNet(model, dataset_train, dataset_val, Loss(), 32, n_epochs, learning_rate=1e-3)
```

This will train on the dataset. Pass `name=name` to save a model file each epoch. The return value is a named tuple, with `.time`, `.cost`, and `.val`. The latter two are arrays with the training and validation loss per epoch, respectively. The original model will be updated, so you should do:

```python
torch.save(model.state_dict(), f'{name}_final.pyt')
```

to save the final model parameters (at least if you didn't set a name).

## runmodel.py

This file runs models directly, without using jupyter notebooks.



## Loading a model

To perform inference, just load a model and the state dict associated with it:

```python
from models import SimpleCNN2Layer as Model
model = Model().to(device) # optional

model.load_state_dict(torch.load(filename))

# Prepare for inference
model.eval()
```
