import time
import torch
from collections import namedtuple
import sys
import os

from .utilities import tqdm_redirect, import_progress_bar, get_device_from_model
from .efficiency import efficiency, ValueSet

Results = namedtuple("Results", ["epoch", "cost", "val", "time", "eff_val"])

## change difference from 5.0 to 7.5 to see what change in eff-v-FP this produces
PARAM_EFF = {
    "difference": 7.5,   ## was 5.0 in training.py
    "threshold": 1e-2,
    "integral_threshold": 0.2,
    "min_width": 3,
}


def select_gpu(selection=None):
    """
    Select a GPU if availale.

    selection can be set to get a specific GPU. If left unset, it will REQUIRE that a GPU be selected by environment variable. If -1, the CPU will be selected.
    """

    if str(selection) == "-1":
        return torch.device("cpu")

    # This must be done before any API calls to Torch that touch the GPU
    if selection is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selection)

    if not torch.cuda.is_available():
        print("Selecting CPU (CUDA not available)")
        return torch.device("CPU")
    elif selection is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is *required* when running with CUDA available"
        )

    print(torch.cuda.device_count(), "available GPUs (initially using device 0):")
    for i in range(torch.cuda.device_count()):
        print(" ", i, torch.cuda.get_device_name(i))

    return torch.device("cuda:0")


def trainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    epoch_start=0,
):
    """
    If notebook = None, no progress bar will be drawn. If False, this will be a terminal progress bar.
    """

    # Print all of the hyperparameters of the training iteration
    if not notebook:
        print("{0:=^80}".format(" HYPERPARAMETERS "))
        print(
            f"""\
n_epochs: {n_epochs}
batch_size: {train_loader.batch_size} events
dataset_train: {train_loader.dataset.tensors[0].size()[0]} events
dataset_val: {val_loader.dataset.tensors[0].size()[0]} events
loss: {loss}
optimizer: {optimizer}
model: {model}"""
        )
        print("=" * 80)

    # Set up notebook or regular progress bar (or none)
    progress = import_progress_bar(notebook)

    # Get the current device
    device = get_device_from_model(model)

    print(f"Number of batches: train = {len(train_loader)}, val = {len(val_loader)}")

    epoch_iterator = progress(
        range(epoch_start, n_epochs),
        desc="Epochs",
        postfix="train=start, val=start",
        dynamic_ncols=True,
        position=0,
        file=sys.stderr,
    )

    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)

        # At the end of the epoch, do a pass on the validation set
        total_val_loss, cur_val_eff = validate(model, loss, val_loader, device)
        val_epoch = total_val_loss / len(val_loader)

        # Record total time
        time_epoch = time.time() - training_start_time

        # Pretty print a description
        if hasattr(epoch_iterator, "postfix"):
            epoch_iterator.postfix = f"train={cost_epoch:.4}, val={val_epoch:.4}"

        # Redirect stdout if needed to avoid clash with progress bar
        write = getattr(progress, "write", print)
        write(
            f"Epoch {epoch}: train={cost_epoch:.6}, val={val_epoch:.6}, took {time_epoch:.5} s"
        )
        write(f"  Validation {cur_val_eff}")

        yield Results(epoch, cost_epoch, val_epoch, time_epoch, cur_val_eff)


def train(model, loss, loader, optimizer, device, progress):
    total_loss = 0.0

    # switch to train mode
    model.train()

    loader = progress(
        loader,
        postfix="train=start",
        desc="Training",
        mininterval=0.5,
        dynamic_ncols=True,
        position=1,
        leave=False,
        file=sys.stderr,
    )

    for inputs, labels in loader:
        if inputs.device != device:
            inputs, labels = inputs.to(device), labels.to(device)

        # Set the parameter gradients to zero
        optimizer.zero_grad()

        # Forward pass, backward pass, optimize
        outputs = model(inputs)
        loss_output = loss(outputs, labels)
        loss_output.backward()
        optimizer.step()

        total_loss += loss_output.data.item()

        if hasattr(loader, "postfix"):
            loader.postfix = f"train={loss_output.data.item():.4g}"

    return total_loss


def validate(model, loss, loader, device):
    total_loss = 0
    eff = ValueSet(0, 0, 0, 0)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            if inputs.device != device:
                inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            val_outputs = model(inputs)
            loss_output = loss(val_outputs, labels)

            total_loss += loss_output.data.item()

            for label, output in zip(labels.cpu().numpy(), val_outputs.cpu().numpy()):
                eff += efficiency(label, output, **PARAM_EFF)
    return total_loss, eff
