import torch
from torch import nn
import inspect


class PVModel(nn.Module):
    LEAKYNESS = 0.01
    KERNEL_SIZE = None  # Must be overriden
    CHANNELS_SIZE = None  # Must be overriden
    DEFAULTS = {}  # Can set default dropouts if desired
    FC = True  # Can turn off fully connected linear output layer
    INPUTS = 1  # Number of inputs (1 for z only, or 3 for x and y too)
    FINAL_ACTIVATION = nn.Sigmoid

    def __init__(self, **kwargs):
        """Make a PV layer model.

        You can pass dropout_<x> = value to set dropout for a layer. Layers start at 1.
        You can pass final_activation or set FINAL_ACTIVATION to set the final activation function.
        """
        super().__init__()

        options = {**self.DEFAULTS, **kwargs}

        self.final_activation = (
            options["final_activation"]
            if "final_activation" in options
            else self.FINAL_ACTIVATION
        )

        nlayers = len(self.KERNEL_SIZE)
        assert len(self.CHANNELS_SIZE) == nlayers, "You need as many channels as layers"
        assert all(v % 2 == 1 for v in self.KERNEL_SIZE), "All kernel sizes must be odd"

        # Make local names to use. Add 1 before and after the channels size.
        kernel_size = self.KERNEL_SIZE
        channels_size = (self.INPUTS, *self.CHANNELS_SIZE)

        leaky = nn.LeakyReLU(self.LEAKYNESS)

        # Build up sequential layers
        items = []
        for i in range(nlayers):
            items.append(
                nn.Conv1d(
                    in_channels=channels_size[i],
                    out_channels=channels_size[i + 1],
                    kernel_size=kernel_size[i],
                    stride=1,
                    padding=(kernel_size[i] - 1) // 2,
                )
            )

            # LeakyReLu
            items.append(
                leaky if self.FC or i < nlayers - 1 else self.final_activation()
            )

            # Get dropout if passed in and not none, and add that
            dropout = options.get(f"dropout_{i+1}")
            if dropout is not None:
                items.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*items)

        if self.FC:
            self.fc = nn.Linear(in_features=4000 * channels_size[-1], out_features=4000)

    def forward(self, x):
        x = self.features(x)

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        if self.FC:
            x = self.final_activation()(self.fc(x))

        return x


def write_model(filename, model, loss=None):
    "Write a model and maybe a loss function to a file"
    info_file = filename

    master_model = PVModel if issubclass(model, PVModel) else model

    with open(info_file, "w") as f:
        print("import torch", file=f)
        print("from torch import nn\n", file=f)

        print(inspect.getsource(master_model), end="\n\n\n", file=f)

        if issubclass(model, PVModel):
            print(
                f"""
class Model(PVModel):
    KERNEL_SIZE =   {model.KERNEL_SIZE}
    CHANNELS_SIZE = {model.CHANNELS_SIZE}
    DEFAULTS = {model.DEFAULTS}
    FC = {model.FC}
    LEAKYNESS = {model.LEAKYNESS}
    INPUTS = {model.INPUTS}
    FINAL_ACTIVATION = {model.FINAL_ACTIVATION.__qualname__}
""",
                file=f,
            )

        if loss is not None:
            print(inspect.getsource(loss), file=f)


def modernize(d, layers=2):
    "Convert old style files to the new style"

    # Factor for layers is 2 or 3, depending on dropout.

    # Convert new style only
    if "fc1.weight" not in d:
        return d

    for key in list(d.keys()):
        new_key = None
        if "conv" in key:
            i = (int(key[4]) - 1) * layers
            new_key = f"features.{i}" + key[5:]
        elif "fc1" in key:
            new_key = "fc" + key[3:]
        elif "finalFilter" in key:
            i = (int(4) - 1) * layers
            new_key = f"features.{i}" + key[11:]
        if new_key:
            d[new_key] = d.pop(key)

    return d


def modernize_state(model, state):
    "Match equal length dicts"
    return {key1: item for key1, item in zip(model.state_dict().keys(), state.values())}
