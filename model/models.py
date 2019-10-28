import torch
from torch import nn

from .core import PVModel


class SimpleCNN2Layer(PVModel):
    KERNEL_SIZE = [25, 15]
    CHANNELS_SIZE = [5, 1]


class SimpleCNN3Layer(PVModel):
    KERNEL_SIZE = [25, 15, 5]
    CHANNELS_SIZE = [10, 5, 1]
    DEFAULTS = {"dropout_3": 0.35}


class SimpleCNN3Layer_A(PVModel):
    KERNEL_SIZE = [25, 15, 5]
    CHANNELS_SIZE = [20, 5, 1]
    DEFAULTS = {"dropout_1": 0.15, "dropout_3": 0.35}


class SimpleCNN3Layer_B(PVModel):
    KERNEL_SIZE = [25, 15, 5]
    CHANNELS_SIZE = [20, 10, 1]
    DEFAULTS = {"dropout_1": 0.15, "dropout_2": 0.15, "dropout_3": 0.35}


class SimpleCNN3Layer_C(PVModel):
    KERNEL_SIZE = [25, 15, 5]
    CHANNELS_SIZE = [20, 10, 1]
    DEFAULTS = {"dropout_1": 0.15, "dropout_2": 0.15, "dropout_3": 0.35}


class All_CNN3Layer_C(PVModel):
    KERNEL_SIZE = [25, 15, 5]
    CHANNELS_SIZE = [20, 10, 1]
    DEFAULTS = {"dropout_1": 0.15, "dropout_2": 0.15, "dropout_3": 0.35}


class SimpleCNN4Layer_C(PVModel):
    KERNEL_SIZE = [25, 15, 15, 5]
    CHANNELS_SIZE = [20, 10, 10, 1]
    DEFAULTS = {
        "dropout_1": 0.15,
        "dropout_2": 0.15,
        "dropout_3": 0.15,
        "dropout_4": 0.35,
    }


class SimpleCNN4Layer_D(PVModel):
    KERNEL_SIZE = [25, 15, 15, 5]
    CHANNELS_SIZE = [25, 25, 25, 1]
    DEFAULTS = {
        "dropout_1": 0.55,
        "dropout_2": 0.55,
        "dropout_3": 0.55,
        "dropout_4": 0.35,
    }


class SimpleCNN4Layer_D35(PVModel):
    KERNEL_SIZE = [25, 15, 15, 5]
    CHANNELS_SIZE = [25, 25, 25, 1]
    DEFAULTS = {
        "dropout_1": 0.35,
        "dropout_2": 0.35,
        "dropout_3": 0.35,
        "dropout_4": 0.35,
    }


class SimpleCNN4Layer_D35_sp(PVModel):
    KERNEL_SIZE = [25, 15, 15, 5]
    CHANNELS_SIZE = [25, 25, 25, 1]
    DEFAULTS = {
        "dropout_1": 0.35,
        "dropout_2": 0.35,
        "dropout_3": 0.35,
        "dropout_4": 0.35,
    }
    FINAL_ACTIVATION = nn.Softplus


class SimpleCNN4Layer_D25(PVModel):
    KERNEL_SIZE = [25, 15, 15, 5]
    CHANNELS_SIZE = [25, 25, 25, 1]
    DEFAULTS = {
        "dropout_1": 0.25,
        "dropout_2": 0.25,
        "dropout_3": 0.25,
        "dropout_4": 0.25,
    }


class SimpleCNN5Layer_C(PVModel):
    KERNEL_SIZE = [25, 15, 15, 15, 5]
    CHANNELS_SIZE = [20, 10, 10, 10, 1]
    DEFAULTS = {
        "dropout_1": 0.15,
        "dropout_2": 0.15,
        "dropout_3": 0.15,
        "dropout_4": 0.15,
        "dropout_5": 0.35,
    }


class AltCNN4Layer_D35_sp(PVModel):
    KERNEL_SIZE = [25, 15, 15, 5, 91]
    CHANNELS_SIZE = [25, 25, 25, 1, 1]
    DEFAULTS = {
        "dropout_1": 0.05,
        "dropout_2": 0.05,
        "dropout_3": 0.05,
        "dropout_4": 0.05,
    }
    FINAL_ACTIVATION = nn.Softplus
    FC = False
