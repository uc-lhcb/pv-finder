import math
from typing import Optional

import torch
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import matplotlib as mpl

class NestedProgressBar:
    """
    Manages nested tqdm progress bars for loops like epochs and batches.

    This class provides a convenient way to display and control separate
    progress bars for outer and inner loops (e.g., training epochs and
    data batches). It supports both terminal and notebook environments
    and allows for conditional message logging at specified intervals.
    """
    def __init__(
        self,
        total_epochs,
        total_batches,
        g_epochs=None,
        g_batches=None,
        epoch_message_freq=None,
        batch_message_freq=None,
        mode="train",
    ):
        """
        Initializes the nested progress bars.

        Args:
            total_epochs: The total number of epochs for the process.
            total_batches: The total number of batches in one epoch.
            g_epochs: The visual granularity for the epoch bar. If None,
                      it defaults to total_epochs.
            g_batches: The visual granularity for the batch bar. If None,
                       it defaults to total_batches.
            epoch_message_freq: The frequency (in epochs) to log messages.
            batch_message_freq: The frequency (in batches) to log messages.
            mode: The operational mode, either 'train' or 'eval'. 'train'
                  mode shows both epoch and batch bars, while 'eval'
                  mode only shows the batch bar.
        """
        # Set the operational mode ('train' or 'eval')
        self.mode = mode

        # Dynamically import the appropriate tqdm implementation
        from tqdm.auto import tqdm as tqdm_impl

        self.tqdm_impl = tqdm_impl

        # Store the actual total counts for epochs and batches
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches

        # Determine the visual granularity for the progress bars
        self.g_epochs = min(g_epochs or total_epochs, total_epochs)
        self.g_batches = min(g_batches or total_batches, total_batches)

        # Set the total steps for the progress bars based on granularity
        self.total_epochs = self.g_epochs
        self.total_batches = self.g_batches

        # Initialize the tqdm progress bar instances based on the mode
        if self.mode == "train":
            # Outer bar for epochs
            self.epoch_bar = self.tqdm_impl(
                total=self.total_epochs, desc="Current Epoch", position=0, leave=True
            )
            # Inner bar for batches
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Current Batch", position=1, leave=False
            )
        elif self.mode == "eval":
            # No epoch bar needed for evaluation
            self.epoch_bar = None
            # A single bar for evaluation progress
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Evaluating", position=0, leave=False
            )

        # Keep track of the last updated visual step to avoid redundant updates
        self.last_epoch_step = -1
        self.last_batch_step = -1

        # Store the frequency for logging messages
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq

    def update_epoch(self, epoch, postfix_dict=None):
        """
        Updates the epoch progress bar and resets the batch bar.

        Args:
            epoch: The current epoch number (1-indexed).
            postfix_dict: An optional dictionary of metrics to display
                          on the epoch bar.
        """
        # Calculate the visual step based on the current epoch and granularity
        epoch_step = math.floor((epoch - 1) * self.g_epochs / self.total_epochs_raw)

        # Update the bar only when the visual step changes
        if epoch_step != self.last_epoch_step:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step
        # Ensure the bar completes on the final epoch
        elif epoch == self.total_epochs_raw and self.epoch_bar.n < self.g_epochs:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step

        # Update the description and postfix for the epoch bar in train mode
        if self.mode == "train":
            self.epoch_bar.set_description(f"Training - Current Epoch: {epoch}")
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)

        # Reset the batch bar for the new epoch
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_batch(self, batch, postfix_dict=None):
        """
        Updates the batch progress bar.

        Args:
            batch: The current batch number (1-indexed).
            postfix_dict: An optional dictionary of metrics to display
                          on the batch bar.
        """
        # Calculate the visual step based on the current batch and granularity
        batch_step = math.floor((batch - 1) * self.g_batches / self.total_batches_raw)

        # Update the bar only when the visual step changes
        if batch_step != self.last_batch_step:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step
        # Ensure the bar completes on the final batch
        elif batch == self.total_batches_raw and self.batch_bar.n < self.g_batches:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step

        # Update the description of the batch bar based on the mode
        if self.mode == "train":
            self.batch_bar.set_description(f"Training - Current Batch: {batch}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"Evaluation - Current Batch: {batch}")

        # Set any provided metrics on the batch bar
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch, message):
        """
        Prints a message at a specified epoch frequency.

        Args:
            epoch: The current epoch number.
            message: The message to print.
        """
        # Check if logging is enabled and if the current epoch is a logging interval
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            print(message)

    def maybe_log_batch(self, batch, message):
        """
        Prints a message at a specified batch frequency.

        Args:
            batch: The current batch number.
            message: The message to print.
        """
        # Check if logging is enabled and if the current batch is a logging interval
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            print(message)

    def close(self, last_message=None):
        """
        Closes the progress bars and optionally prints a final message.

        Args:
            last_message: An optional final message to print after closing
                          the bars.
        """
        # Close the epoch bar if it exists (in 'train' mode)
        if self.mode == "train":
            self.epoch_bar.close()
        # Close the batch bar
        self.batch_bar.close()

        # Print a final message if one is provided
        if last_message:
            print(last_message)