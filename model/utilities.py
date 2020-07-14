from contextlib import contextmanager, redirect_stdout, redirect_stderr
import sys
import time
import torch

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""

    __slots__ = ("file", "progress")

    def __init__(self, file, progress):
        self.file = file
        self.progress = progress

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            self.progress.write(x.strip(), file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


@contextmanager
def tqdm_redirect(progress):
    old_out = sys.stdout

    if hasattr(progress, "postfix"):
        with redirect_stdout(DummyTqdmFile(sys.stdout, progress)), redirect_stderr(
            DummyTqdmFile(sys.stderr, progress)
        ):
            yield old_out
    else:
        yield old_out


def import_progress_bar(notebook):
    """Set up notebook or regular progress bar.

    If None or if piping to a file, just provide an empty do-nothing function."""

    def progress(iterator, **kargs):
        return iterator

    if notebook is None:
        pass
    elif notebook:
        from tqdm import tqdm_notebook as progress
    elif sys.stdout.isatty():
        from tqdm import tqdm as progress
    else:
        # Don't display progress if this is not a
        # notebook and not connected to the terminal
        pass

    return progress


class Timer(object):
    __slots__ = "message verbose start_time".split()

    def __init__(self, message=None, start=None, verbose=True):
        """
        If message is None, add a default message.
        If start is not None, then print start then message.
        Turn off all printing with verbose.
        """

        if verbose and start is not None:
            print(start, end="", flush=True)
        if message is not None:
            self.message = message
        elif start is not None:
            self.message = " took {time:.4} s"
        else:
            self.message = "Operation took {time:.4} s"

        self.verbose = verbose
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.verbose:
            print(self.message.format(time=self.elapsed_time()))


def get_device_from_model(model):
    if hasattr(model, "weight"):
        return model.weight.device
    else:
        return get_device_from_model(list(model.children())[0])
    

def load_full_state(model_to_update, optimizer_to_update, Path, freeze_weights=False):
    """
    Load the model and optimizer state_dict, and the total number of epochs
    The use case for this is if we care about the optimizer state_dict, which we do if we have multiple training 
    sessions with momentum and/or learning rate decay. this will track the decay/momentum.

    Args: 
            model_to_update (Module): Pytorch model with randomly initialized weights. These weights will be updated.
            optimizer_to_update (Module): Optimizer with your learning rate set. 
            THIS FUNCTION WILL NOT UPDATE THE LEARNING RATE YOU SPECIFY.
            Path (string): If we are not training from scratch, this path should be the path to the "run_stats" file in the artifacts 
            directory of whatever run you are using as a baseline. 
            You can find the path in the MLFlow UI. It should end in /artifacts/run_stats  
            

    Returns:
            Nothing

    Note:
            The model and optimizer will not be returned, rather the optimizer and module you pass to this function will be modified.
    """
    checkpoint = torch.load(Path)
    
    # freeze weights of the first model
    update_dict = {k: v for k, v in checkpoint['model'].items() if k in model_to_update.state_dict()}
                # do this so it does not use the learning rate from the previous run. this is unwanted behavior
                # in our scenario since we are not using a learning rate scheduler, rather we want to tune the learning
                # rate further after we have gotten past the stalling
            #     checkpoint['optimizer']['param_groups'][0]['lr'] = optimizer_to_update.state_dict()['param_groups'][0]['lr']
            #     optimizer_to_update.load_state_dict(checkpoint['optimizer'])
    
    # to go back to old behavior, just do checkpoint['model'] instead of update_dict
    model_to_update.load_state_dict(update_dict, strict=False)

    ct = 0
    if freeze_weights:
        for k, v in model_to_update.named_children():
            if ((k+'.weight' in checkpoint['model'].keys()) | (k+'.bias' in checkpoint['model'].keys())) & (k != 'Dropout'):
                v.weight.requires_grad = False
                v.bias.requires_grad = False
                ct += 1
                        
    print('we also froze {} weights'.format(ct))
    
    print('Of the '+str(len(model_to_update.state_dict())/2)+' parameter layers to update in the current model, '+str(len(update_dict)/2)+' were loaded')


def count_parameters(model):
    """
    Counts the total number of parameters in a model
    Args:
        model (Module): Pytorch model, the total number of parameters for this model will be counted. 

    Returns: Int, number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Params(object):
    def __init__(self, batch_size, epochs, lr, epoch_start=0):
        self.epoch_start = epoch_start
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr