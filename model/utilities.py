import sys
import time
import torch
import mlflow 

from contextlib import contextmanager, redirect_stdout, redirect_stderr

#def Params(batch_size, epochs, lr, experiment_name, device, asymmetry_parameter=2.5):
#        return {'batch_size':batch_size, 'epochs':epochs, 'lr':lr, 'experiment_name':experiment_name, 'device':device, 'asymmetry_parameter':asymmetry_parameter}
    
class Params:
    def __init__(self,batch_size,device,epochs,lr,experiment_name,asymmetry_parameter,run_name):
        self.batch_size=batch_size
        self.device=device
        self.epochs=epochs
        self.lr=lr
        self.experiment_name=experiment_name
        self.asymmetry_parameter=asymmetry_parameter
        self.run_name=run_name

def count_parameters(model):
    """
    Counts the total number of parameters in a model
    Args:
        model (Module): Pytorch model, the total number of parameters for this model will be counted. 

    Returns: Int, number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_to_mlflow(stats_dict:dict, step, args=False):
    '''
    Requires that the dictionary be structured as:
    Parameters have the previx "Param: ", metrics have "Metric: ", and artifacts have "Artifact: "
    It will ignore these tags. 
    
    Example: {'Param: Parameters':106125, 'Metric: Training Loss':10.523}
    '''
    for key, value in stats_dict.items():
        if 'Param: ' in key:
            mlflow.log_param(key[7:], value)
        if 'Metric: ' in key:
            mlflow.log_metric(key[8:], value, step)
        if 'Artifact' in key:
            mlflow.log_artifact(value)
#     for key, value in vars(args).items(): # maybe there will be a day when this is wanted for exact experimental reproducability
#         mlflow.log_param(key, value)

def load_full_state(model_to_update, Path, freeze_weights=False):
    """
    Updates the model weights with those given in the model file specified by the Path argument
    The use case for this is if we care about the optimizer state_dict, which we do if we have multiple training 
    sessions with momentum and/or learning rate decay. this will track the decay/momentum.

    Args: 
            model_to_update (Module): Pytorch model with randomly initialized weights. These weights will be updated.
            Path (string): If we are not training from scratch, this path should be the path to the "run_stats" file in the artifacts 
            directory of whatever run you are using as a baseline. 
            You can find the path in the MLFlow UI. It should end in /artifacts/run_stats  
            

    Returns:
            Nothing

    Note:
            The model will not be returned, rather the module you pass to this function will be modified.
    """
    checkpoint = torch.load(Path)
    update_dict = {k: v for k, v in checkpoint.state_dict().items() if k in model_to_update.state_dict()}
    model_to_update.load_state_dict(update_dict, strict=False)
    print('Of the '+str(len(model_to_update.state_dict())/2)+' parameter layers to update in the current model, '+str(len(update_dict)/2)+' were loaded')
        
# def save_summary(model, sample_input):
#     # this part saves the printed output of summary() to a text file
#     orig_stdout = sys.stdout
#     f = open('model_summary.txt', 'w')
#     sys.stdout = f
#     summary(model, sample_input.shape)
#     print(model)
#     sys.stdout = orig_stdout
#     f.close()
#     mlflow.log_artifact('model_summary.txt')
    
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
