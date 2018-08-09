from contextlib import contextmanager, redirect_stdout, redirect_stderr
import sys

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
    
    if progress is not None:
        with redirect_stdout(DummyTqdmFile(sys.stdout, progress)), redirect_stderr(DummyTqdmFile(sys.stderr, progress)):
            yield old_out
    else:
        yield old_out