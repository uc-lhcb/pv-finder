from .efficiency import efficiency, ValueSet
from model.alt_loss_A import Loss
import time
from model.training import PARAM_EFF, Results
from collections import OrderedDict
import pytorch_lightning as pl

class BaseTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print(self.args)
        self.loss = Loss(epsilon=1e-5, coefficient=self.args.asymmetry_parameter)
        self.time_start = time.time()
        
        self.train_step, self.global_step, self.train_loss, self.val_loss, self.val_step = 0, 0, 0, 0, 0
        self.eff = ValueSet(0, 0, 0, 0)
        self.batch_stats = {}
        
    def training_step(self, batch, batch_idx):
        self.train_step += 1
        inputs, labels = batch
        
        outputs = self.forward(inputs)
        loss_output = self.loss(outputs, labels)
        
        self.train_loss += loss_output.data.item()
        
        return loss_output 
    
    def validation_step(self, batch, batch_idx):
        self.val_step += 1
        inputs, labels = batch
        
        val_outputs = self.forward(inputs)
        loss_output = self.loss(val_outputs, labels)
        
        self.val_loss += loss_output.data.item()
        
        for label, output in zip(labels.cpu().numpy(), val_outputs.cpu().numpy()):
            self.eff += efficiency(label, output, **PARAM_EFF)
        
        return loss_output 

    def on_epoch_start(self):
        self.train_step, self.val_step, self.train_loss, self.val_loss = 0, 0, 0, 0
        self.eff = ValueSet(0, 0, 0, 0)
        self.time_start = time.time()
        
    def on_validation_epoch_start(self):
        self.epoch_time = self.time_start - time.time()

    def on_validation_epoch_end(self):
        self.global_step += self.train_step
        if self.global_step > 0:
            results = Results(1, self.train_loss/self.global_step, self.val_loss/self.val_step, self.epoch_time, self.eff)
            result = results._asdict()
            self.batch_stats = {
                'Efficiency':result['eff_val'].eff_rate,
                'False Positive Rate':result['eff_val'].fp_rate,
                'Validation Loss':result['val']*2,
                'Training Loss':result['cost']*2,
            }

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)