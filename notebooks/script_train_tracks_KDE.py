import torch
import mlflow
import numpy as np
from tqdm import tqdm

from sam import SAM

from model.collectdata_tracks_KDE import load_prebuilt_dataset
from model.alt_loss_A import Loss, kde_loss_Ba
from model.training import trainNet

from model.training import train_tracks_kde
from model.utilities import load_full_state, count_parameters, Params, save_to_mlflow
from model.autoencoder_models import UNet
from model.tracks_to_KDE_models import TorchTransformer

args = Params(
    batch_size=64,
    device = 'cuda:0',
    epochs=300,
    lr=5e-6,
    experiment_name='Transformer', 
    asymmetry_parameter=2.5
)

dataset = load_prebuilt_dataset(batch_size=16, num_files=1)
dataset_iterator = iter(dataset)
means = np.zeros(9)
stds = np.zeros(9)
j = 0
for _ in range(20):
    j += 1
    sample = next(dataset_iterator)
    sample[0] *= (sample[0] > -98).int()
    for i in range(9):
        stds[i] += sample[0][:, i, :][sample[0][:, i, :]!=0].std()
        means[i] += sample[0][:, i, :].sum()/(sample[0][:, i, :]!=0).sum()


mlflow.tracking.set_tracking_uri('file:/share/lazy/pv-finder_model_repo')
mlflow.set_experiment(args['experiment_name'])

transformer = TorchTransformer(1, 9, 125, n_layers=2)
# optimizer = torch.optim.Adam(transformer.parameters(), lr=args['lr'])
optimizer = SAM(transformer.parameters(), torch.optim.Adam, lr=args['lr'])
# loss = kde_loss_Ba(epsilon=1e-5,coefficient=2.5)
loss = torch.nn.BCELoss()

transformer.register_buffer('stds', torch.Tensor(stds))
transformer.register_buffer('means', torch.Tensor(means))
transformer = transformer.to(args['device'])

load_full_state(model, optimizer, '/share/lazy/pv-finder_model_repo/14/924ac3a968fc422798e00bd1ad9790c2/artifacts/run_stats.pyt', freeze_weights=False)

run_name = 'transformer refactor test'

with mlflow.start_run(run_name = run_name) as run:
    dataset = load_prebuilt_dataset(batch_size=16, num_files=1)

    for epoch in range(args['epochs']):
        
        train_loss = train_tracks_kde(transformer, loss, dataset, optimizer, args['device'])
        
        torch.save(transformer, 'run_stats.pyt')
        save_to_mlflow({
            'Metric: Training loss':train_loss,
#             'Metric: Validation loss':result.val,
#             'Metric: Efficiency':result.eff_val.eff_rate,
#             'Metric: False positive rate':result.eff_val.fp_rate,
            'Artifact':'run_stats.pyt'
        }, step=epoch)
        
        tracks, kdes = next(iter(dataset))
        pred = transformer(tracks.to(args['device']))
        
        import matplotlib.pyplot as plt
        for i in range(kdes.size(0)):
            plt.autoscale(axis='y')
            plt.plot(kdes[i][:4000].detach().cpu().numpy(), label='truth')
            plt.plot(torch.sigmoid(pred[i].squeeze(0)).detach().cpu().numpy(), label='pred')
            plt.legend()
            plt.savefig(f'plot_{i}.png')
            plt.clf()

