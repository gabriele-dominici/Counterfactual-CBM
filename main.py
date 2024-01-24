import os
import wandb
import yaml
import torch
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger
from ccbm.utils import randomize_class, load_data, RandomSamplerClassBatch
from ccbm.models import (StandardE2E, StandardCBM, Oracle, 
                         CounterfactualCBM_V3, CounterfactualCBM_V3_1, CounterfactualDCR_V3, 
                         VAE_CF, CCHVAE, ConceptVCNet, StandardDCR)

from train import train


# Login to wandb
wandb.login()

# Load config file
with open("config/config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Initialize wandb
run = wandb.init(config=config, dir='./wandb')
wandb_logger = WandbLogger()

# Get config variables
seed = wandb.config.seed
batch_size = wandb.config.batch_size
dataset_name = wandb.config.dataset
model = wandb.config.model
emb_size = wandb.config.emb_size
device = wandb.config.device
learning_rate = wandb.config.learning_rate[dataset_name]
epochs = wandb.config.epochs[dataset_name]
fold = {'0': 0, '8': 1, '13': 2, '24': 3, '42': 4}[str(seed)]

# Set seed
seed_everything(seed, workers=True)

# Load train data
X, c, y = load_data(dataset_name)
if len(y.shape) == 1:
    y = y.unsqueeze(1)
    # Unique [concepts, labels]
    c_cf_set = torch.unique(torch.cat((c, y), dim=-1), dim=0)
    concept_labels = c_cf_set[:, -1]
    c_cf_set = c_cf_set[:, :-1]
else:
    c_cf_set = torch.unique(torch.cat((c, torch.argmax(y, dim=-1).unsqueeze(-1)), dim=-1), dim=0)
    concept_labels = c_cf_set[:, -1]
    c_cf_set = c_cf_set[:, :-1]
    concept_labels = one_hot(concept_labels.long(), num_classes=y.shape[1]).float()
# Load test data
X_test, c_test, y_test = load_data(dataset_name, split='test')
if len(y_test.shape) == 1:
    y_test = y_test.unsqueeze(1)

            
# creates directory for results
results_root_dir = f"./results/"
os.makedirs(results_root_dir, exist_ok=True)
results_dir = f"./results/{dataset_name}/"
os.makedirs(results_dir, exist_ok=True)
figures_dir = f"./results/{dataset_name}/figures/"
os.makedirs(figures_dir, exist_ok=True)
log_dir = f"./results/{dataset_name}/logs/"
os.makedirs(log_dir, exist_ok=True)

# Define model
models = {
            'Oracle': Oracle(),
            'DeepNN': StandardE2E(X.shape[1], y.shape[1], emb_size, learning_rate),
            'StandardCBM': StandardCBM(X.shape[1], c.shape[1], y.shape[1], emb_size, learning_rate, bool_concepts=True, deep=True),
            'StandardDCR': StandardDCR(X.shape[1], c.shape[1], y.shape[1], emb_size, learning_rate),
            'CFCBM': CounterfactualCBM_V3(X.shape[1], c.shape[1], y.shape[1], emb_size, c_cf_set, concept_labels, learning_rate=learning_rate, resample=0, bernulli=False, deep=True, reconstruction=False, dataset=dataset_name) if dataset_name != 'cub'
                    else CounterfactualCBM_V3_1(X.shape[1], c.shape[1], y.shape[1], emb_size, c_cf_set, concept_labels, learning_rate=learning_rate, resample=0, bernulli=False, deep=True, reconstruction=False),
            'VAECF': (StandardCBM(X.shape[1], c.shape[1], y.shape[1], emb_size, learning_rate, bool_concepts=True, deep=True), 
                      VAE_CF(c.shape[1], y.shape[1], emb_size, None, learning_rate)),
            'CCHVAE': (StandardCBM(X.shape[1], c.shape[1], y.shape[1], emb_size, learning_rate, bool_concepts=True, deep=True),
                       CCHVAE(c.shape[1], y.shape[1], emb_size, None, learning_rate)),
            'VCNET': ConceptVCNet(X.shape[1], c.shape[1], y.shape[1], emb_size, learning_rate, bool_concepts=True, deep=True),
            'BayCon': StandardCBM(X.shape[1], c.shape[1], y.shape[1], emb_size, learning_rate, bool_concepts=True, deep=True),
        }
net = models[model]
           
train_data = TensorDataset(X, c, y)
train_dl = torch.utils.data.DataLoader(train_data, batch_sampler=torch.utils.data.BatchSampler(RandomSamplerClassBatch(y, batch_size=batch_size, replacement=False), batch_size=batch_size, drop_last=True), pin_memory=True)
test_data = TensorDataset(X_test, c_test, y_test)
test_dl = torch.utils.data.DataLoader(test_data, batch_sampler=torch.utils.data.BatchSampler(RandomSamplerClassBatch(y_test, batch_size=batch_size, replacement=False), batch_size=batch_size,  drop_last=True), pin_memory=True)

# train model
results, net = train(net,
                      train_dl, test_dl, 
                      epochs, device, learning_rate, emb_size, c_cf_set, concept_labels, batch_size,
                      log_dir, figures_dir, results_dir, 
                      fold, model, seed, wandb_logger)

# save results
for key, value in results.items():
    wandb.run.log({key: value})
# torch.save(results, f'{results_dir}/{model}_{fold}.pt')





