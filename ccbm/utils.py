import torch
import numpy as np
import os
import shap
from ccbm.data import xor, trigonometry
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib.pyplot import draw
from sklearn.linear_model import LinearRegression

def randomize_class(a, include=True):
        # Get the number of classes and the number of samples
        num_classes = a.size(1)
        num_samples = a.size(0)

        # Generate random indices for each row to place 1s, excluding the original positions
        random_indices = torch.randint(0, num_classes, (num_samples,)).to(a.device)

        # Ensure that the generated indices are different from the original positions
        # TODO we inclue also same label to make sure that every class is represented 
        if not include:
          original_indices = a.argmax(dim=1)
          random_indices = torch.where(random_indices == original_indices, (random_indices + 1) % num_classes, random_indices)

        # Create a second tensor with 1s at the random indices
        b = torch.zeros_like(a)
        b[torch.arange(num_samples), random_indices] = 1
        return b

class RandomSamplerClassBatch(torch.utils.data.RandomSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, replacement=False,
                 num_samples=None, generator=None, batch_size=None) -> None:
        super().__init__(data_source, replacement, num_samples, generator)
        self.batch_size = batch_size   

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        elif self.batch_size is not None:
            labels = self.data_source.shape[1]
            all_indexes = list(range(n))
            for _ in range(n // self.batch_size):
                batch = []
                for i in range(labels):
                    filtered_idxes = torch.where(self.data_source[:, i] == 1)[0].numpy()
                    idxes = list(set(all_indexes) & set(filtered_idxes))
                    try:
                        el = np.random.choice(idxes, size=1, replace=False)
                    except:
                        el = np.random.choice(filtered_idxes, size=1, replace=False)
                    all_indexes = list(set(all_indexes) - set([el[0]]))
                    batch += [el[0]]
                b_indexes = torch.randperm(len(all_indexes), generator=generator).tolist()[:self.batch_size - labels]
                b_indexes = [all_indexes[i] for i in b_indexes]
                yield from batch + b_indexes
                all_indexes = list(set(all_indexes) - set(b_indexes))
            batch = []
            for i in range(labels):
                filtered_idxes = torch.where(self.data_source[:, i] == 1)[0].numpy()
                idxes = list(set(all_indexes) & set(filtered_idxes))
                try:
                    el = np.random.choice(idxes, size=1, replace=False)
                except:
                    el = np.random.choice(filtered_idxes, size=1, replace=False)
                all_indexes = list(set(all_indexes) - set([el[0]]))
                batch += [el[0]]
            b_indexes = torch.randperm(len(all_indexes), generator=generator).tolist()[:self.num_samples % self.batch_size - labels]
            b_indexes = [all_indexes[i] for i in b_indexes]
            yield from batch + b_indexes
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]


def sample_bernoulli(distribution):
    sig = torch.sigmoid(distribution.logits)
    return distribution.sample() - sig.detach() + sig

def extract_concepts(v):
      idxes = np.where(v == 1)[0]
      if len(idxes) == 0:
        return "None"
      result = ["c_"+str(idx) for idx in idxes]
      return " ".join(result)

def load_data(dataset_name, n_samples=0, random_seed=42, fold=1, split='train'):
    if dataset_name == 'xor':
        load_data = xor
        X, c, y = load_data(n_samples, random_seed+fold)
    elif dataset_name == 'trigonometry':
        load_data = trigonometry
        X, c, y = load_data(n_samples, random_seed+fold)
    else:   
        save_dir = f'./embeddings/{dataset_name}'
        train_embeddings_file = os.path.join(save_dir, f'{split}_embeddings.pt')
        X, c, y = torch.load(train_embeddings_file)
    return X, c, y

def plot_latents(z2, z3, y_preds, y_cf):
    # unify z2 and z3
    zeros = torch.zeros(z2.shape[0])
    ones = torch.ones(z3.shape[0])
    labels = torch.cat((zeros, ones), dim=0).detach().numpy()
    z2_concat = torch.concat((z2, z3), dim=0).detach().numpy()
    z2_embedded = TSNE(n_components=2).fit_transform(z2_concat)
    # divide again z2 and z3 
    z2 = z2_embedded[:z2.shape[0]]
    z3 = z2_embedded[z2.shape[0]:]
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter1 = ax.scatter(z2[:1000, 0], z2[:1000, 1], c=(y_preds[:1000].argmax(dim=-1)).detach().numpy(), cmap=plt.cm.RdBu, s=150, edgecolors='w')
    scatter2 = ax.scatter(z3[:1000, 0], z3[:1000, 1], c=(y_cf[:1000].argmax(dim=-1)).detach().numpy(), cmap=plt.cm.RdBu, s=150, edgecolors='w', marker='x')
    legend1 = ax.legend(*scatter1.legend_elements(),
                        loc="lower left", title="Classes", fontsize='12')
    ax.add_artist(legend1)
    plt.title('t-SNE on z2 and z3', fontsize=32)
    plt.tight_layout()
    # plt.show()
    # draw()
    import seaborn as sns
    z = pd.DataFrame(z2_embedded, columns=['f1', 'f2'])
    z['z'] = labels
    sns.kdeplot(data=z, x="f1", y="f2", hue="z")
    # plt.show()
    # draw()


def save_set_c_and_cf(c_preds, y_preds, y_cf, c_cf, model_name, fold, log_dir):
    cf_bool = (c_cf > 0.5).float().detach().numpy()
    cf_bool = [extract_concepts(el) for el in cf_bool]
    y_cf_index = y_cf.argmax(dim=-1).numpy()
    y_cf_index = [str(el) for el in y_cf_index]
    cf_df = pd.DataFrame(zip(cf_bool, y_cf_index))
    counts = cf_df.groupby([0,1]).value_counts().sort_values(ascending=False).reset_index(name='count')
    counts.to_csv(os.path.join(log_dir, f'CF_{model_name}_{fold}.csv'))
    c_bool = (c_preds > 0.5).float().detach().numpy()
    c_bool = [extract_concepts(el) for el in c_bool]
    y_pred_index = y_preds.argmax(dim=-1).numpy()
    y_pred_index = [str(el) for el in y_pred_index]
    c_df = pd.DataFrame(zip(c_bool, y_pred_index))
    counts = c_df.groupby([0,1]).value_counts().sort_values(ascending=False).reset_index(name='count')
    counts.to_csv(os.path.join(log_dir, f'C_{model_name}_{fold}.csv'))

def print_concept_importance(net, idx, c_preds_total_train, c_preds_total, show=True):
    weights = net.reasoner[0].weight
    biases = net.reasoner[0].bias
    if weights.shape[0] != 1:
        weights = weights[idx]
        biases = biases[idx]
    else:
        weights = weights[0]
        biases = biases[0]
    weights = weights.data.numpy()
    biases = biases.data.numpy()
    sklearn_model = LinearRegression()
    sklearn_model.coef_ = weights
    sklearn_model.intercept_ = biases
    explainer = shap.Explainer(
        sklearn_model, c_preds_total_train.detach().numpy(), feature_names=FEATURE_NAMES[str(c_preds_total_train.shape[-1])]
    )
    shap_values = explainer(c_preds_total.detach().numpy())
    # shap.plots.beeswarm(shap_values)
    fig = plt.figure()
    weights = np.divide(weights, np.max(np.absolute(weights), axis=-1, keepdims=True))
    shap_values.values = np.round(c_preds_total.detach().numpy()*weights, 2)
    shap_values.data = np.round(shap_values.data, 2)
    order = FEATURE_NAMES[str(c_preds_total_train.shape[-1])]
    col2num = {col: i for i, col in enumerate(order)}

    order = list(map(col2num.get, order))

    ax = shap.plots.beeswarm(shap_values, show=False, order=order)
    ax.set_xlabel('Impact on model output', fontsize=30)
    ax.set_title(DATASET[str(c_preds_total_train.shape[-1])], fontsize=32)
    if show:
        fig.show()
    fig.savefig('ccbm_importance.pdf', format="pdf", bbox_inches="tight")
  

CLASS_TO_VISUALISE = {'20': 15, '112': 0, '7':1}
DATASET = {'20': 'MNIST add', '112': 'CUB', '7':'dSprites'}
FEATURE_NAMES = {'7': ['Shape1', 'Shape2', 'Shape3', 'Two obj', 'Color 1', 'Color 2', 'Color 3'],
                 '20': ['Zero 1',
                        'One 1',
                        'Two 1',
                        'Three 1',
                        'Four 1',
                        'Five 1',
                        'Six 1',
                        'Seven 1',
                        'Eight 1',
                        'Nine 1',
                        'Zero 2',
                        'One 2',
                        'Two 2',
                        'Three 2',
                        'Four 2',
                        'Five 2',
                        'Six 2',
                        'Seven 2',
                        'Eight 2',
                        'Nine 2',], 
                  '112': ["Bill Shape - Dagger",
                        "Bill Shape - Hooked (Seabird)",
                        "Bill Shape - All-Purpose",
                        "Bill Shape - Cone",
                        "Wing Color - Brown",
                        "Wing Color - Grey",
                        "Wing Color - Yellow",
                        "Wing Color - Black",
                        "Wing Color - White",
                        "Wing Color - Buff",
                        "Upperparts Color - Brown",
                        "Upperparts Color - Grey",
                        "Upperparts Color - Yellow",
                        "Upperparts Color - Black",
                        "Upperparts Color - White",
                        "Upperparts Color - Buff",
                        "Underparts Color - Brown",
                        "Underparts Color - Grey",
                        "Underparts Color - Yellow",
                        "Underparts Color - Black",
                        "Underparts Color - White",
                        "Underparts Color - Buff",
                        "Breast Pattern - Solid",
                        "Breast Pattern - Striped",
                        "Breast Pattern - Multi-Colored",
                        "Back Color - Brown",
                        "Back Color - Grey",
                        "Back Color - Yellow",
                        "Back Color - Black",
                        "Back Color - White",
                        "Back Color - Buff",
                        "Tail Shape - Notched",
                        "Upper Tail Color - Brown",
                        "Upper Tail Color - Grey",
                        "Upper Tail Color - Black",
                        "Upper Tail Color - White",
                        "Upper Tail Color - Buff",
                        "Head Pattern - Eyebrow",
                        "Head Pattern - Plain",
                        "Breast Color - Brown",
                        "Breast Color - Grey",
                        "Breast Color - Yellow",
                        "Breast Color - Black",
                        "Breast Color - White",
                        "Breast Color - Buff",
                        "Throat Color - Grey",
                        "Throat Color - Yellow",
                        "Throat Color - Black",
                        "Throat Color - White",
                        "Throat Color - Buff",
                        "Eye Color - Black",
                        "Bill Length - Same as Head",
                        "Bill Length - Shorter than Head",
                        "Forehead Color - Blue",
                        "Forehead Color - Brown",
                        "Forehead Color - Grey",
                        "Forehead Color - Yellow",
                        "Forehead Color - Black",
                        "Forehead Color - White",
                        "Under Tail Color - Brown",
                        "Under Tail Color - Grey",
                        "Under Tail Color - Black",
                        "Under Tail Color - White",
                        "Under Tail Color - Buff",
                        "Nape Color - Brown",
                        "Nape Color - Grey",
                        "Nape Color - Yellow",
                        "Nape Color - Black",
                        "Nape Color - White",
                        "Nape Color - Buff",
                        "Belly Color - Brown",
                        "Belly Color - Grey",
                        "Belly Color - Yellow",
                        "Belly Color - Black",
                        "Belly Color - White",
                        "Belly Color - Buff",
                        "Wing Shape - Rounded",
                        "Wing Shape - Pointed",
                        "Size - Small (5-9 in)",
                        "Size - Medium (9-16 in)",
                        "Size - Very Small (3-5 in)",
                        "Body Shape - Duck-like",
                        "Body Shape - Perching-like",
                        "Back Pattern - Solid",
                        "Back Pattern - Striped",
                        "Back Pattern - Multi-Colored",
                        "Tail Pattern - Solid",
                        "Tail Pattern - Striped",
                        "Tail Pattern - Multi-Colored",
                        "Belly Pattern - Solid",
                        "Primary Color - Brown",
                        "Primary Color - Grey",
                        "Primary Color - Yellow",
                        "Primary Color - Black",
                        "Primary Color - White",
                        "Primary Color - Buff",
                        "Leg Color - Grey",
                        "Leg Color - Black",
                        "Leg Color - Buff",
                        "Bill Color - Grey",
                        "Bill Color - Black",
                        "Bill Color - Buff",
                        "Crown Color - Blue",
                        "Crown Color - Brown",
                        "Crown Color - Grey",
                        "Crown Color - Yellow",
                        "Crown Color - Black",
                        "Crown Color - White",
                        "Wing Pattern - Solid",
                        "Wing Pattern - Spotted",
                        "Wing Pattern - Striped",
                        "Wing Pattern - Multi-Colored"]}

SELECTED_CONCEPTS_CUB = [
    1,
    4,
    6,
    7,
    10,
    14,
    15,
    20,
    21,
    23,
    25,
    29,
    30,
    35,
    36,
    38,
    40,
    44,
    45,
    50,
    51,
    53,
    54,
    56,
    57,
    59,
    63,
    64,
    69,
    70,
    72,
    75,
    80,
    84,
    90,
    91,
    93,
    99,
    101,
    106,
    110,
    111,
    116,
    117,
    119,
    125,
    126,
    131,
    132,
    134,
    145,
    149,
    151,
    152,
    153,
    157,
    158,
    163,
    164,
    168,
    172,
    178,
    179,
    181,
    183,
    187,
    188,
    193,
    194,
    196,
    198,
    202,
    203,
    208,
    209,
    211,
    212,
    213,
    218,
    220,
    221,
    225,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    244,
    249,
    253,
    254,
    259,
    260,
    262,
    268,
    274,
    277,
    283,
    289,
    292,
    293,
    294,
    298,
    299,
    304,
    305,
    308,
    309,
    310,
    311,
]
