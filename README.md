# Climbing the Ladder of Interpretability with Counterfactual Concept Bottleneck Models

Repository related to the **"Climbing the Ladder of Interpretability with Counterfactual Concept Bottleneck Models"** paper submitted to IJCAI 2024.

### Instruction to set up the repository

**Requirement**: Python 3.9+

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Datasets

To reproduce the experiments discussed in out paper, you will have to download the CUB dataset found [here](https://worksheets.codalab.org/bundles/0xd013a7ba2e88481bbc07e787f73109f5), and the entire directory should be placed into ```./experiments/cub/```.

To run all the experiments then you should run the following: 
(you don't need to run the first one as we already provided the files inside ```./embeddings/dsprites/```)

```
python3 experiments/dsprites/save_embeddings.py
python3 experiments/mnist/save_embeddings.py
python3 experiments/cub/save_embeddings.py
```

### Running experiments

**Requirement**: Wandb already configured (see [Section 1 and 2](https://docs.wandb.ai/quickstart))

We set up a Jupyter notebook with a demo that trains and execute the experiments on the dSprites dataset (```./demo.ipynb```) for the Counterfactual CBM.

However, you can run all the experiments of our paper executing the following: 

```
wandb sweep --project CFCBM ./config/cfcbm_sweep.yaml
```

The previous command gives you the ```[AGENT_ID]``` and the full command to run the wandb agent. It is similar to the following one:
```
wandb agent [WANDB_ID]/CFCBM/[AGENT_ID]
```

