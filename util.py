import torch
import numpy as np
from tqdm.notebook import tqdm
from datasets import load_metric


# test utils
def test_full(model, trainer, metric_name):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loss_list = []
    eval_dataloader = trainer.get_eval_dataloader()
    metric = load_metric(metric_name)
    model.eval()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss_list.append(outputs.loss.item())
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    mean_loss = np.mean(loss_list)
    accuracy = metric.compute()[metric_name]

    return mean_loss, accuracy


def test_pruned(model, trainer, metric_name, prng):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # shuffle eval dataset
    seed = prng.randint(10000)
    # print("seed:", seed) # should be different
    trainer.eval_dataset = trainer.eval_dataset.shuffle(seed=seed)
    eval_dataloader = trainer.get_eval_dataloader()

    metric = load_metric(metric_name)
    model.eval()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        loss = outputs.loss.item()

        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        metric = metric.compute()[metric_name]

        return loss, metric


# zerioize utils
def zeroize_head_weights(model, block_num, prune_head_idx):
    # model configs
    n_layers, n_heads, d_model = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size
    d_k = d_model // n_heads

    block = model.distilbert.transformer.layer[block_num]
    prune_idx = range(prune_head_idx * d_k, (prune_head_idx + 1) * d_k)

    with torch.no_grad():
        block.attention.q_lin.weight[prune_idx] = 0
        block.attention.q_lin.bias[prune_idx] = 0

        block.attention.k_lin.weight[prune_idx] = 0
        block.attention.k_lin.bias[prune_idx] = 0

        block.attention.v_lin.weight[prune_idx] = 0
        block.attention.v_lin.bias[prune_idx] = 0

        block.attention.out_lin.weight[:, prune_idx] = 0

        assert (
            torch.isclose(block.attention.q_lin.weight[prune_idx].cpu(), torch.zeros(len(prune_idx), d_model)).all())
        assert (
            torch.isclose(block.attention.k_lin.weight[prune_idx].cpu(), torch.zeros(len(prune_idx), d_model)).all())
        assert (
            torch.isclose(block.attention.v_lin.weight[prune_idx].cpu(), torch.zeros(len(prune_idx), d_model)).all())
        assert (torch.isclose(block.attention.out_lin.weight[:, prune_idx].cpu(),
                              torch.zeros(d_model, len(prune_idx))).all())

    return model


def zeroize_neuron_weights(model, block_num, prune_neuron_idx):
    block = model.distilbert.transformer.layer[block_num]

    with torch.no_grad():
        block.ffn.lin1.weight[prune_neuron_idx] = 0
        block.ffn.lin1.bias[prune_neuron_idx] = 0
        block.ffn.lin2.weight[:, prune_neuron_idx] = 0

    return model


# compute number of weights
def count_weights(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())


# compute number of nonzero weights
def count_nonzero_weights(model):
    non_zeros = 0
    for param in model.parameters():
        if param is not None:
            non_zeros += param.count_nonzero()
    return non_zeros.item()