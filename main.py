import argparse
import copy
import os
import time

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

# Definitions of bandit algorithms
from algorithms.epsilon_greedy.annealing import *
from algorithms.epsilon_greedy.standard import *
from algorithms.exp3.exp3 import *
from algorithms.hedge.hedge import *
from algorithms.softmax.annealing import *
from algorithms.softmax.standard import *
from algorithms.ucb.ucb1 import *
from util import *


# bandit params
# epsilon greedy
epsilon = 0.9  # epsilon = (0,1)
# exp3
exp3_gamma = 0.2  # in [0.1, 0.2, 0.3, 0.4, 0.5]
# Softmax
temperature = 0.9
# Hedge
eta = 0.9  # in [.5, .8, .9, 1, 2]

algo_dict = {"UCB1": UCB1([], []),
             "EpsilonGreedy": EpsilonGreedy(epsilon, [], []),
             "AnnealingEpsilonGreedy": AnnealingEpsilonGreedy([], []),
             "Softmax": Softmax(temperature, [], []),
             "AnnealingSoftmax": AnnealingSoftmax([], []),
             "Hedge": Hedge(eta, [], []),
             "Exp3": Exp3(exp3_gamma, []),
             }

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task",
        default='mnli',
        type=str,
        help="GLUE task to finetune",
    )
    parser.add_argument(
        "--model_checkpoint",
        default="typeform/distilbert-base-uncased-mnli",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models, Default: typeform/distilbert-base-uncased-mnli",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        required=True,
        help="Random test batch size",
    )
    parser.add_argument(
        "--data_dir",
        default="data/mnli",
        type=str,
        help="Data directory",
    )
    parser.add_argument(
        "--pruning_type",
        default=None,
        type=str,
        required=True,
        help="Pruning type: one of ['head', 'head_whole', 'head_whole_nothres', 'head_whole_ablation', 'neuron']",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Whether to overwrite data in output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )

    parser.add_argument(
        "--try_masking", action="store_true", help="Whether to try to mask head until a threshold of accuracy."
    )
    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default="acc", type=str, help="Metric to use for head masking.")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, sequences shorter padded.",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    args = parser.parse_args()

    if args.pruning_type not in ['head', 'head_whole', 'head_whole_nothres', 'head_whole_ablation', 'neuron']:
        raise Exception("Invalid pruning type")

    # load dataset
    encoded_dataset = load_from_disk(args.data_dir)

    # load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint)

    num_labels = 3 if args.task.startswith("mnli") else 1 if args.task == "stsb" else 2  # 3
    metric_name = "pearson" if args.task == "stsb" else "matthews_correlation" if args.task == "cola" else "accuracy"
    model_name = args.model_checkpoint.split("/")[-1]  # 'distilbert-base-uncased-mnli'
    validation_key = "validation_mismatched" if args.task == "mnli-mm" else "validation_matched" if args.task == "mnli" else "validation"

    # inference the full model
    args = TrainingArguments(
        model_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        metric_for_best_model=metric_name,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
    )

    full_mean_loss, full_accuracy = test_full(model, trainer, metric_name)
    full_weights_count = count_weights(model)
    print('full_mean_loss:', full_mean_loss)
    print('full_accuracy:', full_accuracy)
    print('full_weights_count:', full_weights_count)

    # backup model
    model_copy = copy.deepcopy(model)

    # model configs
    n_layers, n_heads, d_model = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size
    d_k = d_model // n_heads
    print('n_layers:', n_layers)
    print('n_heads:', n_heads)
    print('d_model:', d_model)
    print('d_k:', d_k)

    # bandit configs
    n_arms = n_heads if args.pruning_type == 'head' else n_layers * n_heads if 'headwhole' in args.pruning_type else None
    arms = np.arange(n_arms)

    # hyperparameters
    rounds = 2000  # Playing times
    # thres = 0.2 # maximum reward
    algo_name = "Softmax"

    # choose algo
    algo = algo_dict[algo_name]
    algo.initialize(n_arms)
    prng = np.random.RandomState(2021)
    start_time = time.time()

    # bandit playing
    for t in tqdm(range(rounds)):
        print('============ {}th iteration =============='.format(t))
        model = copy.deepcopy(model_copy)  # restore full model
        chosen_arm = algo.select_arm()  # select arm (head) to prune

        # whole layers
        block_num = chosen_arm // n_heads
        head_num = chosen_arm % n_heads
        print('chosen_arm: {}th block {}th head'.format(block_num, head_num))

        # prune head
        zeroize_head_weights(model, block_num, head_num)
        pruned_loss, pruned_accuracy = test_pruned(model, trainer, metric_name, prng)
        print('pruned_accuracy:', pruned_accuracy)
        print('pruned_loss:', pruned_loss)
        print()

        delta_accuracy = pruned_accuracy - full_accuracy
        delta_loss = pruned_loss - full_mean_loss
        print("delta_accuracy:", delta_accuracy)
        print("delta_loss:", delta_loss)

        # ver1 prune by accuracy # TBD
        """
        if accuracy increases after pruning (delta > 0), reward = min(thres + delta
        if accuracy decreases after pruning (delta < 0), reward = max(0, thres + delta)
        """
        reward = delta_accuracy
        #     reward = max(0, thres + delta_accuracy)
        print("reward:", reward)

        index = chosen_arm

        algo.update(chosen_arm, reward)
        print()

    # measure playing time
    print("The time for running {} algorithm is {} seconds ".format(algo_name, time.time() - start_time))

    # pruning
    print('Start pruning')
    expected_rewards = copy.deepcopy(algo.weights) if algo_name == 'Exp3' else copy.deepcopy(algo.values)
    model = copy.deepcopy(model_copy)  # restore full model

    prune_order = []
    prune_after_accuracy = np.zeros(n_arms)  # AccuracyAftrerPrune

    for t in tqdm(range(n_arms)):  # for whole heads
        prune_idx = np.argmax(expected_rewards)
        prune_order.append(prune_idx)

        block_num = prune_idx // n_heads
        head_num = prune_idx % n_heads
        print('Now prune {}th block {}th head -- Remaining heads: {}'.format(block_num, head_num, n_arms - t - 1))

        expected_rewards[prune_idx] = -100

        zeroize_head_weights(model, block_num, head_num)
        pruned_loss, pruned_accuracy = test_full(model, trainer, metric_name)

        prune_after_accuracy[t] = pruned_accuracy
        print("Accuracy after pruning = {:.2f}".format(pruned_accuracy * 100))
        print("Remaining parameters = {:.2f}%".format(count_nonzero_weights(model) * 100 / full_weights_count))
        print()

    # save data
    result_path = os.path.join('./results', args.pruning_type, algo_name)
    print(result_path)

    # Create a new directory if it does not exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print("The new directory is created!")

    if algo_name == 'Exp3':
        np.save(os.path.join(result_path, "expected_rewards.npy"), algo.weights)
        np.save(os.path.join(result_path, "prune_order.npy"), prune_order)
        np.save(os.path.join(result_path, "prune_after_accuracy.npy"), prune_after_accuracy)
    else:
        np.save(os.path.join(result_path, "counts.npy"), algo.counts)
        np.save(os.path.join(result_path, "expected_rewards.npy"), algo.values)
        np.save(os.path.join(result_path, "prune_order.npy"), prune_order)
        np.save(os.path.join(result_path, "prune_after_accuracy.npy"), prune_after_accuracy)

if __name__ == "__main__":
    main()
