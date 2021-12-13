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

    parser.add_argument(
        "--rounds",
        default=None,
        type=int,
        required=True,
        help="How many rounds to play MAB algorithm",
    )

    parser.add_argument(
        "--algo_name",
        default=None,
        type=str,
        required=True,
        help="MAB algorithm type / Random",
    )

    parser.add_argument(
        "--thres",
        default=None,
        type=float,
        help="Threshold of accuracy drop",
    )

    parser.add_argument(
        "--block_num",
        default=None,
        type=int,
        help="Which transformer block to prune",
    )

    parser.add_argument("--seed", type=int, default=2021)

    args = parser.parse_args()

    # assert
    if args.pruning_type not in ['head', 'head_whole', 'head_whole_nothres', 'head_whole_ablation', 'neuron']:
        raise Exception("Invalid pruning type")
    if args.algo_name not in algo_dict.keys() and not 'random':
        raise Exception("Invalid algo name")
    if args.thres is None and args.pruning_type is not 'head_whole_nothres':
        raise Exception("'thres' necessary for this pruning type")
    if args.block_num is None and args.pruning_type in ['head', 'neuron']:
        raise Exception("'block_num' necessary for this pruning type")

    pruning_type = 'head' if args.pruning_type == 'head' \
        else 'headwhole' if 'headwhole' in args.pruning_type \
        else 'neuron'

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
    hidden_dim = model.config.hidden_dim
    d_k = d_model // n_heads
    print('n_layers:', n_layers)
    print('n_heads:', n_heads)
    print('d_model:', d_model)
    print('hidden_dim:', hidden_dim)
    print('d_k:', d_k)

    # bandit configs
    n_arms = n_heads if pruning_type == 'head' \
        else n_layers * n_heads if pruning_type == 'headwhole' \
        else hidden_dim

    # hyperparameters
    rounds = args.rounds  # Playing times
    thres = args.thres # maximum reward
    algo_name = args.algo_name
    block_num = args.block_num

    # choose algo
    algo = algo_dict[algo_name]
    algo.initialize(n_arms)
    prng = np.random.RandomState(args.seed)
    start_time = time.time()

    # bandit playing

    if args.algo_name == 'random':
        prune_order = np.arange(n_arms)
        np.random.shuffle(prune_order)

        print('Start random pruning')
        model = copy.deepcopy(model_copy)  # restore full model
        prune_after_accuracy = np.zeros(n_arms)

        for t in tqdm(range(n_arms)):  # for whole heads
            prune_idx = prune_order[t]  # x=np.argmax(rewards)
            if pruning_type == 'head':
                # prune head
                print('Remaining heads: {} -- Now prune {}th head'.format(n_heads - t - 1, prune_idx))
                zeroize_head_weights(model, block_num, prune_idx)

            elif pruning_type == 'headwhole':  # whole layers
                block_num = prune_idx // n_heads
                head_num = prune_idx % n_heads
                print('Now prune {}th block {}th head -- Remaining heads: {}'.format(block_num, head_num, n_arms - t - 1))
                zeroize_head_weights(model, block_num, head_num)

            else:
                print('Remaining neurons: {} -- Now prune {}th neuron'.format(n_arms - t - 1, prune_idx))
                zeroize_neuron_weights(model, block_num, prune_idx)

            pruned_loss, pruned_accuracy = test_full(model, trainer, metric_name)

            prune_after_accuracy[t] = pruned_accuracy
            print("Accuracy after pruning = {:.2f}".format(pruned_accuracy * 100))
            print("Remaining parameters = {:.2f}%".format(count_nonzero_weights(model) * 100 / full_weights_count))
            print()

    else: # MAB
        print("===========MAB Algorithm : {}===========".format(algo_name))

        for t in tqdm(range(rounds)):
            print('============ {}th iteration =============='.format(t))
            model = copy.deepcopy(model_copy)  # restore full model
            chosen_arm = algo.select_arm()  # select arm (head) to prune

            if pruning_type == 'head':
                # prune head
                zeroize_head_weights(model, block_num, chosen_arm)

            elif pruning_type == 'headwhole': # whole layers
                block_num = chosen_arm // n_heads
                head_num = chosen_arm % n_heads
                print('chosen_arm: {}th block {}th head'.format(block_num, head_num))
                # prune head
                zeroize_head_weights(model, block_num, head_num)

            else:
                zeroize_neuron_weights(model, block_num, chosen_arm)

            pruned_loss, pruned_accuracy = test_pruned(model, trainer, metric_name, prng)
            print('pruned_accuracy:', pruned_accuracy)
            print('pruned_loss:', pruned_loss)
            print()

            delta_accuracy = pruned_accuracy - full_accuracy
            delta_loss = pruned_loss - full_mean_loss
            print("delta_accuracy:", delta_accuracy)
            print("delta_loss:", delta_loss)

            # ver1 prune by accuracy
            """
            if accuracy increases after pruning (delta > 0), reward = min(thres + delta
            if accuracy decreases after pruning (delta < 0), reward = max(0, thres + delta)
            """
            if args.pruning_type == "headwhole_nothres":
                reward = delta_accuracy
            else:
                reward = max(0, thres + delta_accuracy)
            print("reward:", reward)

            algo.update(chosen_arm, reward)
            print()

        # measure playing time
        print("The time for running {} algorithm is {} seconds ".format(algo_name, time.time() - start_time))

        # pruning
        print('Start pruning')
        expected_rewards = copy.deepcopy(algo.weights) if algo_name == 'Exp3' else copy.deepcopy(algo.values)
        model = copy.deepcopy(model_copy)  # restore full model

        prune_order = []
        prune_after_accuracy = np.zeros(n_arms)

        for t in tqdm(range(n_arms)):  # for whole heads
            prune_idx = np.argmax(expected_rewards)
            prune_order.append(prune_idx)

            expected_rewards[prune_idx] = -100

            if pruning_type == 'head':
                # prune head
                print('Remaining heads: {} -- Now prune {}th head'.format(n_heads - t - 1, prune_idx))
                zeroize_head_weights(model, block_num, prune_idx)

            elif pruning_type == 'headwhole':  # whole layers
                block_num = prune_idx // n_heads
                head_num = prune_idx % n_heads
                print('Now prune {}th block {}th head -- Remaining heads: {}'.format(block_num, head_num, n_arms - t - 1))
                zeroize_head_weights(model, block_num, head_num)

            else:
                print('Remaining neurons: {} -- Now prune {}th neuron'.format(n_arms - t - 1, prune_idx))
                zeroize_neuron_weights(model, block_num, prune_idx)

            pruned_loss, pruned_accuracy = test_full(model, trainer, metric_name)

            prune_after_accuracy[t] = pruned_accuracy
            print("Accuracy after pruning = {:.2f}".format(pruned_accuracy * 100))
            print("Remaining parameters = {:.2f}%".format(count_nonzero_weights(model) * 100 / full_weights_count))
            print()

    # save data
    result_path = os.path.join('./results', args.pruning_type, algo_name) if pruning_type == 'headwhole' \
        else os.path.join('./results', pruning_type, str(block_num), algo_name)
    print('result_path:', result_path)

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
