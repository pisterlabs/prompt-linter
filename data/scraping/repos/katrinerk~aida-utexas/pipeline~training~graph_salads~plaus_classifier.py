import dill
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import json
from copy import deepcopy
from pipeline.training.graph_salads.modules import CoherenceNetWithGCN
import random
from tqdm import tqdm
from collections import defaultdict
from pipeline.training.graph_salads.gen_single_doc_graphs import read_graph
from pipeline.training.graph_salads.index_salads import convert_labels_to_indices

# Make a dir (if it doesn't already exist)
def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class DataIterator:
    def __init__(self, index_data_dir, data_dir):
        self.index_data_dir = index_data_dir
        self.file_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
        self.cursor = 0
        self.epoch = 0
        self.shuffle()
        self.size = len(self.file_paths)

    def shuffle(self):
        self.file_paths = sorted(self.file_paths)
        random.seed(self.epoch)
        random.shuffle(self.file_paths)

    def next_batch(self):
        if self.cursor >= self.size:
            self.cursor = 0

        # Data comes in the form of a "graph_dict" dictionary which contains the following key-value pairs:
        # 'graph_mix'    -     --> pickled graph salad object
        # 'ere_mat_ind'        --> Indexer object which maps ERE IDs to indices in an adjacency matrix
        # 'stmt_mat_ind'       --> Indexer object which maps stmt IDs to indices in an adjacency matrix
        # 'adj_head'           --> (num_eres x num_stmts) adjacency matrix;
        #                          contains a 1 in position (x, y) if statement y is attached to ERE x at the head/subject (does NOT include typing statements)
        # 'adj_tail'           --> same as above, but for statements attached to EREs at the tail
        # 'adj_type'           --> same as above, but for typing statements attached to EREs
        # 'ere_labels'         --> lists of indices for labels associated with EREs
        # 'stmt_labels'        --> lists of indices for labels associated with stmts
        # 'num_word2vec_ere'   --> number of ERE names identified in the Word2Vec vocabulary used
        # 'num_word2vec_stmts' --> number of stmt labels identified in the Word2Vec vocabulary used
        # 'origin_id'          --> query merge point
        # 'noisy_merge_points' --> set containing all noisy merge points
        # 'target_graph_id'    --> target graph ID
        # 'query_stmts'        --> list of query stmt indices
        # 'query_eres'         --> list of query ERE indices
        # 'plaus_clusters'     --> list of query sets for plausibility classifier

        data_group = self.file_paths[self.cursor].split('/')[-2]

        if 'Train' in data_group:
            data_group = 'Train'
        elif 'Val' in data_group:
            data_group = 'Val'
        elif 'Test' in data_group:
            data_group = 'Test'

        plaus_dict = dill.load(open(self.file_paths[self.cursor], "rb"))
        graph_dict = dill.load(open(os.path.join(self.index_data_dir, data_group, self.file_paths[self.cursor].split('/')[-1]), "rb"))

        pos_samples = [item for sublist in plaus_dict['pos_samples'].values() for item in sublist]
        neg_samples = [item for sublist in plaus_dict['neg_samples'].values() for item in sublist]

        for iter in range(len(pos_samples)):
            pos_samples[iter] = (1, [graph_dict['stmt_mat_ind'].get_index(stmt_id, add=False) for stmt_id in pos_samples[iter]])

        for iter in range(len(neg_samples)):
            neg_samples[iter] = (0, [graph_dict['stmt_mat_ind'].get_index(stmt_id, add=False) for stmt_id in neg_samples[iter]])

        random.seed(self.epoch)
        random.shuffle(pos_samples)
        random.seed(self.epoch)
        random.shuffle(neg_samples)

        plaus_clusters_with_labels = [item for sublist in list(zip(pos_samples, neg_samples)) for item in sublist]
        graph_dict['plaus_clusters'] = [item[1] for item in plaus_clusters_with_labels]
        graph_dict['plaus_labels'] = [item[0] for item in plaus_clusters_with_labels]

        if self.cursor == self.size - 1:
            self.epoch += 1

        self.cursor += 1

        return graph_dict

# Run the model on a graph salad
def run_classifier(model, optimizer, graph_dict, loss_func, data_group, device):
    plaus_cluster_stmt_ind_sets = graph_dict['plaus_clusters']
    plaus_cluster_stmt_name_sets = []

    for item in plaus_cluster_stmt_ind_sets:
        plaus_cluster_stmt_name_sets.append({graph_dict['stmt_mat_ind'].get_word(sub_item) for sub_item in item})

    graph_mix = graph_dict['graph_mix']

    plaus_cluster_ere_ind_sets = []

    for cluster_stmts in plaus_cluster_stmt_name_sets:
        temp = set.union(*[{graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id} for stmt_id in cluster_stmts if graph_mix.stmts[stmt_id].tail_id])
        temp = [graph_dict['ere_mat_ind'].get_index(item, add=False) for item in temp]

        plaus_cluster_ere_ind_sets.append(temp)

    plaus_labels = graph_dict['plaus_labels']

    num_correct = 0
    predictions, actual_classes = [], []

    for i in range(len(plaus_cluster_stmt_ind_sets)):
        graph_dict['query_stmts'] = plaus_cluster_stmt_ind_sets[i]
        graph_dict['query_eres'] = plaus_cluster_ere_ind_sets[i]

        if i == 0:
            pred_logit, gcn_embeds = model(graph_dict, None, device)
        else:
            pred_logit, _ = model(graph_dict, gcn_embeds, device)

        correct = int(torch.round(torch.sigmoid(pred_logit)).to(dtype=torch.float).item() == plaus_labels[i])
        num_correct += correct
        actual_class = torch.tensor([plaus_labels[i]], dtype=torch.float, device=device)

        predictions.append(pred_logit)
        actual_classes.append(actual_class)

    # Don't need to divide by batch size; this is the default behavior of nn.BCELossWithLogits
    loss = loss_func(torch.cat(predictions, dim=0), torch.cat(actual_classes, dim=0))
    loss_to_report = loss.item()

    if data_group == 'Train':
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Accuracy calculation
    accuracy = float(num_correct) / len(plaus_cluster_stmt_ind_sets)

    return loss_to_report, accuracy

def train(weight_decay, index_data_dir, train_path, valid_path, test_path, indexer_info_dict, attention_type, use_attender_vectors, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout, num_epochs,
    learning_rate, save_path, load_path, load_optim, valid_every, print_every, device):
    model = CoherenceNetWithGCN(True, indexer_info_dict, attention_type, use_attender_vectors, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout).to(device)
    loss_func = nn.BCEWithLogitsLoss()

    # Instantiate Adam optimizer with specified weight decay (if any)
    if weight_decay:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    optimizer.zero_grad()

    start = time.time()

    current_epoch = 0

    best_loss = np.inf

    # Create DataIterator object to manage data
    train_iter = DataIterator(index_data_dir, train_path)

    while train_iter.epoch < num_epochs:
        step = 0
        print("\nEpoch %d\n" % (current_epoch + 1))
        model.train()
        train_losses, train_accuracies, train_diffs = [], [], []

        while train_iter.epoch <= current_epoch:
            # "Artificial" batching; allow gradients to accumulate and only backprop after every <batch_size> graph salads
            train_loss, train_accuracy = run_classifier(model, optimizer, train_iter.next_batch(), loss_func, 'Train', device)

            step += 1
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Print output report after every <print_every> graph salads
            if (step % print_every == 0) or (step == train_iter.size):
                print("At step %d: loss = %.6f | accuracy = %.4f (%.2fs)." % (step, np.mean(train_losses), np.mean(train_accuracies), time.time() - start))
                start = time.time()

            # Validate the model on the validation set after every <valid_every> salads
            if (step % valid_every == 0) or (step == train_iter.size):
                print("\nRunning valid ...\n")
                model.eval()
                average_valid_loss, average_valid_accuracy = run_no_backprop(index_data_dir, valid_path, model, loss_func)
                print("Valid (avg): loss = %.6f | accuracy = %.4f" % (average_valid_loss, average_valid_accuracy))

                # Save checkpoint if the reported loss is lower than all previous reported losses
                if average_valid_loss < best_loss:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(save_path, 'gcn2-cuda_best' + '_' + str(step) + '_' + str(current_epoch) + '.ckpt'))
                    best_loss = average_valid_loss
                    print('New best val checkpoint at step ' + str(step) + ' of epoch ' + str(current_epoch + 1))

                model.train()

        current_epoch += 1

    # Run the model on the test set
    model.eval()
    average_test_loss, average_test_accuracy = run_no_backprop(index_data_dir, test_path, model, loss_func)
    print("Test (avg): loss = %.6f | accuracy = %.4f" % (average_test_loss, average_test_accuracy))
    model.train()

# Run the model on a set of data for evaluative purposes (called when validating and testing only)
def run_no_backprop(index_data_dir, data_path, model, loss_func):
    valid_iter = DataIterator(index_data_dir, data_path)

    valid_losses, valid_accuracies = [], []

    while valid_iter.epoch == 0:
        valid_loss, valid_accuracy = run_classifier(model, None, valid_iter.next_batch(), loss_func, 'Val', device)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

    average_valid_loss = np.mean(valid_losses)
    average_valid_accuracy = np.mean(valid_accuracies)

    return (average_valid_loss, average_valid_accuracy)

def eval_plaus(indexer_info_file, model_path, kb_path, list_of_clusters, attention_type='concat', use_attender_vectors=False, num_layers=2, hidden_size=300, attention_size=300, device='cpu'):
    json_obj = json.load(open(kb_path, 'r'))['theGraph']
    graph = read_graph('', json_obj, False)

    import sys
    from pipeline.training.graph_salads import gen_single_doc_graphs

    sys.modules['gen_single_doc_graphs'] = gen_single_doc_graphs

    indexer_info = dill.load(open(indexer_info_file, 'rb'))

    graph_dict = convert_labels_to_indices(graph, indexer_info)

    indexer_info_dict = dict()
    ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt = indexer_info
    indexer_info_dict['ere_indexer'] = ere_indexer
    indexer_info_dict['stmt_indexer'] = stmt_indexer
    indexer_info_dict['ere_emb_mat'] = ere_emb_mat
    indexer_info_dict['stmt_emb_mat'] = stmt_emb_mat
    indexer_info_dict['num_word2vec_ere'] = num_word2vec_ere
    indexer_info_dict['num_word2vec_stmt'] = num_word2vec_stmt

    model = CoherenceNetWithGCN(True, indexer_info_dict, attention_type, use_attender_vectors, num_layers, hidden_size, attention_size, 0, 0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])

    model.eval()

    preds = []
    plaus_cluster_stmt_ind_sets = []

    for item in list_of_clusters:
        plaus_cluster_stmt_ind_sets.append([graph_dict['stmt_mat_ind'].get_index(stmt_id, add=False) for stmt_id in item])

    graph_mix = graph_dict['graph_mix']

    plaus_cluster_ere_ind_sets = []

    for cluster_stmts in list_of_clusters:
        temp = set.union(*[{graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id} for stmt_id in cluster_stmts if graph_mix.stmts[stmt_id].tail_id])
        temp = [graph_dict['ere_mat_ind'].get_index(item, add=False) for item in temp]

        plaus_cluster_ere_ind_sets.append(temp)

    for i in range(len(plaus_cluster_stmt_ind_sets)):
        graph_dict['query_stmts'] = plaus_cluster_stmt_ind_sets[i]
        graph_dict['query_eres'] = plaus_cluster_ere_ind_sets[i]

        if i == 0:
            pred_logit, gcn_embeds = model(graph_dict, None, device)
        else:
            pred_logit, _ = model(graph_dict, gcn_embeds, device)

        preds.append(torch.sigmoid(pred_logit).item())

    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="Mode to run script in (can be either train or validate)")
    parser.add_argument("--data_dir", type=str, default="/home/atomko/Plaus_50k_Data",
                        help="Data directory")
    parser.add_argument("--index_data_dir", type=str, default="/home/atomko/M36_50k_Indexed",
                        help="Data directory")
    parser.add_argument("--train_dir", type=str, default="Train",
                        help="Name of subdir under data directory containing training mixtures")
    parser.add_argument("--valid_dir", type=str, default="Val",
                        help="Name of subdir under data directory containing validation mixtures")
    parser.add_argument("--test_dir", type=str, default="Test",
                        help="Name of subdir under data directory containing test mixtures")
    parser.add_argument("--indexer_info_file", type=str, default="/home/atomko/M36_50k_Indexed/indexers.p",
                        help="Indexer info file (contains word-to-index maps for ERE and statement names, contains top k most frequent Word2Vec embeds in training set)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers in the GCN")
    parser.add_argument("--attention_type", type=str, default='concat',
                        help="Attention score type for inference (can be concat or bilinear)")
    parser.add_argument("--use_attender_vectors", action='store_true',
                        help="Use both the attender and context vectors when calculating attention vector for plausibility classifier")
    parser.add_argument("--hidden_size", type=int, default=300,
                        help="Size of hidden representations in model")
    parser.add_argument("--attention_size", type=int, default=300,
                        help="Size of attention vectors in model")
    parser.add_argument("--conv_dropout", type=float, default=.5,
                        help="Dropout rate for convolution layers")
    parser.add_argument("--attention_dropout", type=float, default=.3,
                        help="Dropout rate for attention layers")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of epochs for training")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay for Adam optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--save_path", type=str, default="SavedModels_Test",
                        help="Directory to which model/optimizer checkpoints will be saved")
    parser.add_argument("--load_path", type=str, default=None,
                        help="File path for pretrained model/optimizer")
    parser.add_argument("--load_optim", action='store_true',
                        help="Load optimizer state (if <load_path> is also supplied)")
    parser.add_argument("--valid_every", type=int, default=15000,
                        help="Model runs on the validation set every <valid_every> training instances")
    parser.add_argument("--print_every", type=int, default=100,
                        help="Precision and loss reports are generated every <print_every> training instances")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device number (or 'cpu' for cpu)")

    args = parser.parse_args()
    locals().update(vars(args))
    load_path = None if load_path == "" else load_path
    device = torch.device(("cuda:" + device) if torch.cuda.is_available() and device != "cpu" else "cpu")

    print("Config:\n", args, "\nDevice:\n", device, "\n")
    print("START TRAINING ...\n\n")

    # Check that the directory for saving model checkpoints exists
    verify_dir(save_path)

    # Set seeds for debugging/hyperparam comparison
    torch.manual_seed(0)
    np.random.seed(0)

    train_path = os.path.join(data_dir, train_dir)
    valid_path = os.path.join(data_dir, valid_dir)
    test_path = os.path.join(data_dir, test_dir)

    # Load indexer info
    indexer_info_dict = dict()
    ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt = dill.load(open(indexer_info_file, 'rb'))
    indexer_info_dict['ere_indexer'] = ere_indexer
    indexer_info_dict['stmt_indexer'] = stmt_indexer
    indexer_info_dict['ere_emb_mat'] = ere_emb_mat
    indexer_info_dict['stmt_emb_mat'] = stmt_emb_mat
    indexer_info_dict['num_word2vec_ere'] = num_word2vec_ere
    indexer_info_dict['num_word2vec_stmt'] = num_word2vec_stmt

    # Train a fresh model
    if mode == 'train':
        train(weight_decay, index_data_dir, train_path, valid_path, test_path, indexer_info_dict, attention_type, use_attender_vectors, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout, num_epochs,
                      learning_rate, save_path, load_path, load_optim, valid_every, print_every, device)
