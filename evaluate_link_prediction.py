import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.CrossDyG import CrossDyG
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
# from evaluate_models_utils import evaluate_model_link_prediction
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
# from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

from tqdm import tqdm
from utils.utils import get_link_prediction_metrics
from utils.utils import NegativeEdgeSampler, NeighborSampler
from torch.utils.data import DataLoader
from utils.DataLoader import Data

def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['TGAT', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'CrossDyG']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            elif model_name in ['CrossDyG']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=True)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, test_data = get_link_prediction_data(dataset_name=args.test_dataset_name)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=test_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=test_data.src_node_ids, dst_node_ids=test_data.dst_node_ids, seed=0)

    # get data loaders
    # np.random.seed(0)
    # select_list = np.random.choice(np.arange(len(test_data.src_node_ids)), int(1.0*len(test_data.src_node_ids)), replace=False).tolist()
    # test_idx_data_loader = get_idx_data_loader(indices_list=select_list, batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    if True:
        val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

        for run in range(args.num_runs):

            set_random_seed(seed=run) # this will not influence results

            args.seed = run
            args.train_model_name = f'{args.model_name}_seed{args.seed}'
            args.save_result_name = f'test_{args.model_name}_loadepoch_{args.load_epoch}_seed{args.seed}'

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            os.makedirs(f"./logs/{args.model_name}/{args.train_dataset_name}/{args.test_dataset_name}/{args.save_result_name}/", exist_ok=True)
            # create file handler that logs debug and higher level messages
            fh = logging.FileHandler(f"./logs/{args.model_name}/{args.train_dataset_name}/{args.test_dataset_name}/{args.save_result_name}/{str(time.time())}.log")
            fh.setLevel(logging.DEBUG)
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # add the handlers to logger
            logger.addHandler(fh)
            logger.addHandler(ch)

            run_start_time = time.time()
            logger.info(f"********** Run {run + 1} starts. **********")

            logger.info(f'configuration is {args}')

            # create model
            if args.model_name == 'TGAT':
                dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
            elif args.model_name == 'CAWN':
                dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                        num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
            elif args.model_name == 'TCL':
                dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                       num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
            elif args.model_name == 'GraphMixer':
                dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                              time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
            elif args.model_name == 'DyGFormer':
                dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                             time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                                             num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                             max_input_sequence_length=args.max_input_sequence_length, device=args.device)
            elif args.model_name == 'CrossDyG':
                dynamic_backbone = CrossDyG(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                                            num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                            max_input_sequence_length=args.max_input_sequence_length, gamma=args.gamma, device=args.device)
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                        hidden_dim=node_raw_features.shape[1], output_dim=1)
            model = nn.Sequential(dynamic_backbone, link_predictor)
            # logger.info(f'model -> {model}')
            logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

            # load the saved model
            load_model_folder = f"./saved_models/{args.model_name}/{args.train_dataset_name}/{args.train_model_name}"
            load_model_path = os.path.join(load_model_folder, f"{args.model_name}_{args.load_epoch}.pkl")
            model.load_state_dict(torch.load(load_model_path, map_location='cpu'))

            model = convert_to_gpu(model, device=args.device)

            loss_func = nn.BCELoss()

            # evaluate the best model
            logger.info(f'get final performance on dataset {args.test_dataset_name}...')

            test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                       model=model,
                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                       evaluate_idx_data_loader=test_idx_data_loader,
                                                                       evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                       evaluate_data=test_data,
                                                                       loss_func=loss_func,
                                                                       num_neighbors=args.num_neighbors,
                                                                       time_gap=args.time_gap)

            # store the evaluation metrics at the current run
            test_metric_dict = {}

            logger.info(f'test loss: {np.mean(test_losses):.4f}')
            for metric_name in test_metrics[0].keys():
                average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                logger.info(f'test {metric_name}, {average_test_metric:.4f}')
                test_metric_dict[metric_name] = average_test_metric

            single_run_time = time.time() - run_start_time
            logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

            test_metric_all_runs.append(test_metric_dict)

            # avoid the overlap of logs
            if run < args.num_runs - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)

            # save model result
            result_json = {"test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}}
            result_json = json.dumps(result_json, indent=4)

            save_result_folder = f"./saved_results/{args.model_name}/{args.train_dataset_name}/{args.test_dataset_name}"
            os.makedirs(save_result_folder, exist_ok=True)
            save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
            with open(save_result_path, 'w') as file:
                file.write(result_json)
            logger.info(f'save negative sampling results at {save_result_path}')

        # store the average metrics at the log of the last run
        logger.info(f'metrics over {args.num_runs} runs:')

        for metric_name in test_metric_all_runs[0].keys():
            logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
