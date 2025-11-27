import argparse
import gc
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from sklearn import preprocessing
from torch import utils, nn, optim

from model.models import STeGEformer, STeGE
from script import dataloader, utility, earlystopping, visualisation_scatter

import statistics


def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_parameters():
    parser = argparse.ArgumentParser(description='Train STeGEformer.')
    parser.add_argument('--dataset', type=str, default='pems-bay', choices=['pemsd7-m', 'pems-bay'],
                        help='Dataset to use')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='Enable CUDA')
    args = parser.parse_args()

    dataset_name = args.dataset
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        gc.collect()

    # Load Configs
    config_dir = Path('config')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    model_config = yaml.safe_load((config_dir / f'model_config.yml').open('r'))
    local_config = yaml.safe_load((config_dir / f'{dataset_name}_config.yml').open('r'))

    data_dir = global_config['data_dir']
    results_dir = global_config['results_dir']
    visualisation_dir = global_config['visualisation_dir']
    seeds = global_config['seeds']

    train_config = model_config['train_config']

    data_config = local_config['data_config']
    vis_config = local_config['vis_config']
    info_config = local_config['info_config']

    # Calculate Blocks
    n_history = train_config['n_history']
    n_prediction = train_config['n_prediction']
    temporal_kernel_size = train_config['temporal_kernel_size']
    use_explanation = train_config['use_explanation']
    use_lgt_encoder = train_config['use_LGTEncoder']

    if use_lgt_encoder:
        sequence_length_after_convs = n_history - (temporal_kernel_size - 1) * 2
    else:
        sequence_length_after_convs = n_history - (temporal_kernel_size - 1)

    blocks = [[1], [64, 16, 64]]
    if sequence_length_after_convs == 0:
        blocks.append([128])
    elif sequence_length_after_convs > 0:
        blocks.append([128, 128])
    blocks.append([n_prediction])

    print('=' * 50)
    print('=' * 50)
    print(f'[INFO] Running {dataset_name} with STeGEformer over random seeds ({seeds}).')
    print(f'[INFO] Using {device}.')
    print(f'[INFO] Using interpretation: {use_explanation}.')
    print(f'[INFO] Using LGTEncoder: {use_lgt_encoder}.')
    print('=' * 50)
    print("=" * 50 + "\n")

    return device, data_dir, results_dir, visualisation_dir, seeds, train_config, data_config, info_config, vis_config, blocks


def data_loader(data_dir, data_config, train_config, device):
    dataset_name = str(data_config['dataset_name'])
    dataset_path = os.path.join(data_dir, dataset_name)

    adj, num_nodes = dataloader.load_adj(dataset_path)
    edge_index = dataloader.convert_adj_to_edge_index(adj)
    gso = utility.calc_gso(adj)
    gso = gso.toarray().astype(np.float32)
    gso = torch.from_numpy(gso).to(device)

    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    data_splits = data_config['splits']
    val_and_test_rate = data_splits['val_and_test_rate']

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(data_dir, dataset_name, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    n_his, n_pred = train_config['n_history'], train_config['n_prediction']
    x_train, y_train = dataloader.data_transform(train, n_his, n_pred, device)
    x_val, y_val = dataloader.data_transform(val, n_his, n_pred, device)
    x_test, y_test = dataloader.data_transform(test, n_his, n_pred, device)

    batch_size = data_config['batch_size']
    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return num_nodes, zscore, train_iter, val_iter, test_iter, edge_index, gso


def prepare_model(train_config, data_config, info_config, blocks, num_nodes, edge_index, gso, device, seed,
                  results_dir):
    loss = nn.MSELoss()
    patience = train_config['patience']
    dataset_name = data_config['dataset_name']
    use_explanation = train_config['use_explanation']
    use_lgt_encoder = train_config['use_LGTEncoder']
    path = ""
    is_visualisation = False
    if use_lgt_encoder and use_explanation:
        model = STeGEformer(train_config, data_config, info_config, blocks, num_nodes, edge_index, gso).to(device)
        path = "STeGEformer_" + dataset_name + f"_{seed}" + ".pt"
        is_visualisation = True
    elif use_lgt_encoder and not use_explanation:
        model = STeGEformer(train_config, data_config, info_config, blocks, num_nodes, edge_index, gso).to(device)
        path = "STeGformer_" + dataset_name + f"_{seed}" + ".pt"
        is_visualisation = False
    elif not use_lgt_encoder and use_explanation:
        model = STeGE(train_config, data_config, info_config, blocks, num_nodes, edge_index, gso).to(device)
        path = "STeGE_" + dataset_name + f"_{seed}" + ".pt"
        is_visualisation = False

    early_stop = earlystopping.EarlyStopping(dataset_name=dataset_name,
                                             delta=0.0,
                                             patience=patience,
                                             verbose=True,
                                             results_dir=results_dir,
                                             path=path,
                                             )
    lr, weight_decay, step_size, gamma = train_config['lr'], train_config['weight_decay'], train_config['step_size'], \
        train_config['gamma']
    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return loss, path, early_stop, model, optimizer, scheduler, is_visualisation


def train(model, loss, optimizer, scheduler, early_stop, train_iter, val_iter, train_config, info_config, results_dir):
    epochs = train_config['epochs']
    use_explanation = train_config['use_explanation']
    info_loss_coef = info_config['info_loss_coef']
    for epoch in range(epochs):
        pred_loss_sum, info_loss_sum, loss_sum, n = 0.0, 0.0, 0.0, 0
        model.train()

        all_edge_atts = []

        #for x, y in tqdm.tqdm(train_iter, file=sys.stdout):
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred, edge_att = model(x, training=True)
            pred_loss = loss(y_pred, y)
            info_loss = torch.tensor(0.0)
            if use_explanation:
                info_loss, r = model.get_info_loss(edge_att, epoch)
                all_edge_atts.append(edge_att.detach())
            total_loss = pred_loss + info_loss_coef * info_loss

            total_loss.backward()
            optimizer.step()

            loss_sum += total_loss.item() * y.shape[0]
            n += y.shape[0]
            pred_loss_sum += pred_loss.item() * y.shape[0]
            info_loss_sum += info_loss.item() * y.shape[0]
        scheduler.step()

        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print("=" * 50)
        print(
            f'[INFO] Epoch {epoch}/{epochs} | Train Loss: {loss_sum / n:.6f} | Pred. Loss: {pred_loss_sum / n:.6f} | Info. Loss: {info_loss_sum / n:.6f} | GPU Memory Allocated: {gpu_mem_alloc:.6f} MB')
        if use_explanation:
            utility.show_edge_info(all_edge_atts, epoch, r, 'Train')

        val_loss = val(model, loss, epoch, val_iter, info_config, use_explanation)

        early_stop(val_loss, model)
        if early_stop.early_stop:
            print("[INFO] Early stopping")
            break


@torch.no_grad()
def val(model, loss, epoch, val_iter, info_config, use_explanation):
    model.eval()
    info_loss_coef = info_config['info_loss_coef']
    pred_loss_sum, info_loss_sum, loss_sum, n = 0.0, 0.0, 0.0, 0

    all_edge_atts = []

    #for x, y in tqdm.tqdm(val_iter, file=sys.stdout):
    for x, y in val_iter:
        y_pred, edge_att = model(x, training=False)
        pred_loss = loss(y_pred, y)
        info_loss = torch.tensor(0.0)
        r = 0.0
        if use_explanation:
            info_loss, r = model.get_info_loss(edge_att, epoch)
            all_edge_atts.append(edge_att.detach())
        total_loss = pred_loss + info_loss_coef * info_loss

        loss_sum += total_loss.item() * y.shape[0]
        n += y.shape[0]
        pred_loss_sum += pred_loss.item() * y.shape[0]
        info_loss_sum += info_loss.item() * y.shape[0]

    if n > 0:
        print(
            f'[INFO] Validation Loss: {loss_sum / n:.6f} | Pred. Loss: {pred_loss_sum / n:.6f} | Info. Loss: {info_loss_sum / n:.6f}')
        if use_explanation:
            utility.show_edge_info(all_edge_atts, epoch, r, 'Val')

    return torch.tensor(loss_sum / n)


@torch.no_grad()
def test(zscore, loss, path, model, test_iter, data_config, train_config, info_config, seed, results_dir):
    dataset_name = data_config['dataset_name']
    use_explanation = train_config['use_explanation']
    info_loss_coef = info_config['info_loss_coef']
    results_dir = str(os.path.join(results_dir, dataset_name))
    result_model = os.path.join(results_dir, path)
    model.load_state_dict(torch.load(result_model))  # , map_location=torch.device('cpu')
    model.eval()

    utility.evaluate_model(model, loss, test_iter, info_loss_coef, use_explanation)
    test_MAE, test_RMSE, test_MAPE = utility.evaluate_metric(model, test_iter, zscore)
    print("=" * 50)
    print(
        f'[INFO] Test MAE: {test_MAE:.6f} | Test RMSE: {test_RMSE:.6f} | Test MAPE: {test_MAPE:.6f}')
    print("=" * 50)
    return test_MAE, test_RMSE, test_MAPE, model


def vis_dataloader(data_dir, data_config, train_config, device, zscore):
    dataset_name = str(data_config['dataset_name'])
    dataset_path = os.path.join(data_dir, dataset_name)

    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    data_splits = data_config['splits']
    val_and_test_rate = data_splits['val_and_test_rate']

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(data_dir, dataset_name, len_train, len_val)
    test = zscore.transform(test)

    n_his, n_pred = train_config['n_history'], train_config['n_prediction']
    x_test, y_test = dataloader.data_transform(test, n_his, n_pred, device)

    batch_size = 32
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return test_iter


def visualise(vis_path, vis_config, test_iter, model, edge_index, num_nodes, pos):
    os.makedirs(vis_path, exist_ok=True)
    num_vis_samples = vis_config['num_vis_samples']
    vis_interval = vis_config['vis_interval']
    important_threshold = vis_config['important_threshold']

    processed_samples = 0
    for idx, (x, y) in enumerate(test_iter):
        if processed_samples >= num_vis_samples:
            break
        if idx % vis_interval != 0:
            continue

        print(f"Processing sample {processed_samples + 1}/{num_vis_samples}")

        model.eval()
        with torch.no_grad():
            y_pred, edge_att = model(x, training=False)

        if edge_att.dim() > 1:
            edge_att = edge_att.squeeze()

        visualisation_scatter.visualise_single_explanation_graph(edge_index, edge_att, num_nodes, pos, vis_path, processed_samples, important_threshold)
        processed_samples += 1

    visualisation_scatter.visualise_overall_explanation_graph(edge_index, edge_att, num_nodes, pos, vis_path, test_iter, model)


def main():
    device, data_dir, results_dir, visualisation_dir, seeds, train_config, data_config, info_config, vis_config, blocks = get_parameters()

    all_results = []
    for seed in seeds:
        print(f"[INFO] Using seed: {seed}")
        set_env(seed)
        num_nodes, zscore, train_iter, val_iter, test_iter, edge_index, gso = data_loader(data_dir,
                                                                                          data_config,
                                                                                          train_config, device)
        loss, path, early_stop, model, optimizer, scheduler, is_visualisation = prepare_model(train_config, data_config,
                                                                                              info_config, blocks,
                                                                                              num_nodes, edge_index,
                                                                                              gso, device, seed,
                                                                                              results_dir)
        train(model, loss, optimizer, scheduler, early_stop, train_iter, val_iter, train_config, info_config,
              results_dir)
        test_MAE, test_RMSE, test_MAPE, model = test(zscore, loss, path, model, test_iter, data_config, train_config,
                                               info_config,
                                               seed, results_dir)

        all_results.append({
            'seed': seed,
            'MAE': test_MAE,
            'RMSE': test_RMSE,
            'MAPE': test_MAPE
        })

        print("\n" + "=" * 50)
        print("[INFO] Visualise HeatMap")
        dataset_name = data_config['dataset_name']
        vis_path = os.path.join(visualisation_dir, dataset_name, str(seed))
        pos = visualisation_scatter.load_node_positions(dataset_name, data_dir)
        results_dir = str(os.path.join(results_dir, dataset_name))
        result_model = os.path.join(results_dir, path)
        model.load_state_dict(torch.load(result_model, map_location=torch.device('cpu')))
        model.eval()

        test_iter_single_batch = vis_dataloader(data_dir, data_config, train_config, device, zscore)

        if is_visualisation:
            visualise(vis_path, vis_config, test_iter_single_batch, model, edge_index, num_nodes, pos)
        print("=" * 50)

    if len(all_results) > 1:
        mae_values = [r['MAE'] for r in all_results]
        rmse_values = [r['RMSE'] for r in all_results]
        mape_values = [r['MAPE'] for r in all_results]

        print("\n" + "=" * 50)
        print("SUMMARY OF ALL SEEDS:")
        print("=" * 50)
        print(f"MAE  - Mean: {statistics.mean(mae_values):.6f}, Std: {statistics.stdev(mae_values):.6f}")
        print(f"RMSE - Mean: {statistics.mean(rmse_values):.6f}, Std: {statistics.stdev(rmse_values):.6f}")
        print(f"MAPE- Mean: {statistics.mean(mape_values):.6f}, Std: {statistics.stdev(mape_values):.6f}")
        print("=" * 50)


if __name__ == '__main__':
    main()
