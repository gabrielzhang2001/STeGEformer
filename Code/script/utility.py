import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import tqdm


def calc_gso(dir_adj):
    n_vertex = dir_adj.shape[0]

    if not sp.issparse(dir_adj):
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)

    row_sum = adj.sum(axis=1).A1
    row_sum_inv_sqrt = np.power(row_sum, -0.5)
    row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
    deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
    sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

    sym_norm_lap = id - sym_norm_adj
    gso = sym_norm_lap

    return gso


def show_edge_info(edge_atts, epoch, current_r, state):
    with torch.no_grad():
        all_edge_atts = torch.cat(edge_atts, dim=0)

        att_mean = all_edge_atts.mean().item()
        att_std = all_edge_atts.std().item()
        att_max = all_edge_atts.max().item()
        att_min = all_edge_atts.min().item()
        high_att_ratio = (all_edge_atts >= 0.5).float().mean().item()
        low_att_ratio = (all_edge_atts < 0.5).float().mean().item()
        print(
            f'[INFO] {state} Epoch {epoch}: Current R: {current_r:.6f} | Mean: {att_mean:.6f} | Std: {att_std:.6f} |\nHigh Att. Ratio (>=0.5): {high_att_ratio:.6f} | Low Att. Ratio: {low_att_ratio:.6f} |\nMax: {att_max:.6f} | Min: {att_min:.6f}')


def evaluate_model(model, loss, data_iter, info_loss_coef, use_explanation):
    model.eval()
    pred_loss_sum, info_loss_sum, loss_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        all_edge_atts = []
        #for x, y in tqdm.tqdm(data_iter, file=sys.stdout):
        for x, y in data_iter:
            y_pred, edge_att = model(x, training=False)
            pred_loss = loss(y_pred, y)
            info_loss = torch.tensor(0.0)
            r = 0.0
            if use_explanation:
                info_loss, r = model.get_info_loss(edge_att, epoch=1000)
                all_edge_atts.append(edge_att.detach())
            total_loss = pred_loss + info_loss_coef * info_loss

            loss_sum += total_loss.item() * y.shape[0]
            n += y.shape[0]
            pred_loss_sum += pred_loss.item() * y.shape[0]
            info_loss_sum += info_loss.item() * y.shape[0]

        print("=" * 50)
        print(
            f'[INFO] Test Loss: {loss_sum / n:.6f} | Pred. Loss: {pred_loss_sum / n:.6f} | Info. Loss: {info_loss_sum / n:.6f}')
        if use_explanation:
            show_edge_info(all_edge_atts, 0, r, 'Test')
        print("=" * 50)


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with (torch.no_grad()):
        mae, sum_y, mape, mse = [], [], [], []
        eps = 1e-5
        #all_predictions = []

        #for x, y in tqdm.tqdm(data_iter, file=sys.stdout):
        for x, y in data_iter:
            y_original_shape = y.shape
            y = y.cpu().numpy().reshape(-1, y_original_shape[-1])
            y = scaler.inverse_transform(y).reshape(-1)
            y_pred, _ = model(x, training=False)

            # last_step_pred = y_pred[:, -1, :]
            # last_step_pred = last_step_pred.cpu().numpy().reshape(-1, y_original_shape[-1])
            # last_step_pred = scaler.inverse_transform(last_step_pred)
            # all_predictions.append(last_step_pred)

            y_pred = y_pred.cpu().numpy().reshape(-1, y_original_shape[-1])
            y_pred = scaler.inverse_transform(y_pred)
            y_pred = y_pred.reshape(-1)
            y = y.reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mse += (d ** 2).tolist()

            mask = np.abs(y) > eps
            if np.any(mask):
                safe_mape = d[mask] / np.abs(y[mask])
                mape.extend(safe_mape.tolist())

        # final_predictions = np.vstack(all_predictions)  # shape: (total_batches * batch_size, num_nodes)
        # final_predictions = np.round(final_predictions, 1)
        # df = pd.DataFrame(final_predictions)
        # df.to_csv('predictions.csv', index=False)

        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        #WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        # return MAE, MAPE, RMSE
        return MAE, RMSE, MAPE


