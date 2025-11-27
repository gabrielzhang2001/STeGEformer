import os
import sys

import folium
import numpy as np
import torch
import tqdm
from folium.plugins import HeatMap
import pandas as pd


def load_node_positions(dataset_name, data_dir):
    if dataset_name == 'pemsd7-m':
        station_info_path = os.path.join(data_dir, dataset_name, 'PeMSD7_M_Station_Info.csv')
        if os.path.exists(station_info_path):
            df = pd.read_csv(station_info_path)
            pos = {}
            for idx, row in df.iterrows():
                pos[idx] = (row['Longitude'], row['Latitude'])
            return pos

    if dataset_name == 'pems-bay':
        station_info_path = os.path.join(data_dir, dataset_name, 'PeMS_Bay_Station_Info.csv')
        if os.path.exists(station_info_path):
            df = pd.read_csv(station_info_path, header=None)
            pos = {}
            for idx, row in df.iterrows():
                pos[idx] = (row.iloc[2], row.iloc[1])
            return pos


def get_heatmap_color(normalized_value):
    if normalized_value < 0.5:
        t = normalized_value * 2  # 0 to 1
        r = 0
        g = int(255 * t)
        b = int(255 * (1 - t))
    else:
        t = (normalized_value - 0.5) * 2  # 0 to 1
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 0
    return f'#{r:02x}{g:02x}{b:02x}'


def visualise_single_explanation_graph(edge_index, edge_att, num_nodes, pos, vis_path, processed_samples, edge_threshold):
    src_nodes = edge_index[0].cpu().numpy()
    dst_nodes = edge_index[1].cpu().numpy()
    non_self_loop_mask = src_nodes != dst_nodes
    src_nodes = src_nodes[non_self_loop_mask]
    dst_nodes = dst_nodes[non_self_loop_mask]

    if isinstance(edge_att, torch.Tensor):
        att_weights = edge_att.cpu().numpy()
    else:
        att_weights = np.array(edge_att)

    node_weights = np.zeros(num_nodes)
    for i in range(len(src_nodes)):
        src, dst = src_nodes[i], dst_nodes[i]
        weight = att_weights[i]
        node_weights[src] += weight
        node_weights[dst] += weight

    if np.max(node_weights) > 0:
        node_weights = node_weights / np.max(node_weights)

    heatmap_data = []
    for i in range(num_nodes):
        longitude, latitude = pos.get(i, (None, None)) if pos else (None, None)
        importance_score = node_weights[i]
        heatmap_data.append([i, longitude, latitude, importance_score])

    edges_above_threshold = []
    weights_above_threshold = []
    for i in range(len(src_nodes)):
        if att_weights[i] > edge_threshold:
            edges_above_threshold.append((src_nodes[i], dst_nodes[i]))
            weights_above_threshold.append(att_weights[i])

    df = pd.DataFrame(heatmap_data, columns=['node_id', 'longitude', 'latitude', 'importance_score'])
    heatmap_data = df[['latitude', 'longitude', 'importance_score']].values.tolist()
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    HeatMap(heatmap_data, radius=15).add_to(m)

    if weights_above_threshold:
        min_weight = min(weights_above_threshold)
        max_weight = max(weights_above_threshold)
        weight_range = max_weight - min_weight if max_weight != min_weight else 1
    else:
        weight_range = 1

    for i, (src, dst) in enumerate(edges_above_threshold):
        src_pos = pos.get(src, (None, None))
        dst_pos = pos.get(dst, (None, None))

        if src_pos[0] is not None and src_pos[1] is not None and \
                dst_pos[0] is not None and dst_pos[1] is not None:
            line_weight = 5
            opacity = 0.1
            if weights_above_threshold:
                normalized_weight = (weights_above_threshold[i] - min_weight) / weight_range
                color = get_heatmap_color(normalized_weight)
            else:
                color = 'blue'

            folium.PolyLine(
                locations=[[src_pos[1], src_pos[0]], [dst_pos[1], dst_pos[0]]],
                color=color,
                weight=line_weight,
                opacity=opacity
            ).add_to(m)

    picture_path = os.path.join(vis_path, f"sample_{processed_samples + 1}_heatmap.html")
    m.save(picture_path)


def visualise_overall_explanation_graph(edge_index, edge_att, num_nodes, pos, vis_path, test_iter, model):
    model.eval()

    overall_node_weights = np.zeros(num_nodes)

    for x, y in tqdm.tqdm(test_iter, file=sys.stdout):
        with torch.no_grad():
            y_pred, edge_att = model(x, training=False)

        if isinstance(edge_att, torch.Tensor):
            att_weights = edge_att.cpu().numpy()
        else:
            att_weights = np.array(edge_att)

        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()
        non_self_loop_mask = src_nodes != dst_nodes
        src_nodes = src_nodes[non_self_loop_mask]
        dst_nodes = dst_nodes[non_self_loop_mask]

        for i in range(len(src_nodes)):
            overall_node_weights[src_nodes[i]] += att_weights[i]
            overall_node_weights[dst_nodes[i]] += att_weights[i]

    if np.max(overall_node_weights) > 0:
        overall_node_weights = overall_node_weights / np.max(overall_node_weights)

    print(f"Node importance range: {np.min(overall_node_weights):.6f} - {np.max(overall_node_weights):.6f}")

    top_k = 5
    top_indices = np.argsort(overall_node_weights)[-top_k:][::-1]
    top_values = overall_node_weights[top_indices]

    print(f"Top {top_k} most important nodes:")
    for i, (idx, value) in enumerate(zip(top_indices, top_values)):
        node_coords = pos.get(idx, (None, None)) if pos else (None, None)
        print(f"  Rank {i + 1}:")
        print(f"    Node ID: {idx}")
        print(f"    Importance Score: {value:.6f}")
        print(f"    Longitude: {node_coords[0]}")
        print(f"    Latitude: {node_coords[1]}")

    heatmap_data = []
    for i in range(num_nodes):
        longitude, latitude = pos.get(i, (None, None)) if pos else (None, None)
        importance_score = overall_node_weights[i]
        heatmap_data.append([i, longitude, latitude, importance_score])

    df = pd.DataFrame(heatmap_data, columns=['node_id', 'longitude', 'latitude', 'importance_score'])

    visualize_top_k_nodes_on_map(df, pos, top_indices, vis_path, f"{top_k}_nodes.html")

    heatmap_data = df[['latitude', 'longitude', 'importance_score']].values.tolist()
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    HeatMap(heatmap_data, radius=15).add_to(m)

    picture_path = os.path.join(vis_path, f"overall_heatmap.html")
    m.save(picture_path)


def visualize_top_k_nodes_on_map(df, pos, top_k_indices, vis_path, filename="top_k_nodes.html"):
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for idx in top_k_indices:
        longitude, latitude = pos.get(idx, (None, None)) if pos else (None, None)
        if longitude is not None and latitude is not None:
            folium.CircleMarker(
                location=[latitude, longitude],
                radius=8,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7,
                popup=f"Node ID: {idx}"
            ).add_to(m)

            folium.map.Marker(
                [latitude, longitude],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 14px; font-weight: bold; color: black;">{idx}</div>',
                )
            ).add_to(m)

    picture_path = os.path.join(vis_path, filename)
    m.save(picture_path)

