# STeGEformer: Interpretable Spatio-Temporal Graph Learning for Traffic Flow Forecasting

## About
This project has implemented the STeGEformer model, which is an interpretable spatio-temporal graph learning framework for traffic flow prediction. 
This model combines graph neural networks and the Transformer architecture, enabling it to simultaneously capture the spatio-temporal dependencies in traffic data and providing an interpretability analysis of the prediction results.

## Directory
```
STeGEformer/
├── Code/
│   ├── config/
│   ├── data/
│   ├── model/
│   ├── script/
│   └── main.py
```

## Dataset
The project supports the following two traffic datasets:
PeMS-Bay: Contains traffic sensor data from the Bay Area of California
PeMSD7-M: Contains traffic sensor data from the California region
Each dataset should include the following files:
vel.csv: Speed data
adj.npz: Adjacency matrix
PeMS_Bay_Station_Info.csv or PeMSD7_M_Station_Info.csv: Sensor location information

## Usage instructions
```
# Train the PeMS-Bay dataset using the default configuration
python main.py --dataset pems-bay

# Train the PeMSD7-M dataset using the default configuration
python main.py --dataset pemsd7-m
```
Parameter Configuration
The main parameters can be specified through the command line:
--dataset: Select the dataset (pemsd7-m or pems-bay)
--enable_cuda: Whether to enable CUDA (default: True)

Other hyperparameters can be modified in the configuration file:
config/global_config.yml: Global configuration
config/model_config.yml: Model configuration
config/{dataset}_config.yml: Dataset-specific configuration

## Model Characteristics
### Core Components

- Temporal Gated Convolution: Capturing time-dependent relationships
- Graph Convolution Layer: Modeling spatial dependencies
- LGTEncoder: A spatial-temporal feature extractor combining position encoding and Transformer
- InfoGraphExplainer: Providing interpretability of model decisions

### Interpretability Features

The model integrates the InfoGraphExplainer module, which can:

- Generate edge importance weights
- Visualize key transportation connections
- Analyze the ranking of node importance

### Visualize

After training, automatic visualization results will be generated, including:

- Single sample heat maps and scatter plots
- Overall analysis of important nodes
- Visualization of key connections

The results are saved in the "Visualisations/" directory.
