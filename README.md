# AGSP

The source code of AGSP (Adaptive Gated Simplicial Propagation for Node Classification in Multimodal Graphs) in NeurIPS2026.

## Environment Setup

### 1. Create and activate a conda environment

```bash
   conda create --name agsp python=3.10
   conda activate agsp
```

### 2. Install required dependencies:

```bash
   pip install -r requirements.txt
```

## Dataset Preparation

This project supports two datasets: **Grocery**, ​**Toys**​, ​**Ele-fashion** and ​**Books** Prepare the datasets as follows:

### 1. Grocery and Toys datasets

* Download the datasets from the official [MAGB repository](https://github.com/sktsherlock/MAGB).

### 2. Ele-fashion and Books datasets

* Download the datasets from the [mm-graph-benchmark repository](https://github.com/mm-graph-benchmark/mm-graph-benchmark).

## Running the Code

Use the following commands to train the model on each dataset:

### For Grocery and Toys datasets
```bash
   nohup python -u run.py --dataset grocery > logs/train_grocery_$(date +%Y%m%d_%H%M%S).log 2>&1 &

   nohup python -u run.py --dataset toys > logs/train_toys_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```


### For Books and Ele-fashion datasets

```bash
   nohup python -u run.py --dataset books-nc --use_subgraph --keep_data_cpu --mixed_precision --train_batch_size 7200000 --batch_size_triangles 300 > logs/train_books_$(date +%Y%m%d_%H%M%S).log 2>&1 & 

   nohup python -u run.py --dataset ele-fashion --use_subgraph --keep_data_cpu --mixed_precision > logs/train_ele-fashion_$(date +%Y%m%d_%H%M%S).log 2>&1 & 
```