import argparse
import os
import sys
from utils import prepare_save_dir
from STELLAR import STELLAR
import numpy as np
import torch
import pandas as pd
import anndata
import scanpy as sc
from datasets import GraphDataset, load_own_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='STELLAR')
    parser.add_argument('--training-dataset', help='training dataset')
    parser.add_argument('--testing-dataset', help='testing dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='STELLAR')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--num-heads', type=int, default=22)
    parser.add_argument('--num-seed-class', type=int, default=0)
    parser.add_argument('--sample-rate', type=float, default=1.0)
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size')
    parser.add_argument('--distance_thres', default=50, type=int)
    parser.add_argument('--savedir', type=str, default='./')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.name = '_'.join([args.training_dataset, args.name])
    args = prepare_save_dir(args, __file__)

    dfile = args.training_dataset
    nafile = args.testing_dataset

    full_dataset, inverse_dict = process_training(dfile)
    na_dataset = pd.read_csv(nafile, sep=',', header=0, float_precision='round_trip')
    df_preds, adata = make_preds(full_dataset, na_dataset, inverse_dict, args)
    df_preds.to_csv(dfile + "__" + nafile + '__STELLAR_preds.csv', sep=',', index=False)
    sc_out = '__' + dfile + "__" + nafile + '__STELLAR.pdf'
    plot_umap(adata, sc_out)

def process_training (ptfile):
    pt_dataset = pd.read_csv(ptfile, sep=',', header=0, float_precision='round_trip')
    ptlabels = pt_dataset['cell_type']
    ptcell_types = np.sort(list(set(ptlabels))).tolist()
    ptcell_type_dict = {}
    ptinverse_dict = {}    
    for i, ptcell_type in enumerate(ptcell_types):
        ptcell_type_dict[ptcell_type] = i
        ptinverse_dict[i] = ptcell_type

    ptlabels = np.array([ptcell_type_dict[x] for x in ptlabels])
    ptscores = pt_dataset.drop('cell_type', axis=1)
    ptfull_dataset = ptscores.copy()
    ptfull_dataset['cell_type'] = ptlabels

    return ptfull_dataset, ptinverse_dict

def make_preds (mpf_dataset, mpn_dataset, mpinverse_dict, mpargs):
    labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges = load_own_data(mpf_dataset, mpn_dataset, mpargs.distance_thres, mpargs.sample_rate)
    dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)
    stellar = STELLAR(mpargs, dataset)
    stellar.train()
    pred_mean_uncert, pred_scores, pred_labels = stellar.pred()
    mpadata = anndata.AnnData(unlabeled_X)
    pred_labels = pred_labels.astype('object')

    for i in range(len(pred_labels)):
        if pred_labels[i] in mpinverse_dict.keys():
            pred_labels[i] = mpinverse_dict[pred_labels[i]]

    mpadata.obs['pred'] = pd.Categorical(pred_labels)
    mp_preds = mpn_dataset.copy()
    mp_preds['Pred'] = pred_labels.tolist()
    mp_preds['Score'] = pred_scores.tolist()
    mp_preds.rename(columns = {'Unnamed: 0':''}, inplace = True)

    return mp_preds, mpadata

def plot_umap (puadata, puout):
    sc.pp.neighbors(puadata)
    sc.tl.umap(puadata)
    sc.pl.umap(puadata, color=['pred'], size=5, save=puout)

if __name__ == '__main__':
    main()
