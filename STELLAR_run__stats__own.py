import argparse
import os
import sys
from utils import prepare_save_dir
from STELLAR import STELLAR
import numpy as np
import torch
import pandas as pd
from datasets import GraphDataset, load_own_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, matthews_corrcoef, f1_score, cohen_kappa_score
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description='STELLAR')
    parser.add_argument('--dataset', help='training dataset')
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
    args.name = '_'.join([args.dataset, args.name])
    args = prepare_save_dir(args, __file__)

    scores, labels, cell_types = process_input(args.dataset)
    mod_mean_mcc, mod_mean_acc, mod_mean_bacc, mod_mean_ppv_micro, mod_mean_ppv_macro, mod_mean_ppv_weighted, mod_mean_recall_micro, mod_mean_recall_macro, mod_mean_recall_weighted, mod_mean_f1_micro, mod_mean_f1_macro, mod_mean_f1_weighted, mod_mean_kappa, mod_creport, mod_mean_cm = get_metrics(args.dataset, scores, labels, args, cell_types)
    output_metrics(args.dataset, mod_mean_mcc, mod_mean_acc, mod_mean_bacc, mod_mean_ppv_micro, mod_mean_ppv_macro, mod_mean_ppv_weighted, mod_mean_recall_micro, mod_mean_recall_macro, mod_mean_recall_weighted, mod_mean_f1_micro, mod_mean_f1_macro, mod_mean_f1_weighted, mod_mean_kappa, mod_mean_cm)
    plot_cm(args.dataset, mod_mean_cm, cell_types, True)
    plot_cm(args.dataset, mod_mean_cm, cell_types, False)

def process_input (pifile):
    pi_dataset = pd.read_csv(pifile, sep=',', header=0, float_precision='round_trip')
    pilabels = pi_dataset['cell_type']
    picell_types = np.sort(list(set(pilabels))).tolist()
    picell_type_dict = {}
    for i, picell_type in enumerate(picell_types):
        picell_type_dict[picell_type] = i

    pilabels = np.array([picell_type_dict[x] for x in pilabels])
    piscores = pi_dataset.drop('cell_type', axis=1)

    return piscores, pilabels, picell_types 

def output_metrics (omfile, om_mcc, om_acc, om_bacc, om_ppv_micro, om_ppv_macro, om_ppv_weighted, om_recall_micro, om_recall_macro, om_recall_weighted, om_f1_micro, om_f1_macro, om_f1_weighted, om_kappa, om_creport, om_cm):
    statsfile = omfile + "__stats.txt"
    sout = open(statsfile, "w")
    sout.write(f"mod_mean_mcc\t{om_mcc}\n")
    sout.write(f"mod_mean_accuracy\t{om_acc}\n")
    sout.write(f"mod_mean_balanced_accuracy\t{om_bacc}\n")
    sout.write(f"mod_mean_precision_micro\t{om_ppv_micro}\n")
    sout.write(f"mod_mean_precision_macro\t{om_ppv_macro}\n")
    sout.write(f"mod_mean_precision_weighted\t{om_ppv_weighted}\n")
    sout.write(f"mod_mean_recall_micro\t{om_recall_micro}\n")
    sout.write(f"mod_mean_recall_macro\t{om_recall_macro}\n")
    sout.write(f"mod_mean_recall_weighted\t{om_recall_weighted}\n")
    sout.write(f"mod_mean_f1_micro\t{om_f1_micro}\n")
    sout.write(f"mod_mean_f1_macro\t{om_f1_macro}\n")
    sout.write(f"mod_mean_f1_weighted\t{om_f1_weighted}\n")
    sout.write(f"mod_mean_cohen_kappa\t{om_kappa}\n")
    sout.close()

def plot_cm (pcfile, pccm, pccell_types, pcannot):
    plt.figure(figsize = (12,10))
    ax = sns.heatmap(pccm, linewidths=0.1, annot=pcannot, fmt='.0f', cmap='PuBuGn')
    ax.set_xticklabels(pccell_types, size=10)
    ax.set_yticklabels(pccell_types, size=10, rotation='horizontal')
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_xlabel("predicted cell type", fontsize = 12)
    ax.set_ylabel("true cell type", fontsize = 12)
    ax.set_title("STELLAR Predictions for Melanoma", fontsize = 14)

    pcout = pcfile
    if pcannot == True:
        pcout = pcout + "__confusionMatrix__sns__vals.png"
    else:
        pcout = pcout + "__confusionMatrix__sns.png"

    plt.savefig(pcout, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.cla()
    plt.close()

def get_metrics (gmfile, gmscores, gmlabels, gmargs, gmcell_types):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=gmargs.seed)

    mccs = []
    accs = []
    baccs = []
    ppvs_micro = []
    ppvs_macro = []
    ppvs_weighted = []
    recalls_micro = []
    recalls_macro = []
    recalls_weighted = []
    f1s_micro = []
    f1s_macro = []
    f1s_weighted = []
    kappas = []
    creports = []
    cms = []
    mcms = []

    for train_index,test_index in skf.split(gmscores, gmlabels):
        gmscores_train, gmscores_test = gmscores.iloc[train_index, :], gmscores.iloc[test_index, :]
        gmlabels_train, gmlabels_test = gmlabels[train_index], gmlabels[test_index]
        df_train = gmscores_train.copy()
        df_train['cell_type'] = gmlabels_train
        df_test = gmscores_test.copy()
        df_test['cell_type'] = gmlabels_test

        gmlabeled_X, gmlabeled_y, gmunlabeled_X, gmlabeled_edges, gmunlabeled_edges = load_own_data(df_train, df_test, gmargs.distance_thres, gmargs.sample_rate)
        gmdataset = GraphDataset(gmlabeled_X, gmlabeled_y, gmunlabeled_X, gmlabeled_edges, gmunlabeled_edges)
        gmstellar = STELLAR(gmargs, gmdataset)
        gmstellar.train()
        gmpred_mean_uncert, gmpred_scores, gmpred_labels = gmstellar.pred()

        mcc = matthews_corrcoef(gmlabels_test, gmpred_labels)
        mccs.append(mcc)

        accuracy = accuracy_score(gmlabels_test, gmpred_labels)
        accs.append(accuracy)
        balanced_accuracy = balanced_accuracy_score(gmlabels_test, gmpred_labels)
        baccs.append(balanced_accuracy)

        precision_micro = precision_score(gmlabels_test, gmpred_labels, average="micro")
        ppvs_micro.append(precision_micro)
        precision_macro = precision_score(gmlabels_test, gmpred_labels, average="macro")
        ppvs_macro.append(precision_macro)
        precision_weighted = precision_score(gmlabels_test, gmpred_labels, average="weighted")
        ppvs_weighted.append(precision_weighted)

        recall_micro = recall_score(gmlabels_test, gmpred_labels, average="micro")
        recalls_micro.append(recall_micro)
        recall_macro = recall_score(gmlabels_test, gmpred_labels, average="macro")
        recalls_macro.append(recall_macro)
        recall_weighted = recall_score(gmlabels_test, gmpred_labels, average="weighted")
        recalls_weighted.append(recall_weighted)

        f1_micro = f1_score(gmlabels_test, gmpred_labels, average="micro")
        f1s_micro.append(f1_micro)
        f1_macro = f1_score(gmlabels_test, gmpred_labels, average="macro")
        f1s_macro.append(f1_macro)
        f1_weighted = f1_score(gmlabels_test, gmpred_labels, average="weighted")
        f1s_weighted.append(f1_weighted)

        cohen_kappa = cohen_kappa_score(gmlabels_test, gmpred_labels)
        kappas.append(cohen_kappa)

        cm = confusion_matrix(gmlabels_test, gmpred_labels)
        cms.append(cm)

    return np.mean(mccs), np.mean(accs), np.mean(baccs), np.mean(ppvs_micro), np.mean(ppvs_macro), np.mean(ppvs_weighted), np.mean(recalls_micro), np.mean(recalls_macro), np.mean(recalls_weighted), np.mean(f1s_micro), np.mean(f1s_macro), np.mean(f1s_weighted), np.mean(kappas), np.mean(cms, axis=0)

if __name__ == '__main__':
    main()
