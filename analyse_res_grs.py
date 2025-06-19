import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import scipy.io

# Assure-toi que le répertoire 'CARE-GNN' est bien au niveau attendu
current_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
care_gnn_modules_path = os.path.join(current_script_dir, 'CARE-GNN')
if care_gnn_modules_path not in sys.path:
    sys.path.append(care_gnn_modules_path)

from model import OneLayerCARE 
from graphsage import GraphSage, MeanAggregator
from layers import InterAgg, IntraAgg
from utils import *
import random

# --- CONFIGURATION POUR GRAPH_SAGE UNIQUEMENT ---
DATASET_PATHS = {
    'yelp': '/content/drive/MyDrive/Colab Notebooks/PFE_AMZIL_ANASS/CARE-GNN/data/YelpChi.mat',
    'amazon': '/content/drive/MyDrive/Colab Notebooks/PFE_AMZIL_ANASS/CARE-GNN/data/Amazon.mat'
}
LOG_DIR = '/content/drive/MyDrive/Colab Notebooks/PFE_AMZIL_ANASS/CARE-GNN/experiment_logs' # Nouveau dossier de logs
EXPERIMENT_NAMES = ['GraphSAGE']
DATASETS_TO_ANALYZE = ['yelp', 'amazon']
SEEDS_TO_ANALYZE = [1, 2, 3]
EXPERIMENTS_EPOCHS = {
    'GraphSAGE': {'yelp': 100, 'amazon': 100}
}
TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 300

# --- FONCTIONS UTILITAIRES (Inchangées) ---
# ... (tes fonctions load_data_mat, visualize_tsne_embeddings, etc.) ...
# ... (le code des fonctions utilitaires est ici pour la complétude) ...
def load_data_mat(dataset_name, dataset_path):
    try:
        data = scipy.io.loadmat(dataset_path)
        labels = data['labels'].flatten()
        feat_data = data['feats']
        if scipy.sparse.issparse(feat_data):
            feat_data = feat_data.todense()
        return feat_data, labels
    except FileNotFoundError:
        print(f"Erreur: Fichier dataset non trouvé à {dataset_path}")
        return None, None
    except Exception as e:
        print(f"Erreur lors du chargement ou traitement de {dataset_path}: {e}")
        return None, None

def visualize_tsne_embeddings(embeddings, labels, title="t-SNE Visualization"):
    print(f"Effectuer la réduction de dimension t-SNE pour '{title}'...")
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if len(embeddings) < TSNE_PERPLEXITY or len(np.unique(labels)) < 2:
        print(f"AVERTISSEMENT: Pas assez de points ({len(embeddings)}) ou de classes ({len(np.unique(labels))}) pour t-SNE. Skipped.")
        return
    if len(embeddings) > 5000:
        print(f"AVERTISSEMENT: t-SNE est lent sur {len(embeddings)} points. Échantillonnage à 5000.")
        indices_pos = np.where(labels == 1)[0]
        indices_neg = np.where(labels == 0)[0]
        sample_size_pos = min(len(indices_pos), 500)
        sample_size_neg = min(len(indices_neg), 5000 - sample_size_pos)
        sampled_indices_pos = np.random.choice(indices_pos, sample_size_pos, replace=False) if sample_size_pos > 0 else np.array([])
        sampled_indices_neg = np.random.choice(indices_neg, sample_size_neg, replace=False) if sample_size_neg > 0 else np.array([])
        sampled_indices = np.concatenate((sampled_indices_pos, sampled_indices_neg))
        np.random.shuffle(sampled_indices)
        sampled_embeddings = embeddings[sampled_indices]
        sampled_labels = labels[sampled_indices]
    else:
        sampled_embeddings = embeddings
        sampled_labels = labels
    sampled_labels_int = sampled_labels.astype(int)
    tsne = TSNE(n_components=TSNE_COMPONENTS, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER, random_state=42)
    embeddings_2d = tsne.fit_transform(sampled_embeddings)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
        hue=sampled_labels_int,
        palette=sns.color_palette("hls", len(np.unique(sampled_labels_int))),
        legend="full", alpha=0.6
    )
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    os.makedirs(LOG_DIR, exist_ok=True)
    plt.savefig(os.path.join(LOG_DIR, f"{title.replace(' ', '_').replace('/', '_')}_tsne.png"))
    plt.close()

def plot_metrics_evolution(metrics_df, metric_names, title="Évolution des Métriques"):
    if metrics_df.empty:
        print(f"Pas de données de métriques à tracer pour '{title}'.")
        return
    plt.figure(figsize=(12, 6))
    for metric_name in metric_names:
        if metric_name in metrics_df.columns:
            plt.plot(metrics_df['Epoch'], metrics_df[metric_name], label=metric_name)
    plt.title(title)
    plt.xlabel("Époque")
    plt.ylabel("Valeur de la Métrique")
    plt.legend()
    plt.grid(True)
    os.makedirs(LOG_DIR, exist_ok=True)
    plt.savefig(os.path.join(LOG_DIR, f"{title.replace(' ', '_').replace('/', '_')}_metrics_evolution.png"))
    plt.close()

def parse_metrics_from_log(log_path):
    metrics_data = []
    try:
        with open(log_path, 'r') as f: lines = f.readlines()
    except FileNotFoundError: return pd.DataFrame()
    for line_idx, line in enumerate(lines):
        if line.startswith('Epoch:') or "Test" in line:
            current_epoch_metrics = {}
            epoch_match = re.search(r'Epoch: (\d+)', line)
            if epoch_match:
                current_epoch_metrics['Epoch'] = int(epoch_match.group(1))
            
            f1_match = re.search(r'(GNN F1|Test F1): ([\d.]+)', line)
            acc_match = re.search(r'(GNN Accuracy|Test Accuracy): ([\d.]+)', line)
            recall_match = re.search(r'(GNN Recall|Test Recall): ([\d.]+)', line)
            auc_match = re.search(r'(GNN auc|Test AUC): ([\d.]+)', line)
            ap_match = re.search(r'(GNN ap|Test AP): ([\d.]+)', line)

            for i in range(1, 11):
                if line_idx + i >= len(lines): break
                next_line = lines[line_idx + i]
                if not f1_match: f1_match = re.search(r'(GNN F1|Test F1): ([\d.]+)', next_line)
                if not acc_match: acc_match = re.search(r'(GNN Accuracy|Test Accuracy): ([\d.]+)', next_line)
                if not recall_match: recall_match = re.search(r'(GNN Recall|Test Recall): ([\d.]+)', next_line)
                if not auc_match: auc_match = re.search(r'(GNN auc|Test AUC): ([\d.]+)', next_line)
                if not ap_match: ap_match = re.search(r'(GNN ap|Test AP): ([\d.]+)', next_line)
                if all([f1_match, acc_match, recall_match, auc_match, ap_match]): break

            if all([f1_match, acc_match, recall_match, auc_match, ap_match]):
                current_epoch_metrics['GNN_F1'] = float(f1_match.group(2))
                current_epoch_metrics['GNN_Accuracy'] = float(acc_match.group(2))
                current_epoch_metrics['GNN_Recall'] = float(recall_match.group(2))
                current_epoch_metrics['GNN_AUC'] = float(auc_match.group(2))
                current_epoch_metrics['GNN_AP'] = float(ap_match.group(2))
                metrics_data.append(current_epoch_metrics)
    
    if not metrics_data:
        return pd.DataFrame(columns=['Epoch', 'GNN_F1', 'GNN_Accuracy', 'GNN_Recall', 'GNN_AUC', 'GNN_AP'])
    return pd.DataFrame(metrics_data)

# --- MAIN ANALYSIS SCRIPT ---
if __name__ == "__main__":
    last_epoch_metrics_for_aggregation = []
    all_runs_metrics_df_list = []

    print("\n--- Parsing des logs de GraphSAGE ---")
    for exp_name in EXPERIMENT_NAMES:
        for dataset_name in DATASETS_TO_ANALYZE:
            for seed in SEEDS_TO_ANALYZE:
                log_path = os.path.join(LOG_DIR, f'{exp_name}_{dataset_name}_seed{seed}.log')
                if os.path.exists(log_path):
                    print(f"  Parsage du log : {log_path}")
                    metrics_df_for_run_full = parse_metrics_from_log(log_path)
                    
                    if not metrics_df_for_run_full.empty:
                        last_epoch_row = metrics_df_for_run_full.iloc[-1].to_dict()
                        last_epoch_row.update({
                            'Experiment': exp_name, 'Dataset': dataset_name, 'Seed': seed
                        })
                        last_epoch_metrics_for_aggregation.append(last_epoch_row)
                        
                        metrics_df_for_run_full['Experiment'] = exp_name
                        metrics_df_for_run_full['Dataset'] = dataset_name
                        metrics_df_for_run_full['Seed'] = seed
                        all_runs_metrics_df_list.append(metrics_df_for_run_full)
                    else:
                        print(f"    AVERTISSEMENT: Aucune métrique trouvée dans le log {log_path}.")
                else:
                    print(f"    AVERTISSEMENT: Log non trouvé {log_path}.")
    
    if not last_epoch_metrics_for_aggregation:
        print("ERREUR: Aucun run n'a pu être parsé. Impossible de continuer.")
        sys.exit(1)
        
    final_aggregation_df = pd.DataFrame(last_epoch_metrics_for_aggregation)

    print("\n--- Calcul des performances moyennes et écarts-types pour GraphSAGE ---")
    aggregated_results_df = final_aggregation_df.groupby(['Experiment', 'Dataset']).agg(
        Mean_GNN_F1=('GNN_F1', 'mean'), Std_GNN_F1=('GNN_F1', 'std'),
        Mean_GNN_Accuracy=('GNN_Accuracy', 'mean'), Std_GNN_Accuracy=('GNN_Accuracy', 'std'),
        Mean_GNN_Recall=('GNN_Recall', 'mean'), Std_GNN_Recall=('GNN_Recall', 'std'),
        Mean_GNN_AUC=('GNN_AUC', 'mean'), Std_GNN_AUC=('GNN_AUC', 'std'),
        Mean_GNN_AP=('GNN_AP', 'mean'), Std_GNN_AP=('GNN_AP', 'std')
    ).round(4)
    
    print(aggregated_results_df.to_string())

    print("\n--- Génération des Visualisations pour GraphSAGE ---")
    all_runs_metrics_df = pd.concat(all_runs_metrics_df_list, ignore_index=True)
    for exp_name in EXPERIMENT_NAMES:
        for dataset_name in DATASETS_TO_ANALYZE:
            for seed in SEEDS_TO_ANALYZE:
                log_path = os.path.join(LOG_DIR, f'{exp_name}_{dataset_name}_seed{seed}.log')
                num_epochs_for_run = EXPERIMENTS_EPOCHS[exp_name][dataset_name]

                if os.path.exists(log_path):
                    print(f"  Génération vis. pour : {log_path}")

                    run_metrics_df = all_runs_metrics_df[
                        (all_runs_metrics_df['Experiment'] == exp_name) &
                        (all_runs_metrics_df['Dataset'] == dataset_name) &
                        (all_runs_metrics_df['Seed'] == seed)
                    ]
                    plot_metrics_evolution(run_metrics_df, metric_names=['GNN_AUC', 'GNN_Recall'], title=f"AUC & Recall - {exp_name} sur {dataset_name} (Seed {seed})")
                    
                    embeddings_save_path = os.path.join(LOG_DIR, f'final_embeddings_{exp_name}_{dataset_name}_seed{seed}.pt')
                    if os.path.exists(embeddings_save_path):
                        try:
                            saved_data = torch.load(embeddings_save_path, map_location=torch.device('cpu'))
                            embeddings = saved_data['embeddings'].numpy()
                            current_labels = saved_data['labels'].numpy()
                            
                            if current_labels.ndim > 1: current_labels = current_labels.flatten()
                            
                            if embeddings.shape[0] != len(current_labels):
                                print(f"    AVERTISSEMENT: Taille incohérente pour t-SNE: {embeddings.shape[0]} embeddings vs {len(current_labels)} labels. Skipped.")
                            else:
                                visualize_tsne_embeddings(embeddings, current_labels, title=f"t-SNE Embeddings - {exp_name} sur {dataset_name} (Seed {seed})")
                        except Exception as e:
                            print(f"    ERREUR lors du chargement ou traitement des embeddings pour {embeddings_save_path}: {e}")
                    else:
                        print(f"    AVERTISSEMENT: Embeddings finaux non trouvés pour {embeddings_save_path}. Skipping t-SNE.")

    print("\n--- Toutes les visualisations ont été générées ---")