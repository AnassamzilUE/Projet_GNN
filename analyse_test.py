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

# --- CONFIGURATION DE L'ANALYSE ---
DATASET_PATHS = {
    'yelp': '/content/drive/MyDrive/Colab Notebooks/PFE_AMZIL_ANASS/CARE-GNN/data/YelpChi.mat',
    'amazon': '/content/drive/MyDrive/Colab Notebooks/PFE_AMZIL_ANASS/CARE-GNN/data/Amazon.mat'
}
LOG_DIR = '/content/drive/MyDrive/Colab Notebooks/PFE_AMZIL_ANASS/CARE-GNN/experiment_logs'

# --- CONFIGURATION MODIFIÉE POUR LE TEST ---
EXPERIMENT_NAMES = ['CARE-GNN_Full']
DATASETS_TO_ANALYZE = ['amazon']
SEEDS_TO_ANALYZE = [1]
EXPERIMENTS_EPOCHS = {
    'CARE-GNN_Full': {'yelp': 3, 'amazon': 3}
}

TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 300

# --- FONCTIONS UTILITAIRES ---
def load_data_mat(dataset_name, dataset_path):
    """Charge les données du fichier .mat et retourne feat_data et labels."""
    try:
        data = scipy.io.loadmat(dataset_path)
        labels = data['labels'].flatten()
        feat_data = data['feats']
        if scipy.sparse.issparse(feat_data):
            feat_data = feat_data.todense()
        return feat_data, labels
    except FileNotFoundError:
        print(f"Erreur: Fichier dataset non trouvé à {dataset_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors du chargement ou traitement de {dataset_path}: {e}")
        sys.exit(1)

def visualize_tsne_embeddings(embeddings, labels, title="t-SNE Visualization of Node Embeddings"):
    """Visualise les embeddings des nœuds en 2D avec t-SNE."""
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

# --- MAIN ANALYSIS SCRIPT ---
if __name__ == "__main__":
    print("\n--- Génération des Visualisations pour le Run de Test ---")
    for exp_name in EXPERIMENT_NAMES:
        for dataset_name in DATASETS_TO_ANALYZE:
            for seed in SEEDS_TO_ANALYZE:
                log_path = os.path.join(LOG_DIR, f'{exp_name}_{dataset_name}_seed{seed}.log')
                num_epochs_for_run = EXPERIMENTS_EPOCHS[exp_name][dataset_name]

                if os.path.exists(log_path):
                    print(f"  Génération vis. pour : {log_path}")

                    # --- CORRECTION FINALE ICI ---
                    # On reconstruit le nom du fichier d'embeddings en se basant sur le nom du modèle
                    # comme le fait train.py (`args.model`)
                    if "CARE" in exp_name:
                        model_name_for_file = "CARE"
                    elif "GraphSAGE" in exp_name:
                        model_name_for_file = "SAGE"
                    else:
                        print(f"    AVERTISSEMENT: Modèle non reconnu dans l'exp_name '{exp_name}'. Skipping t-SNE.")
                        continue # Passe à l'itération suivante de la boucle

                    embeddings_save_path = os.path.join(LOG_DIR, f'final_embeddings_{model_name_for_file}_{dataset_name}_seed{seed}.pt')
                    
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

    print("\n--- Visualisations de Test Générées ---")