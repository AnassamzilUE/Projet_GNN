import subprocess
import os
import re
import pandas as pd
import time

# --- CONFIGURATION DES EXPÉRIENCES (POUR GRAPHSAGE UNIQUEMENT) ---

DATASETS = ['yelp', 'amazon']
SEEDS = [1, 2, 3]

COMMON_ARGS = [
    '--batch-size', '512',
    '--lr', '1e-3',
    '--test-epochs', '1'
]

EXPERIMENTS = {
    'GraphSAGE': {
        'model_args': ['--model', 'SAGE'],
        'epochs': {'yelp': 100, 'amazon': 100}
    }
}

LOG_DIR = '/content/drive/MyDrive/Colab Notebooks/PFE_AMZIL_ANASS/CARE-GNN/experiment_logs'
os.makedirs(LOG_DIR, exist_ok=True)

results_summary = []

# --- BOUCLE PRINCIPALE DES EXPÉRIENCES ---
print("--- Démarrage des expériences (GRAPHSAGE UNIQUEMENT) ---")

for exp_name, exp_config in EXPERIMENTS.items():
    print(f"\nExécution de l'expérience : {exp_name}")
    for dataset in DATASETS:
        print(f"  Dataset : {dataset}")
        num_epochs = exp_config['epochs'][dataset]

        for seed in SEEDS:
            print(f"    Seed : {seed}")
            
            command = [
                'python', 'train.py',
                '--data', dataset,
                '--num-epochs', str(num_epochs),
                '--seed', str(seed),
                '--exp_name', exp_name # Important pour la sauvegarde des embeddings
            ] + COMMON_ARGS + exp_config['model_args']

            log_filename = os.path.join(LOG_DIR, f'{exp_name}_{dataset}_seed{seed}.log')

            print(f"      Commande : {' '.join(command)}")
            print(f"      Log sauvegardé dans : {log_filename}")

            start_time_run = time.time()
            process = subprocess.run(command, capture_output=True, text=True)

            with open(log_filename, 'w') as f:
                f.write(process.stdout)
                if process.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(process.stderr)

            end_time_run = time.time()
            total_time_run = end_time_run - start_time_run
            print(f"      Run terminé en {total_time_run:.2f} secondes.")
            
            # La partie parsing reste, mais elle ne trouvera probablement rien
            # dans le CSV si la sortie de SAGE est différente. C'est ok.
            output_lines = process.stdout.splitlines()
            final_epoch_metrics = {}
            for line in reversed(output_lines):
                 if "Test" in line: # SAGE utilise "Test F1:", "Test AUC:" etc.
                     f1_match = re.search(r'Test F1: ([\d.]+)', line)
                     acc_match = re.search(r'Test Accuracy: ([\d.]+)', line)
                     recall_match = re.search(r'Test Recall: ([\d.]+)', line)
                     auc_match = re.search(r'Test AUC: ([\d.]+)', line)
                     ap_match = re.search(r'Test AP: ([\d.]+)', line)
                     if f1_match: final_epoch_metrics['GNN_F1'] = float(f1_match.group(1))
                     if acc_match: final_epoch_metrics['GNN_Accuracy'] = float(acc_match.group(1))
                     if recall_match: final_epoch_metrics['GNN_Recall'] = float(recall_match.group(1))
                     if auc_match: final_epoch_metrics['GNN_AUC'] = float(auc_match.group(2)) # Note: group(2) for SAGE
                     if ap_match: final_epoch_metrics['GNN_AP'] = float(ap_match.group(2)) # Note: group(2) for SAGE
                     print(f"      (Using last reported Test metrics for {exp_name} on {dataset} seed {seed})")
                     break

            run_data = {
                'Experiment': exp_name, 'Dataset': dataset, 'Seed': seed, 'Num_Epochs': num_epochs,
                'Run_Time_Sec': total_time_run
            }
            run_data.update(final_epoch_metrics)
            results_summary.append(run_data)

summary_df = pd.DataFrame(results_summary)
summary_csv_path = os.path.join(LOG_DIR, 'graphsage_summary.csv')
summary_df.to_csv(summary_csv_path, index=False)

print(f"\n--- Expériences GraphSAGE terminées ---")
print(f"Un résumé a été sauvegardé dans : {summary_csv_path}")