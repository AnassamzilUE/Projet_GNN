import subprocess
import os
import re
import pandas as pd
import time

# --- CONFIGURATION DES EXPÉRIENCES (MODIFIÉE POUR UN TEST RAPIDE) ---

# Définis les datasets sur lesquels tu veux lancer les tests
# DATASETS = ['yelp', 'amazon'] # Ligne originale
DATASETS = ['amazon'] # <-- POUR LE TEST, JUSTE AMAZON QUI EST RAPIDE

# Définis le nombre de runs pour chaque expérience
# SEEDS = [1, 2, 3] # Ligne originale
SEEDS = [1] # <-- POUR LE TEST, JUSTE UNE SEULE SEED

# Arguments communs à toutes les commandes
COMMON_ARGS = [
    '--batch-size', '512',
    '--lr', '1e-3',
    '--test-epochs', '1' # Pour avoir les métriques à chaque époque
]

# Définition de chaque expérience (modèle + ses hyperparamètres spécifiques)
# J'ai commenté toutes les autres expériences pour ne lancer que CARE-GNN_Full
EXPERIMENTS = {
    'CARE-GNN_Full': {
        'model_args': ['--model', 'CARE', '--lambda_1', '2', '--step-size', '2e-2'],
        'epochs': {'yelp': 3, 'amazon': 3} # <-- NOMBRE D'ÉPOQUES TRÈS RÉDUIT (3)
    },
    # 'CARE-GNN_NoLabelAwareSim': {
    #     'model_args': ['--model', 'CARE', '--lambda_1', '0', '--step-size', '2e-2'],
    #     'epochs': {'yelp': 3, 'amazon': 3}
    # },
    # 'CARE-GNN_NoRL': {
    #     'model_args': ['--model', 'CARE', '--lambda_1', '2', '--step-size', '0'],
    #     'epochs': {'yelp': 3, 'amazon': 3}
    # },
    # 'GraphSAGE': {
    #     'model_args': ['--model', 'SAGE'], 
    #     'epochs': {'yelp': 3, 'amazon': 3}
    # }
}

# Dossier où seront sauvegardés les logs et le résumé des résultats
LOG_DIR = 'experiment_logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Liste pour stocker les résultats finaux de chaque run
results_summary = []

# --- BOUCLE PRINCIPALE DES EXPÉRIENCES ---

print("--- Démarrage des expériences (MODE TEST RAPIDE) ---")

for exp_name, exp_config in EXPERIMENTS.items():
    print(f"\nExécution de l'expérience : {exp_name}")
    for dataset in DATASETS:
        print(f"  Dataset : {dataset}")
        num_epochs = exp_config['epochs'][dataset]

        for seed in SEEDS:
            print(f"    Seed : {seed}")
            
            # Construire la commande complète pour train.py
            command = [
                'python', 'train.py',
                '--data', dataset,
                '--num-epochs', str(num_epochs),
                '--seed', str(seed)
            ] + COMMON_ARGS + exp_config['model_args']

            log_filename = os.path.join(LOG_DIR, f'{exp_name}_{dataset}_seed{seed}.log')

            print(f"      Commande : {' '.join(command)}")
            print(f"      Log sauvegardé dans : {log_filename}")

            # Exécuter la commande et capturer la sortie
            start_time_run = time.time()
            process = subprocess.run(command, capture_output=True, text=True) 

            # Sauvegarder la sortie complète dans un fichier log
            with open(log_filename, 'w') as f:
                f.write(process.stdout)
                if process.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(process.stderr)

            end_time_run = time.time()
            total_time_run = end_time_run - start_time_run
            print(f"      Run terminé en {total_time_run:.2f} secondes.")

            # --- PARSING DES RÉSULTATS POUR LE RÉSUMÉ ---
            # ... (la partie parsing reste la même que dans ta version complète) ...
            # ... elle fonctionnera pour cette unique exécution ...

            # -- La logique de parsing est ici pour la complétude, mais tu peux la garder telle quelle --
            output_lines = process.stdout.splitlines()
            final_epoch_metrics = {}
            for line in reversed(output_lines):
                if line.startswith(f'Epoch: {num_epochs-1},'): 
                    f1_match = re.search(r'GNN F1: ([\d.]+)', line)
                    acc_match = re.search(r'GNN Accuracy: ([\d.]+)', line)
                    recall_match = re.search(r'GNN Recall: ([\d.]+)', line)
                    auc_match = re.search(r'GNN auc: ([\d.]+)', line)
                    ap_match = re.search(r'GNN ap: ([\d.]+)', line)
                    thresholds_match = re.search(r'thresholds: \[([\d.,\s-]+)\]', line)
                    
                    if f1_match: final_epoch_metrics['GNN_F1'] = float(f1_match.group(1))
                    if acc_match: final_epoch_metrics['GNN_Accuracy'] = float(acc_match.group(1))
                    if recall_match: final_epoch_metrics['GNN_Recall'] = float(recall_match.group(1))
                    if auc_match: final_epoch_metrics['GNN_AUC'] = float(auc_match.group(1))
                    if ap_match: final_epoch_metrics['GNN_AP'] = float(ap_match.group(1))
                    if thresholds_match: final_epoch_metrics['RL_Thresholds'] = thresholds_match.group(1)
                    break
            
            run_data = {
                'Experiment': exp_name, 'Dataset': dataset, 'Seed': seed, 'Num_Epochs': num_epochs,
                'Run_Time_Sec': total_time_run,
                'Lambda_1': exp_config['model_args'][3] if '--lambda_1' in exp_config['model_args'] else 'N/A',
                'Step_Size': exp_config['model_args'][5] if '--step-size' in exp_config['model_args'] else 'N/A'
            }
            run_data.update(final_epoch_metrics)
            results_summary.append(run_data)

# --- Sauvegarder le résumé global dans un fichier CSV ---
summary_df = pd.DataFrame(results_summary)
summary_csv_path = os.path.join(LOG_DIR, 'test_summary.csv') # Sauvegarde dans un fichier de test
summary_df.to_csv(summary_csv_path, index=False)

print(f"\n--- Test rapide terminé ---")
print(f"Un résumé de test a été sauvegardé dans : {summary_csv_path}")

# Afficher le résumé final (optionnel)
print("\nRésumé du Test :")
print(summary_df)