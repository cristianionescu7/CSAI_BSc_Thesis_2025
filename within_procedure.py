import os
import sys
import mne
import warnings
import numpy as np
from datetime import datetime
from models import CNN, LSTM_CNN
from Inner_Speech_Dataset.Python_Processing.Data_extractions import extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import filter_by_condition, select_time_window
from Inner_Speech_Dataset.Python_Processing.Utilitys import picks_from_channels
from train_utils import normalize_eeg_data, train_and_evaluate_model, log_results
from sklearn.model_selection import StratifiedKFold

"""
Script for running within-subject decoding experiments on the inner speech EEG dataset.

This script handles the full pipeline:
- Loads and preprocesses EEG data (including optional interval and electrode filtering)
- Initializes and trains a selected deep learning model (e.g., CNN, LSTM-CNN)
- Performs within-subject evaluation using stratified K-fold cross-validation
- Logs performance metrics, confusion matrices, and results for downstream analysis

Used to benchmark model performance in the within-procedure setting, where each subject is evaluated independently.
"""

def setup_environment():
    """
    Set up the environment by cloning the Inner Speech Dataset
    repository (Nieto et al., 2021) and creating necessary directories.
    """
    # Clone repo once manually or ensure it's already cloned
    if not os.path.exists("Inner_Speech_Dataset"):
        os.system("git clone https://github.com/N-Nieto/Inner_Speech_Dataset -q")

    # Add path to repo
    sys.path.append("./Inner_Speech_Dataset")

    # Environment settings
    np.random.seed(42)
    mne.set_log_level(verbose='warning')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    setup_environment()

    root_dir = "."
    fs = 256
    t_start = 1 #start of action interval
    t_end = 3.5 #end of action interval

    #BioSemi 128
    channel_names = (
        [f"A{i}" for i in range(1, 33)] +
        [f"B{i}" for i in range(1, 33)] +
        [f"C{i}" for i in range(1, 33)] +
        [f"D{i}" for i in range(1, 33)]
    )
    left_channels = (
        #picks_from_channels("FRONT_L") +
        #picks_from_channels("C_L") +
        #picks_from_channels("OCC_L") #+
        #["D4", "D9", "D11", "D20", "D22", "D23", "D25", "D27", "D29"]
        #Broca's and Wernicke's areas channels: 
        #["C31", "C32", "D5", "D6", "D26", "D27", "D28", "D29"]
        ["C18", "C20", "C27", "C31", "C32", 
    "D6", "D5", "D10", "D15", "D19", "D21", "D26", "D30", 
    "A1", "A2", "A4", "A5", "A8", "A9", "A10", "A15", 
    "A19", "A20", "A22", "A23", "A24"]
    )
    #used for filtering electrodes
    channel_name_to_index = {name:idx 
                            for idx, name in enumerate(channel_names)}
    left_indices = [channel_name_to_index[ch] 
                    for ch in left_channels]

    #set hyperparameters
    n_splits = 5
    lr = 0.0006
    eps = 40
    b_s = 64

    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
    scores = {metric: [] for metric in metrics_list}
    overall_confusion_matrix = np.zeros((4, 4), dtype=int)

    # Used for logging the results.
    # Variables that are used in the code have a comment next to them.
    # The rest are just for logging.
    config = {
        "model": "LSTM-CNN",
        "condition": "INNER", # change to select other conditions["ALL", "INNER", "VIS", "PRON"]
        "paradigm": "within-subject",
        "k-fold": True,
        "normalization": "standard", #["standard", "robust"]
        "channels": "26", # [26, "all"]
        "interval": "action", # ["full", "action"]
        "hidden_size": 64,
        "LSTM num_layers": 2,
        "LSTM bidirectional": False,
        "CNN num_layers": 3,
        "CNN dropout": 0.25,
        "learning_rate": lr,
        "epochs": eps,
        "batch_size": b_s,
        "weight_decay": 1e-2,
        "note": ""
    }

    # Write config + results to a txt file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_{timestamp}.txt"
    print(f"Logging to {log_path}")

    with open(log_path, "w") as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

        f.write("\nResults:\n")


    print("Starting training and evaluation...")
    for subject in range(1, 11):
        print(f"\nSubject {subject}")

        # Load and slice data
        X, Y = extract_data_from_subject(root_dir, subject, datatype="eeg")
        X, Y = filter_by_condition(X, Y, config.get("condition", "INNER"))
        Y = Y[:, 1] # Use only the second column (labels)
        
        # Time interval cropping
        if config.get("interval", "full") != "full":
            X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)

        # Channel selection
        if config.get("channels", "all") != "all":
            X = X[:, left_indices, :]

        n_c = X.shape[1]  # Number of channels
        n_s = X.shape[2]  # Number of samples
        subject_metrics = {metric: [] for metric in metrics_list}
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
            print(f"--Fold {fold_idx + 1}/{n_splits}--")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = Y[train_idx], Y[test_idx]

            X_train, X_test = normalize_eeg_data(X_train, X_test, config.get("normalization", "standard"))
        
            acc, prec, rec, f1, cm = train_and_evaluate_model(
                X_train, y_train,
                X_test, y_test, 
                model_class=lambda: LSTM_CNN(num_channels=n_c, num_samples=n_s),
                learning_rate=lr, 
                epochs=eps, 
                batch_size=b_s)
            
            print(f"\nFold {fold_idx + 1} - " + ", ".join(
                [f"{m.capitalize()}: {v:.4f}" 
                for m, v in zip(metrics_list[:-1], [acc, prec, rec, f1])]
            ))
            for metric, value in zip(metrics_list, [acc, prec, rec, f1, cm]):
                subject_metrics[metric].append(value)
        
        subject_summary = {
            metric: np.mean(subject_metrics[metric]) if metric != 'confusion_matrix' 
            else np.sum(subject_metrics[metric], axis=0)
            for metric in metrics_list
        }

        print(f"\nSubject {subject} final accuracy: {subject_summary['accuracy']:.4f}; "
            f"precision: {subject_summary['precision']:.4f}; "
            f"recall: {subject_summary['recall']:.4f}; "
            f"F1 score: {subject_summary['f1']:.4f};")
        print(f"Confusion matrix for subject {subject}:\n{subject_summary['confusion_matrix']}")

        for metric in metrics_list:
            scores[metric].append(subject_summary[metric])
        overall_confusion_matrix += subject_summary['confusion_matrix']

    print("\nFinal scores across all subjects:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize()}: {np.mean(scores[metric]):.4f}; ", end="")
    print(f"\nOverall confusion matrix:\n{overall_confusion_matrix}\n")

    log_results(log_path, scores, overall_confusion_matrix)

if __name__ == "__main__":
    main()
    print("Training and evaluation completed successfully.")