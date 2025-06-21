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
from train_utils import normalize_eeg_data, train_and_evaluate_loso, log_results

"""
Script for running leave-one-subject-out (LOSO) cross-validation on the inner speech EEG dataset.

This script handles the full pipeline:
- Loads and preprocesses EEG data (with optional interval and electrode filtering)
- Trains a selected deep learning model on all subjects except one
- Tests the model on the held-out subject
- Repeats this process for each subject to obtain cross-subject generalization performance
- Logs accuracy, confusion matrices, and other evaluation results for analysis

Used to evaluate the model's ability to generalize across participants in the LOSO setting.
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

    #create folder to store confusion matrices images
    if not os.path.exists("confusion_matrices"):
        os.makedirs("confusion_matrices")

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
    "A1", "A2", "A4", "A5", "A8", "A9", "A10", "A15", "A19", "A20", "A22", "A23", "A24"]
    )
    #used for filtering electrodes
    channel_name_to_index = {name:idx 
                            for idx, name in enumerate(channel_names)}
    left_indices = [channel_name_to_index[ch] 
                    for ch in left_channels]

    #set hyperparameters
    lr = 0.0006 
    eps = 40
    b_s = 64

    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
    scores = {metric: [] for metric in metrics_list}
    overall_confusion_matrix = np.zeros((4, 4), dtype=int)

    X_all_subjects = []
    Y_all_subjects = []
    subject_ids = []

    # Used for logging the results.
    # Variables that are used in the code have a comment next to them.
    # The rest are just for logging.
    config = {
        "model": "LSTM-CNN hybrid",
        "condition": "INNER", # change to select other conditions["ALL", "INNER", "VIS", "PRON"]
        "paradigm": "LOSO",
        "k-fold": None,
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

        # Placeholder to append results later
        f.write("\nResults:\n")

    print("Starting training and evaluation...")
    # Load and store all subjects' data
    for subject in range(1, 11):
        X, Y = extract_data_from_subject(root_dir, subject, datatype="eeg")
        X, Y = filter_by_condition(X, Y, config.get("condition", "INNER"))
        
        # Time interval cropping
        if config.get("interval", "full") != "full":
            X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)
        # Channel selection
        if config.get("channels", "all") != "all":
            X = X[:, left_indices, :]

        X = normalize_eeg_data(X, method = config.get("normalization", "standard"))
        y_classes = Y[:, 1].astype(int) # Use only the second column (labels)
        
        X_all_subjects.append(X)
        Y_all_subjects.append(y_classes)
        subject_ids.extend([subject] * len(y_classes))

    print("\nFinished loading and normalizing data for all subjects.")

    X_all = np.concatenate(X_all_subjects, axis=0)
    Y_all = np.concatenate(Y_all_subjects, axis=0)
    subject_ids = np.array(subject_ids)

    unique_subjects = np.unique(subject_ids)

    n_c = X_all.shape[1]  # Number of channels
    n_s = X_all.shape[2]  # Number of samples

    for test_subj in unique_subjects:
        print(f"LOSO: Subject {test_subj}")
        #test for current subject
        train_mask = subject_ids != test_subj
        test_mask = subject_ids == test_subj

        X_train, Y_train = X_all[train_mask], Y_all[train_mask]
        X_test, Y_test = X_all[test_mask], Y_all[test_mask]

        acc, prec, rec, f1, cm = train_and_evaluate_loso(
            X_train, Y_train,
            X_test, Y_test,
            model_class=lambda: LSTM_CNN(num_channels=n_c, num_samples=n_s),
            learning_rate=lr,
            epochs=eps,
            batch_size=b_s,
        )

        print(
            f"Subject {test_subj} final accuracy: {acc:.4f}; "
            f"precision: {prec:.4f}; "
            f"recall: {rec:.4f}; "
            f"F1 score: {f1:.4f};\n"
            f"Confusion matrix:\n{cm}\n"
        )

        for metric, value in zip(metrics_list, [acc, prec, rec, f1, cm]):
            scores[metric].append(value)

        overall_confusion_matrix += cm

    print("\nFinal scores across all subjects:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize()}: {np.mean(scores[metric]):.4f}; ", end="")
    print(f"\nOverall confusion matrix:\n{overall_confusion_matrix}\n")

    log_results(log_path, scores, overall_confusion_matrix)

if __name__ == "__main__":
    main()
    print(f"Training and evaluation completed.")