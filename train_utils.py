import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import numpy as np

def normalize_eeg_data(X_train, X_test=None, method='standard'):
    """
    Normalizes EEG data either:
    - across trials per channel using statistics from the training set,
        applied to both train and test (if X_test is provided), OR
    - using only training data (if X_test is None).

    Supports two normalization methods:
    - 'standard': subtract mean and divide by standard deviation (default)
    - 'robust': subtract median and divide by interquartile range

    Parameters:
        X_train : np.ndarray of shape (trials, channels, samples)
            The training EEG data.
        X_test  : np.ndarray of shape (trials, channels, samples), optional
            The test EEG data to normalize using training statistics.
        method  : str, optional
            One of {'standard', 'robust'} indicating the normalization strategy.

    Returns:
        If X_test is provided:
            Tuple of (X_train_norm, X_test_norm)
        Else:
            X_train_norm
    """

    X_train_norm = np.empty_like(X_train)
    X_test_norm = np.empty_like(X_test) if X_test is not None else None

    for ch in range(X_train.shape[1]):
        data = X_train[:, ch, :].reshape(-1)

        if method == 'standard':
            mean = data.mean()
            std = data.std() + 1e-6
            X_train_norm[:, ch, :] = (X_train[:, ch, :] - mean) / std
            if X_test is not None:
                X_test_norm[:, ch, :] = (X_test[:, ch, :] - mean) / std

        elif method == 'robust':
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25 + 1e-6
            X_train_norm[:, ch, :] = (X_train[:, ch, :] - median) / iqr
            if X_test is not None:
                X_test_norm[:, ch, :] = (X_test[:, ch, :] - median) / iqr

    return (X_train_norm, X_test_norm) if X_test is not None else X_train_norm

def train_and_evaluate_model(
    X_train, y_train, 
    X_test, y_test, 
    model_class, batch_size=32, 
    epochs=10, learning_rate = 0.001, device=None
    ):
    """
    Trains and evaluates a neural network model for EEG classification.

    The model is instantiated from the provided "model_class", trained on the given
    training data, and evaluated on the test set. Returns test metrics and the
    confusion matrix.

    Parameters:
        X_train : np.ndarray
            Training data of shape (samples, channels, time).
        y_train : np.ndarray
            Training labels (integer class labels).
        X_test  : np.ndarray
            Test data of the same shape as X_train.
        y_test  : np.ndarray
            Test labels (integer class labels).
        model_class : class
            A PyTorch model class (not an instance) that defines the network.
        batch_size : int, optional
            Batch size used for training (default: 32).
        epochs : int, optional
            Number of training epochs (default: 10).
        learning_rate : float, optional
            Learning rate for the optimizer (default: 0.001).
        device : torch.device, optional
            Device to use for training and evaluation (CPU or CUDA). If None, auto-detects.

    Returns:
        test_acc : float
            Test set accuracy.
        test_prec : float
            Test set macro-averaged precision.
        test_rec : float
            Test set macro-averaged recall.
        test_f1 : float
            Test set macro-averaged F1 score.
        cm : np.ndarray
            Confusion matrix of shape (n_classes, n_classes).
    """


    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size
    )

    # Initialize model, criterion, optimizer
    model = model_class()
    model.to(device)
    
    #Xavier initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-2)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds_label = preds.argmax(dim=1)
            all_preds.append(preds_label.cpu())
            all_labels.append(yb.cpu())

        avg_loss = epoch_loss / len(train_loader.dataset)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_acc = accuracy_score(all_labels, all_preds)
        if epochs > 10 and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        elif epochs <= 10:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # Evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            preds_label = preds.argmax(dim=1).cpu()
            all_preds.append(preds_label)
            all_labels.append(yb)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, 
                                all_preds, 
                                average='macro', 
                                zero_division=0)
    test_rec = recall_score(all_labels, 
                            all_preds, 
                            average='macro', 
                            zero_division=0)
    test_f1 = f1_score(all_labels, 
                       all_preds, 
                       average='macro', 
                       zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return test_acc, test_prec, test_rec, test_f1, cm

def train_and_evaluate_loso(
    X_train, y_train, X_test, y_test, 
    model_class, batch_size=32, 
    epochs=10, learning_rate=0.001, device=None
):
    """
    Trains and evaluates a model in a Leave-One-Subject-Out (LOSO) setting.

    This function follows the same process as "train_and_evaluate_model", but is
    used in the LOSO procedure where one subject is held out entirely for testing
    and the rest are used for training.

    Parameters:
        X_train : np.ndarray
            Training data of shape (samples, channels, time).
        y_train : np.ndarray
            Training labels (integer class labels).
        X_test : np.ndarray
            Test data for the held-out subject.
        y_test : np.ndarray
            True labels for the held-out subject.
        model_class : class
            A PyTorch model class to instantiate and train.
        batch_size : int, optional
            Batch size for training (default: 32).
        epochs : int, optional
            Number of training epochs (default: 10).
        learning_rate : float, optional
            Learning rate for the optimizer (default: 0.001).
        device : torch.device, optional
            Device to use (e.g., torch.device("cuda") or "cpu"). Auto-detects if None.

    Returns:
        test_acc : float
            Accuracy on the held-out subject.
        test_prec : float
            Macro-averaged precision.
        test_rec : float
            Macro-averaged recall.
        test_f1 : float
            Macro-averaged F1 score.
        cm : np.ndarray
            Confusion matrix (n_classes x n_classes).
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                              batch_size=batch_size, 
                              shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), 
                             batch_size=batch_size)

    # Initialize model, criterion, optimizer
    model = model_class()
    model.to(device)

    #Xaver initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            all_preds.append(preds.argmax(dim=1).cpu())
            all_labels.append(yb.cpu())

        if epochs <= 10 or (epoch + 1) % 10 == 0:
            train_acc = accuracy_score(torch.cat(all_labels), 
                                       torch.cat(all_preds))
            print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            all_preds.append(preds.argmax(dim=1).cpu())
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm

def log_results(log_path, scores, overall_confusion_matrix):
    """
    Logs per-subject and overall evaluation results to the given file path.

    Parameters:
        log_path (str): Path to the log file.
        scores (dict): Dictionary with keys 'accuracy', 'precision', 'recall',
                       'f1', and 'confusion_matrix', each mapping to a list of values (1 per subject).
        overall_confusion_matrix (np.ndarray): Aggregated confusion matrix.
    """
    with open(log_path, "a") as f:
        for subject, acc, prec, rec, f1, cm in zip(
            range(1, len(scores['accuracy']) + 1),
            scores['accuracy'],
            scores['precision'],
            scores['recall'],
            scores['f1'],
            scores['confusion_matrix']
        ):
            f.write(f"Subject {subject} final accuracy: {acc:.4f}; ")
            f.write(f"precision: {prec:.4f}; ")
            f.write(f"recall: {rec:.4f}; ")
            f.write(f"F1 score: {f1:.4f};\n")
            f.write(f"Confusion matrix:\n{cm}\n")

        f.write("\nFinal scores across all subjects:\n")
        f.write(f"Accuracy: {np.mean(scores['accuracy']):.4f} ± {np.std(scores['accuracy']):.4f}; ")
        f.write(f"Precision: {np.mean(scores['precision']):.4f} ± {np.std(scores['precision']):.4f}; ")
        f.write(f"Recall: {np.mean(scores['recall']):.4f} ± {np.std(scores['recall']):.4f}; ")
        f.write(f"F1 Score: {np.mean(scores['f1']):.4f} ± {np.std(scores['f1']):.4f};\n")
        f.write(f"Overall confusion matrix:\n{overall_confusion_matrix}\n")