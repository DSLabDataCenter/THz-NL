import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils.tools import augmented_features


def dataloader(
    dataset_input_list: list,
    dataset_output_list_list: list,
    param_list: list,
    window_size: int = 11,
    batch_size: int = 32,
    device: str = "cuda",
):

    X_train, y_train, X_test, y_test = [], [], [], []
    param_train, param_test = [], []

    # ===================== Data Loading =====================
    for idx, input_path in enumerate(dataset_input_list):
        output_paths = dataset_output_list_list[idx]
        param = param_list[idx]

        # --- Load input signal (complex -> real+imag concatenation) ---
        input_data = scipy.io.loadmat(input_path)["x"]
        input_data = np.hstack([input_data.real, input_data.imag])

        # --- Load multiple output signals ---
        outputs = []
        for output_path in output_paths:
            data = scipy.io.loadmat(output_path)["PA_baseband"]
            outputs.append(np.hstack([data.real, data.imag]))

        # Split first 4 outputs for training, last one for testing
        train_outputs = outputs[:4]
        test_output = outputs[-1]

        # --- Construct sliding windows ---
        num_samples = len(input_data) - window_size
        for output in train_outputs:
            for i in range(num_samples):
                X_train.append(input_data[i:i + window_size])
                y_train.append(output[i + window_size - 1])

        for i in range(num_samples):
            X_test.append(input_data[i:i + window_size])
            y_test.append(test_output[i + window_size - 1])

        # --- Expand parameters ---
        param_train.extend(np.tile(np.array(param), (num_samples * 4, 1)))
        param_test.extend(np.tile(np.array(param), (num_samples, 1)))

    # ===================== Convert to NumPy Arrays =====================
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    param_train, param_test = np.array(param_train), np.array(param_test)

    # ===================== Data Augmentation =====================
    X_train = augmented_features(X_train)
    X_test = augmented_features(X_test)

    # ===================== Normalization =====================
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    X_train_norm = x_scaler.fit_transform(X_train_flat)
    X_test_norm = x_scaler.transform(X_test_flat)
    y_train_norm = y_scaler.fit_transform(y_train)
    y_test_norm = y_scaler.transform(y_test)

    # Reshape back to original 3D form
    X_train = X_train_norm.reshape(X_train.shape)
    X_test = X_test_norm.reshape(X_test.shape)

    # ===================== Convert to PyTorch Tensors =====================
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32, device=device)
    param_train_tensor = torch.tensor(param_train, dtype=torch.float32, device=device)
    param_test_tensor = torch.tensor(param_test, dtype=torch.float32, device=device)

    # ===================== Create Datasets and Loaders =====================
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, param_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, param_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, y_scaler
