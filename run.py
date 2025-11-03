import os
import torch
import torch.optim as optim
from utils.tools import plt_loss, plt_animation, plt_static
from dataset.dataloader import dataloader
from model.ARVMCTN import ARVMCTN
from exp_train.exp_train import train


# ===================== Configuration =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
task = "Multi"
checkpoint_path = os.path.join("checkpoints", task)
os.makedirs(checkpoint_path, exist_ok=True)
print(f"------------------------------- {task} -------------------------------------")


# ===================== Load Datasets =====================
BANDS = ["D", "H", "W"]
GAINS = ["1", "2", "4"]
QAMS = ["16", "64"]

dataset_input_list = []
dataset_output_list_list = []
param_list = []

for i, band in enumerate(BANDS):
    for j, g in enumerate(GAINS):
        for k, qam in enumerate(QAMS):
            input_path = (
                f"dataset/DATA_W_D_H_band/Baseband_input_output/input/"
                f"{band}_band/PA_input_BS_{g}G_60G_{qam}QAM_{band}.mat"
            )
            output_paths = [
                f"dataset/DATA_W_D_H_band/Baseband_input_output/output/"
                f"{band}_band/PA_baseband_{g}G_{qam}QAM_{band}_{idx}.mat"
                for idx in range(1, 6)
            ]

            dataset_input_list.append(input_path)
            dataset_output_list_list.append(output_paths)
            param_list.append([i, j, k])

# Create dataloaders
train_loader, test_loader, y_scaler = dataloader(
    dataset_input_list,
    dataset_output_list_list,
    param_list,
    window_size=11,
    batch_size=32,
    device=device,
)


# ===================== Model and Optimizer =====================
model = ARVMCTN(in_dim=5, embed_dim=16, kernel_sizes=(3, 5, 7),
                 stride=1, num_heads=2, num_layers=1, out_dim=2,
                 dropout=0.1, seq_len=11, param_dim=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)


# ===================== Training =====================
epochs = 20
train_losses, min_mse, y_pred, y_true = train(
    epochs,
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    y_scaler,
    log_path=os.path.join(checkpoint_path, "log.txt"),
    pth_path=os.path.join(checkpoint_path, "model.pth"),
    eval_every=1
)


# ===================== Plot Loss Curve =====================
plt_loss(train_losses, epochs, os.path.join(checkpoint_path, "loss_curve.png"))


# ===================== Visualization =====================
y_test_real = y_true[:, 0]
y_test_imag = y_true[:, 1]
y_pred_real = y_pred[:, 0]
y_pred_imag = y_pred[:, 1]

# Static plot
plt_static(
    y_test_real,
    y_test_imag,
    y_pred_real,
    y_pred_imag,
    save_path=os.path.join(checkpoint_path, "static_prediction_plot.png"),
)

# Animation
window_size = 500   # Number of samples per frame
step = 10           # Sliding step
plt_animation(
    window_size,
    step,
    y_test_real,
    y_test_imag,
    y_pred_real,
    y_pred_imag,
    save_path=os.path.join(checkpoint_path, "prediction_animation.gif"),
)
