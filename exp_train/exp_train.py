import numpy as np
import torch
from torch import nn
from utils.tools import NMSE

def train(epochs, model, optimizer, scheduler, train_loader, test_loader, y_scaler, log_path, pth_path, device="cuda", eval_every=5):
    log_file = open(log_path, "w")
    criterion = nn.MSELoss()
    best_y_pred = []
    best_y_true = []
    train_losses = []
    best_mse = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y, p in train_loader:
            batch_x, batch_y, p = batch_x.to(device), batch_y.to(device), p.to(device)
            optimizer.zero_grad()
            output = model(batch_x, p)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()
        if (epoch + 1) % eval_every == 0:

            train_msg = f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.5f}\n"
            print(train_msg.strip())
            log_file.write(train_msg)

            model.eval()
            y_pred_list = []
            y_true_list = []
            with torch.no_grad():
                for batch_x, batch_y, p in test_loader:
                    batch_x, batch_y, p = batch_x.to(device), batch_y.to(device), p.to(device)
                    output = model(batch_x, p)
                    y_pred_list.append(output.cpu().numpy())
                    y_true_list.append(batch_y.cpu().numpy())
            y_pred = y_scaler.inverse_transform(np.vstack(y_pred_list))
            y_true = y_scaler.inverse_transform(np.vstack(y_true_list))
            nmse = NMSE(y_pred, y_true)
            nmsedb = 10 * np.log10(nmse if nmse > 1e-10 else 1e-10)

            test_msg = f"Test NMSE: {nmsedb:.5f}, --{nmse}\n"
            print(test_msg.strip())
            log_file.write(test_msg)

            if best_mse > nmsedb:
                best_mse =  nmsedb
                best_y_pred = y_pred
                best_y_true = y_true
                torch.save(model, pth_path)
    log_file.close()
    return train_losses, best_mse, best_y_pred, best_y_true