import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def NMSE(pred, true):
    return np.mean((pred - true) ** 2) / (np.mean(true**2) + 1e-10)

def augmented_features(x):
    real = x[..., 0]
    imag = x[..., 1]

    mod_square = real ** 2 + imag ** 2
    mod = np.sqrt(mod_square + 1e-8)
    mod_cubic = mod_square * mod

    out = np.concatenate([
        real[..., np.newaxis],
        imag[..., np.newaxis],
        mod[..., np.newaxis],
        mod_square[..., np.newaxis],
        mod_cubic[..., np.newaxis]
    ], axis=-1)

    return out

def plt_loss(train_losses, epochs, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plt_static(y_test_real, y_test_imag, y_pred_real, y_pred_imag, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_test_real[:500], label='True Real', linestyle='--')
    plt.plot(y_pred_real[:500], label='Pred Real', alpha=0.7)
    plt.title("Real Part Prediction")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_test_imag[:500], label='True Imag', linestyle='--')
    plt.plot(y_pred_imag[:500], label='Pred Imag', alpha=0.7)
    plt.title("Imaginary Part Prediction")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plt_animation(window_size, step, y_test_real, y_test_imag, y_pred_real, y_pred_imag, save_path):
    frames = (len(y_test_real) - window_size) // step
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    def update(frame, window_size, step, ):
        start = frame * step
        end = start + window_size

        axs[0].clear()
        axs[1].clear()

        axs[0].plot(y_test_real[start:end], label='True Real', linestyle='--')
        axs[0].plot(y_pred_real[start:end], label='Pred Real', alpha=0.7)
        axs[0].set_title("Real Part Prediction")
        axs[0].set_xlabel("Sample Index")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(y_test_imag[start:end], label='True Imag', linestyle='--')
        axs[1].plot(y_pred_imag[start:end], label='Pred Imag', alpha=0.7)
        axs[1].set_title("Imaginary Part Prediction")
        axs[1].set_xlabel("Sample Index")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend()
        axs[1].grid(True)

        fig.suptitle(f"Sample Range: {start} - {end}", fontsize=14)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)
    ani.save(save_path, writer="pillow")
    plt.close()