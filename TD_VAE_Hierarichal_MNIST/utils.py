import matplotlib.pyplot as plt

import torch
import math



## Architecture Utils
def sample_gaussian(mu, logvar):
  """
  Generate a gaussian sample given the mean and log variance.
  """
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return mu + std * eps

def log_normal_pdf(x, mean, logvar, eps=1e-6):
    """
    Obtain the log of the normal distribution
    """
    log_two_pi = torch.log(torch.tensor(2. * math.pi, device=x.device, dtype=x.dtype))
    return -0.5 * torch.sum(
        log_two_pi + logvar + ((x - mean) ** 2) / (torch.exp(logvar) + eps),
        dim=-1
    )

def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL Divergence between two multivariate gaussian distributions
    """
    var_q = torch.exp(logvar_q) #getting back the variance from the log of variance
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.sum(  #kl divergence of twogaussian distributions
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p)**2) / var_p - 1,
        dim=-1
    )


#Beta Scheduling
def linear_beta_schedule(epoch, max_epochs, max_beta=1.0, min_beta=0.0):
    return min_beta + (max_beta - min_beta) * (epoch / max_epochs)

def cyclic_beta_schedule(step, warmup_steps, beta_max = 1.0):
    factor = step // warmup_steps + 1
    if factor % 2 == 1: # odd means ramping up
       current_max = warmup_steps * factor
       normalized_step = 1 - (current_max - step) / warmup_steps
       beta = beta_max * normalized_step
    else:
        beta = beta_max

    return beta


#Plotting Funcs
def ax_standard(ax):

    ax.grid(True, alpha=0.5)
    ax.set_xlabel("Epoch")

def plot_results(history):

    line_color = "#f07167"

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout(pad=5)

    ax[0].plot(history["loss"], color = line_color)
    ax_standard(ax[0])
    ax[0].set_ylabel("Total Loss")


    ax[1].plot(history["kl_loss"], color = line_color)
    ax_standard(ax[1])
    ax[1].set_ylabel("KL Loss")


    ax[2].plot(history["reconstruction_loss"], color = line_color)
    ax_standard(ax[2])
    ax[2].set_ylabel("Reconstruction Loss")

def plot_training_vs_validation_loss(history):
    """
    Plots the total loss, reconstruction loss, and KL divergence loss
    for both training and validation.

    Parameters:
    history: Keras history object containing training and validation loss values per epoch.
    """
    plt.figure(figsize=(10, 5))

    # Plot total loss
   # plt.plot(history.history["loss"], label="Train Loss", color="#545f66")
   # plt.plot(history.history["val_loss"], label="Validation Loss", color="#829399", linestyle="dashed")

    # Plot reconstruction loss
    plt.plot(history.history["reconstruction_loss"], label="Train Reconstruction Loss", color="#8BE4CB")
    #plt.plot(history.history["val_reconstruction_loss"], label="Validation Reconstruction Loss", color="#DAFA9E", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Training vs. Validation Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot KL loss
    plt.plot(history.history["loss"], label="Total Loss", color="#b1cc74")
    #plt.plot(history.history["val_kl_loss"], label="Validation KL Loss", color="#DAFA9E", linestyle="dashed")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Training vs. Validation Kl_loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(history.history["kl_loss"], label="KL Loss", color="#b1cc74")
    #plt.plot(history.history["val_kl_loss"], label="Validation KL Loss", color="#DAFA9E", linestyle="dashed")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Training vs. Validation Kl_loss")
    plt.legend()
    plt.grid(True)
    plt.show()
