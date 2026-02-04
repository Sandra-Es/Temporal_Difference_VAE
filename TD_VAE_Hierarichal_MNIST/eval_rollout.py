import os
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from arguments import get_args
from data import MovingMNIST
from arch_modules import *


def main():

    #Parse parameters
    args = get_args()

    batch_size = args.batch_size
    learn_rate = args.learn_rate

    latent_dim = args.latent_dim
    belief_dim = args.belief_dim
    num_sequences = args.num_sequences
    sequence_length = args.len_sequence
    speed = args.mnist_speed
    digit = args.mnist_digit

    load_chkpt_path = args.load_chkpt_path
    assert os.path.exists(load_chkpt_path), "Please provide a valid checkpoint path!"

    save_dir = f"{args.save_dir}/rollout"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #Rollout Function
    def tdvae_rollout(vae, x_sample, t1=10, t_end=15, device='cuda'):
        """
        Performs autoregressive rollout from t1 to t_end for a single batch item.

        Args:
            vae: Trained TD-VAE model
            x: Input tensor of shape [B, T, 1, 32, 32]
            t1: Starting timestep for rollout
            t_end: Ending timestep for rollout
            batch_idx: Which index from the batch to visualize
            device: 'cuda' or 'cpu'

        Returns:
            List of predicted frames [ (1, 32, 32), ... ] for the selected batch_idx
        """
        vae.eval()
        with torch.no_grad():

            T, C, H, W = x_sample.shape
            data = x_sample.view(T, -1).unsqueeze(0)  # shape: [1, T, 784]

            #Preprocess Data
            processor = PreProcess(784, 784).to(device)
            preprocessed_data = processor(data)  # shape: [1, T, 784]

            #Get belief states using LSTM
            bt = vae.beliefs(preprocessed_data)

            # Get z_t1 by passing through the layer 2 and then layer 1
            mu2, logvar2 = vae.belief_layer2(bt[:, t1 - 1, :])
            zt2_layer2 = sample_gaussian(mu2, logvar2)
            mu1, logvar1 = vae.belief_layer1(torch.cat((bt[:, t1 - 1, :], zt2_layer2), dim=-1))
            zt2_layer1 = sample_gaussian(mu1, logvar1)
            zt1 = torch.cat((zt2_layer1, zt2_layer2), dim=-1)

            xt2_list = []

            for dt in range(1, t_end - t1 + 1):
                
                #Predict z_t2 by passing z_t1 through the transition layers (2, then 1)
                mu_trans2, logvar_trans2 = vae.transition_layer2(zt1)
                zt2_layer2 = sample_gaussian(mu_trans2, logvar_trans2)

                mu_trans1, logvar_trans1 = vae.transition_layer1(torch.cat((zt1, zt2_layer2), dim=-1))
                zt2_layer1 = sample_gaussian(mu_trans1, logvar_trans1)

                zt2 = torch.cat((zt2_layer1, zt2_layer2), dim=-1)

                #Decoder. Provides x_t2
                logits = vae.decoder(zt2)
                probs = torch.sigmoid(logits)
                xt2 = probs.view(1, 28, 28)

                # Append only the selected index
                xt2_list.append(xt2)  # shape [1, 28, 28]

                zt1 = zt2

            return xt2_list

    def to_numpy(img_tensor):
        return img_tensor.detach().cpu().numpy()

    def plot_predictions_with_gt(x_batch, t1, xt2_list, save_path):

        total_cols = 20     #Sequence Length
        fig, axes = plt.subplots(2, total_cols, figsize=(2 * total_cols, 4))

        # Row 1: Ground Truth
        for i in range(total_cols):

            axes[0, i].imshow(to_numpy(x_batch[i, 0]), cmap='gray')

            #Title according to time stamp
            if i < t1: axes[0, i].set_title(f"GT {i}")
            else:  axes[0, i].set_title(f"GT t+{i - t1 + 1}")

            axes[0, i].axis('off')

        # Row 2: Predictions
        for j, pred_frame in enumerate(xt2_list):
            col_idx = t1 + j
            axes[1, col_idx].imshow(to_numpy(pred_frame[0]), cmap='gray')
            axes[1, col_idx].set_title(f"Pred t+{j+1}")
            axes[1, col_idx].axis('off')

        #Don't plot boundaries for emtpy spaces
        for i in range(t1):
            axes[1, i].axis('off')

        for i in range(t1+len(xt2_list), total_cols):
            axes[1, i].axis('off')

        plt.tight_layout()

        fig.savefig(save_path)


    # Generate data
    X_train = MovingMNIST(num_sequences=num_sequences,
                        sequence_length=sequence_length, 
                        speed=speed, 
                        digit=digit)

    # # Create data loaders
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

    #Initializing TD-VAE Model
    vae = TDVAE_hierachical(Distribution_Generator, 
                            BeliefLSTM,
                            PreProcess, 
                            Decoder,
                            784, 784, 
                            belief_dim=belief_dim, latent_dim_1=latent_dim, latent_dim_2=latent_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learn_rate)

    #Load Checkpoint Weights
    chkpt = torch.load(load_chkpt_path)
    chkpt_epoch = chkpt["epoch"]
    vae.load_state_dict(chkpt["model_state_dict"])
    optimizer.load_state_dict(chkpt["optimizer_state_dict"])

    vae.to(device)
    vae.eval()

    for batch in train_loader:
        x = batch.to(device)
        break

    #Run rollout
    for batch_idx in np.arange(5):
        print(f"Rollout for Batch {batch_idx}")

        # x_batch = X_train[batch_idx].to(device)
        x_batch = x[batch_idx]

        preds = tdvae_rollout(vae, x_batch, 
                            t1 = 10, t_end = 15,
                            device = device)
        
        plot_predictions_with_gt(x_batch,  
                                t1 = 10, 
                                xt2_list = preds,
                                save_path = f"{save_dir}/epoch_{chkpt_epoch}_batch_{batch_idx}.png")
        

if __name__ == "__main__":
    main()