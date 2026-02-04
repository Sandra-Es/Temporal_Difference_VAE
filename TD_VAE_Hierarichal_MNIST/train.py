import os
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from arguments import get_args
from data import MovingMNIST
from arch_modules import *


def main():

    #Parse parameters
    args = get_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    latent_dim = args.latent_dim
    belief_dim = args.belief_dim
    num_sequences = args.num_sequences
    sequence_length = args.len_sequence
    speed = args.mnist_speed
    digit = args.mnist_digit

    load_chkpt_path = args.load_chkpt_path
    if os.path.exists(load_chkpt_path): load_chkpt = True       #If state_dict checkpoint exists, load it
    else: load_chkpt = False

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Generate data
    X_train = MovingMNIST(num_sequences=num_sequences,
                        sequence_length=sequence_length, 
                        speed=speed, 
                        digit=digit)

    # Create data loader
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

    #Initializing TD-VAE Model
    vae = TDVAE_hierachical(Distribution_Generator, 
                            BeliefLSTM,
                            PreProcess, 
                            Decoder,
                            784, 784,                                               #Input and Preprocessed Sizes. Since MNIST data is shaped (28, 28), the flattened data will be (768,)
                            belief_dim=belief_dim,                                  
                            latent_dim_1=latent_dim, latent_dim_2=latent_dim)  

    optimizer = torch.optim.Adam(vae.parameters(), lr=learn_rate)

    #If checkpoint exists, then load parameters to model
    if load_chkpt:
        chkpt = torch.load(load_chkpt_path)
        vae.load_state_dict(chkpt["model_state_dict"])
        optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        start_epoch = chkpt["epoch"] + 1
    else:
        start_epoch = 0


    vae.to(device)        # move to GPU/CPU
    vae.train()     
    vae.reset_loss_trackers()

    #Save file
    save_loss_path = f"{save_dir}/loss_per_epoch.txt"           #Track loss info per epoch
    save_temp_dict_path = f"{save_dir}/checkpoint.pt"           #Temporary checkpoint (in case of interruptions)

    #Training loop
    loss_history = []
    reconstruction_history = []
    kl_history = []

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch in tqdm(train_loader):
            x = batch.to(device)
            #beta = linear_beta_schedule(epoch, num_epochs)
            #beta = cyclic_beta_schedule(epoch, warmup_steps = 20)
            metrics = vae.train_step(x, optimizer, device, beta=1)

            total_loss += metrics['loss']
            total_rec += metrics['reconstruction_loss']
            total_kl += metrics['kl_loss']
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_rec  = total_rec / num_batches
        avg_kl   = total_kl / num_batches

        loss_history.append(avg_loss)
        reconstruction_history.append(avg_rec)
        kl_history.append(avg_kl)

        #Save the loss terms
        with open(save_loss_path, "a") as f:
            f.write(f"{epoch}, {avg_loss}, {avg_rec}, {avg_kl}\n")

        #Save temporary checkpoint for state dict
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_temp_dict_path)

        #Save checkpoint with state dict every 50 epochs
        if (epoch % 50 == 0) and (epoch > 0):
            save_state_dict_path = f"{save_dir}/checkpoint_{epoch}.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_state_dict_path)


        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Recon: {avg_rec:.4f} | KL: {avg_kl:.4f}")



if __name__ == "__main__":
    main()