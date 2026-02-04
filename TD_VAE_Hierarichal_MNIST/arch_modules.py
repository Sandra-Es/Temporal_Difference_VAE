import torch
import torch.nn as nn

from utils import *
from data import time_interval


class BeliefLSTM(nn.Module):
    def __init__(self, input_dim=1, belief_dim=50):
        super(BeliefLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=belief_dim, 
                            batch_first=True)

    def forward(self, x):
        b, _= self.lstm(x)
        return b                #(batch_size, sequence_length, belief_dim) = (512, 20, 50)

class PreProcess(nn.Module):
  """ 
  The pre-processing layer for an MNIST image
  """
  def __init__(self, input_size=1024, processed_x_size=1024):
    super(PreProcess, self).__init__()
    self.input_size = input_size
    self.fc1 = nn.Linear(input_size, processed_x_size)
    self.fc2 = nn.Linear(processed_x_size, processed_x_size)

  def forward(self, input):
    t = torch.relu(self.fc1(input))
    t = torch.relu(self.fc2(t))
    return t

class Distribution_Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim = 50, latent_dim = 8):
        """
        input_dim: dimensionality of the input context (e.g., b_t, z_t2, etc.)
        hidden_dim: dimensionality of the hidden layer
        output_dim: dimensionality of the output (i.e., z size)
        """
        super(Distribution_Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # W1 and B1
        self.fc2 = nn.Linear(input_dim, hidden_dim)  # W2 and B2
        #self.fc3 = nn.Linear(hidden_dim, 2 * latent_dim)  # W3 and B3 (outputs both mu and log sigma)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        x: context input (batch_size, input_dim)
        returns: mu, log_sigma of shape (batch_size, output_dim)
        """
        t1 = torch.tanh(self.fc1(x))        # W1x + B1 → tanh
        t2 = torch.sigmoid(self.fc2(x))     # W2x + B2 → sigmoid
        t = t1 * t2                         # element-wise product
        #out = self.fc3(t)                   # W3·(t) + B3 → outputs both mu and log sigma
        #mu, log_sigma = torch.chunk(out, 2, dim=-1)

        mu = self.fc_mu(t)
        log_sigma = self.fc_logsigma(t)

        return mu, log_sigma

class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        logits = (self.fc3(t))
        return logits
    
class TDVAE_hierachical(nn.Module):
    def __init__(self, dist_gen, BeliefLSTM, preprocess, decoder, input_size, processed_x_size, belief_dim=50, latent_dim_1=8, latent_dim_2=8):
        super().__init__()
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2

        self.preprocess = preprocess(input_size, processed_x_size)

        self.beliefs = BeliefLSTM(processed_x_size)

        self.belief_layer2 = dist_gen(belief_dim, 50, latent_dim_2)
        self.belief_layer1 = dist_gen(belief_dim + latent_dim_2, 50, latent_dim_1)

        self.smoothing_layer2 = dist_gen(belief_dim + latent_dim_1 + latent_dim_2, 50, latent_dim_1)
        self.smoothing_layer1 = dist_gen(belief_dim + latent_dim_2 + latent_dim_1 + latent_dim_2, 50, latent_dim_1)

        self.transition_layer2 = dist_gen(latent_dim_2 + latent_dim_1, 50, latent_dim_1)
        self.transition_layer1 = dist_gen(latent_dim_2 + latent_dim_1 +latent_dim_2, 50, latent_dim_1)

        self.decoder = decoder(latent_dim_1 + latent_dim_2, 200,input_size)

        self.total_loss = 0.0
        self.reconstruction_loss = 0.0
        self.kl_loss = 0.0

    def reset_loss_trackers(self):
        self.total_loss = 0.0
        self.reconstruction_loss = 0.0
        self.kl_loss = 0.0

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data, optimizer, device, beta):
        """
        The loss terms are calculated with reference to the TD-VAE paper, corresponding to equation (6) and (8).
        Reference Link: https://arxiv.org/abs/1806.03107
        """

        data = data.to(device)
        t1, t2, drxn= time_interval()

        #Preprocess Image Data
        B, T, C, H, W = data.shape
        data = data.view(B, T, -1)
        original_data = data.clone()
        preprocessed_data = self.preprocess(data)

        #Encoder. Obtain z at time t2, passing through layer 2 and then layer 1 (hierarchical).
        bt = self.beliefs(preprocessed_data)            

        mu2, logvar2 = self.belief_layer2(bt[:, t2, :])  # shape: [batch, latent_dim_2]
        zt2_layer2 = sample_gaussian(mu2, logvar2)                                                                  #Term 1

        mu1, logvar1 = self.belief_layer1(torch.cat([bt[:, t2, :], zt2_layer2], dim=-1))
        zt2_layer1 = sample_gaussian(mu1, logvar1)                                                                  #Term 2

        zt2 = torch.cat([zt2_layer1, zt2_layer2], dim =-1)                                                          #Term 3

        mut1_layer2, logvart1_layer2 = self.belief_layer2(bt[:, t1, :])  # shape: [batch, latent_dim_2]             #Term 4


        #Smoothing (backward prediction). 
        # Predict z at time t1, conditioned on z_t2, again going through layer 2 and then layer 1. 
        mu_smooth_layer2, logvar_smooth_layer2 = self.smoothing_layer2(torch.cat([bt[:, t1, :],zt2], dim=-1))
        zt1_layer2_smooth = sample_gaussian(mu_smooth_layer2, logvar_smooth_layer2)                                 #Term 5

        mut1_layer1, logvart1_layer1 = self.belief_layer1(torch.cat([bt[:, t1, :], zt1_layer2_smooth], dim=-1))     #Term 6

        mu_smooth_layer1, logvar_smooth_layer1 = self.smoothing_layer1(
            torch.cat([bt[:, t1, :], zt2, zt1_layer2_smooth], dim=-1)
            )
        zt1_layer1_smooth = sample_gaussian(mu_smooth_layer1, logvar_smooth_layer1)                                 #Term 7

        zt1 = torch.cat([zt1_layer1_smooth, zt1_layer2_smooth ], dim = -1)                                          #Term 8

        #Transition. 
        mu_trans_layer2 , logvar_trans_layer2 = self.transition_layer2(zt1)                                         #Term 9
        mu_trans_layer1 , logvar_trans_layer1 = self.transition_layer1(torch.cat([zt1,zt2_layer2], dim = -1))       #Term 10

        #Decoder.
        reconstruction = self.decoder(zt2)
        target = original_data[:, t2, :]          # shape [B, 768]

        #BCE Loss
        bce = nn.BCEWithLogitsLoss(reduction='none')
        Lx = bce(reconstruction, target).sum(dim=1)  # sum over input dimensions

        #Calculating losses now
        L1 = kl_divergence(zt1_layer2_smooth, logvar_smooth_layer2, mut1_layer2, logvart1_layer2)
        L2 = kl_divergence(zt1_layer1_smooth, logvar_smooth_layer1, mut1_layer1, logvart1_layer1)
        L3 = log_normal_pdf(zt2_layer2, mu2, logvar2) - log_normal_pdf(zt2_layer2, mu_trans_layer2, logvar_trans_layer2)
        L4 = log_normal_pdf(zt2_layer1, mu1, logvar1) - log_normal_pdf(zt2_layer1, mu_trans_layer1, logvar_trans_layer1)

        total_loss = (Lx + beta*(L1 + L2 + L3 + L4)).mean()
        reconstruction_loss = Lx.mean()
        kl_loss = (L1).mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # === Return metrics ===
        return {
            'loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item()
        }
