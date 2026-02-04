import torch
from torchvision.datasets import MNIST
from torchvision import transforms

import numpy as np
import random
import matplotlib.pyplot as plt

#Generating Data
class MovingMNIST(torch.utils.data.Dataset):
    def __init__(self, num_sequences=10000, sequence_length=20, image_size=28, speed=2, digit=99):
        self.mnist = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

        self.num_sequences = num_sequences          #Total number of sequences to return as data
        self.sequence_length = sequence_length      #Number of image instances in the moving sequence.
        self.image_size = image_size                #MNIST Image Size         
        self.speed = speed                          #Moving Speed in px/frame. If speed not in [1,5], then it is randomly sampled from [1,5].
        self.digit = digit

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        canvas_size = self.image_size
        frames = np.zeros((self.sequence_length, canvas_size, canvas_size), dtype=np.float32)   #Shape: (20, 28, 28)

        digit_img, _ = self.mnist[idx]       #Shape: (1, 28, 28)
        digit_img = digit_img[0]             #Shape: (28, 28)

        if self.speed not in range(0,5):     #Randomly sample speed if not specified
            self.speed = random.randint(1,4)

        # Random direction: -1 (left) or +1 (right)
        direction = random.choice([-1, 1]) #direction controlled by the training
        dx = direction * self.speed  # no of pixel left/right

        #Generate sequence roll
        for t in range(self.sequence_length):
            frames[t] = np.roll(digit_img, shift=t * dx, axis=1)

        frames = torch.tensor(frames).unsqueeze(1)  # shape: (T, 1, H, W)
        return frames

def show_sequence(frames):
    
    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames), 1.5))
    for i, ax in enumerate(axes):
        ax.imshow(frames[i, 0], cmap="gray")
        ax.axis("off")

    plt.show()

def time_interval():
    """
    Choose a random t1 in [1,19]
    Choose a random dt between 1 and 4
    Choose random direction +/-1
    Compute t2 = t1 + dt * direction
    Clip t2 to stay within [0,20]
    """
    t1 = random.randint(1, 18)
    dt = random.randint(1, 4)
    direction = random.choice([-1, 1])
    direction = 1  
    t2 = t1 + dt * direction

    # Border check: clip to [0, 20]
    if t2 < 0:
        t2 = 0
    elif t2 > 19:
        t2 = 19

    return t1, t2, direction
