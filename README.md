# Temporal_Difference_VAE

This repository contains an implementation and experimental evaluation of **Temporal Difference Variational Autoencoders (TD-VAE)** for learning **latent dynamics at multiple temporal scales**. The project reproduces and extends the framework proposed by Gregor et al. (2019), with a focus on **temporal abstraction, smoothing, and long-horizon prediction**.

## Navigating the Repo

We experiment with four different variations, each with its own directory within this repo:

- **TD_VAE_Harmonic**: Skip-state prediction for noisy Harmonic Oscillator using Hierarchical architecture of TD-VAE
- **TD_VAE_Hierarchical_MNIST**: Sequence prediction for Moving MNIST with Hierarchical architecture of TD-VAE.
- **TD_VAE_MNIST_CNN** : Sequence prediction for Moving MNIST using classical CNN-based TD-VAE
- **TD_VAE_MNIST_flat**: Sequence predictino for Moving MNIST using baseline MLP-based TD-VAE 

## Overview

Traditional sequential VAEs (e.g., VRNNs, SRNNs) rely on Markovian assumptions, which limit their ability to capture long-range temporal dependencies. TD-VAEs address this limitation by learning **temporally abstract latent states** that support inference over both past and future observations.

In this project, we implement and evaluate:
- A **standard TD-VAE**
- A **hierarchical TD-VAE** with multi-level latent variables
- TD-VAE variants with and without CNN preprocessing

The models are evaluated on both **low-dimensional dynamical systems** and **high-dimensional visual sequences**.

## Datasets

### Noisy Harmonic Oscillator
Synthetic time-series data generated from a harmonic oscillator with additive Gaussian noise.
- Sequence length: 200
- Noise level: σ = 0.1
- Task: Long-horizon prediction with temporal skips (Δt = 20, 100)

### Moving MNIST
Custom-generated Moving MNIST sequences with horizontally moving digits.
- Sequence length: 20 frames
- Image size: 32 × 32 (grayscale)
- Tasks:
  - Single-digit sequence prediction
  - Multi-digit sequence prediction

## Model Architecture

Each TD-VAE variant consists of four core components:
- **Encoder**: Infers stochastic latent variables and deterministic belief states
- **Smoothing Network**: Refines past latent states using future observations
- **Transition Network**: Models latent evolution across variable temporal gaps
- **Decoder**: Reconstructs observations from latent states

The **hierarchical TD-VAE** introduces a top-down latent structure, enabling separation of **global (slow)** and **local (fast)** dynamics.

## Training Objective

The training loss combines:
- Reconstruction loss  
  - Binary cross-entropy (Moving MNIST)
  - Mean squared error (harmonic oscillator)
- KL-like consistency terms enforcing temporal coherence
- **β-scheduling** (linear and cyclic) to balance reconstruction fidelity and latent regularization

## Results

### Noisy Harmonic Oscillator
- Accurate recovery of oscillatory dynamics
- Stable frequency prediction for short temporal skips (Δt = 20)
- Gradual degradation for long-horizon prediction (Δt = 100)

### Moving MNIST
- CNN-based preprocessing was unstable and hard to scale
- Flattened-input TD-VAE scaled better for single-digit sequences
- **Hierarchical TD-VAE achieved the best performance**, with faster convergence and improved sequence generation