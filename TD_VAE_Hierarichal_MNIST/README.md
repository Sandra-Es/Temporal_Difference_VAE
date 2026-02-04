### NNDL Project: Learning Multi-Scale Latent Dynamics with TD-VAE.

Members: 
- Sandra Elsa Sanjai (Matricola: 2113951)
- Jayanthvikram Chekkala (Matricola: 2106034)

#### Training the TD-VAE Model

The model can be trained by the simple CLI command as shown below. For changing arguments (arguments.py), we can use the 
CLI to select alternate values (--arg value).

```python
python -m train
```

#### Evaluating the Model on sample sequences

The model can be evaluated as below. Similar to the above case,
one can add arguments (to update the save_dir or load_chkpt_path) 
in the CLI.

Note that this requires a saved checkpoint path (produced when 
running train.py).

```python
python -m eval_rollout
```

#### Guide to modules

- arguments.py : Arguments to train and evaluate the model.
- train.py : The main module to train the TD-VAE model.
- eval_rollout.py : Evaluate the TD-VAE model using saved state-dict.
- data.py : Load the MNIST data as a moving sequence.
- arch_modules.py : Classes that make up the TD-VAE architecture.
- utils.py : General utils like plotting and experimental utils like gaussian sampling