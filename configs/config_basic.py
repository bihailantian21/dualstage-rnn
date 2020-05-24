"""
Config file
"""
import os

config_dict = {
    'lr': 0.001,
    'data_file': os.path.join("data", "aapl.csv"),
    'save_plots': True,
    'save_model': True,
    'debug': False,
    'n_epochs': 1000,
    'targ_cols': ("Target",),
    'da_rnn_kwargs': {"batch_size": 128, "T": 5}
}
