"""
Main to launch the da-rnn, note that many of the variables
below are in the configuration file.
"""

import json
import torch
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
import utils
from constants import device
import argparse
import os
from utilities.config_reader import read_config
from utilities.preprocessing import preprocess_data
from utilities.train_pred_necessities import da_rnn, train, predict

logger = utils.setup_log()
logger.info(f"Using computation device: {device}")


if __name__ == '__main__':

    # Fetch configuration file and execute variables
    parser = argparse.ArgumentParser(description='Configurations to run a model')
    parser.add_argument('-c', '--configfile', default=f"{os.getcwd()}/configs/config_basic.py",
                        help='file to read the config from')
    args = vars(parser.parse_args())
    read = read_config(args['configfile'])
    print(f'Config: {read}')
    locals().update(read)

    # Loading the data and preprocessing it
    raw_data = pd.read_csv(data_file, nrows=100 if debug else None)
    logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
    data, scaler = preprocess_data(raw_data, targ_cols)

    # Model creation, training and predicting
    config, model = da_rnn(data, logging=logger, n_targs=len(targ_cols), learning_rate=lr, **da_rnn_kwargs)
    iter_loss, epoch_loss = train(model, data, config, logging=logger, n_epochs=n_epochs, save_plots=save_plots)
    final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)

    # Showing the results
    plt.figure()
    plt.semilogy(range(len(iter_loss)), iter_loss)
    utils.save_or_show_plot("iter_loss.png", save_plots)

    plt.figure()
    plt.semilogy(range(len(epoch_loss)), epoch_loss)
    utils.save_or_show_plot("epoch_loss.png", save_plots)

    plt.figure()
    plt.plot(final_y_pred, label='Predicted')
    plt.plot(data.targs[config.train_size:], label="True")
    plt.legend(loc='upper left')
    utils.save_or_show_plot("final_predicted.png", save_plots)

    # Saving the model
    if save_model:
        with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
            json.dump(da_rnn_kwargs, fi, indent=4)
        joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
        torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
        torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))
