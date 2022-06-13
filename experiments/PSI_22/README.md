An example directory consisting of training, plotting, and hyperparameter searching for the proposed model.
Each relevant file is outlined below.

# run.py: the training script

This is a training script, which results in a saved model `model_results/` and logs in `tb_logs/`

If you have tensorboard installed, you can visualize resulting losses over the training cycle, via: `tensorboard --logdir tb_logs`.

You can change hyperparameters.

There might be argparsing later, if I am not lazy.

# plotting.py: making pretty pictures

This is a script that takes the trained model (by running `run.py` above), and 
