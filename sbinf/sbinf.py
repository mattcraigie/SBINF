import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import normflows as nf
import matplotlib.pyplot as plt
import copy

class InferenceModel(nn.Module):
    """
    A class for performing simulation-based inference with a conditional normalizing flow. Given data, the model is
    trained to learn the conditional posterior distribution. The model can then be used to sample from this posterior
    distribution given an observation (i.e. a conditioning for the probability distribution).

    Usage is as follows:
    1. Create an instance of the InferenceModel class
    2. Add simulated data and targets to the model using the `add_simulations` method
    3. Set up the conditional flow model using the `setup` method
    4. Train the model using the `train_model` method
    5. Sample from the posterior using the `sample` method
    """
    def __init__(self,
                 flow_type=None,
                 num_flow_layers=None,
                 num_hidden_units=None,
                 num_hidden_layers=None,
                 num_blocks=None
                 ):
        super(InferenceModel, self).__init__()

        self.flow_type = 'neural_spline_flow' if flow_type is None else flow_type
        self.model = None
        self.device = 'cpu'

        # training parameters
        self.optimizer = None
        self.train_losses = None
        self.val_losses = None

        # best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None

        # data parameters
        self.simulated_features = None
        self.simulated_targets = None
        self.num_features = None
        self.num_targets = None

        # setup method flags
        self.setup_mapping = {'neural_spline_flow': self._neural_spline_flow_setup,
                              'masked_autoregressive_flow': self._masked_autoregressive_flow_setup}

        # neural spline flow parameters
        self.num_flow_layers = num_flow_layers
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers
        self.num_blocks = num_blocks

        # other model parameters...

    def add_simulations(self, simulated_features, simulated_targets):
        """
        Add simulated data and targets to the model
        :param simulated_features: the data vector from the simulations (num_simulations x num_features)
        :param simulated_targets: the targets from the simulations (num_simulations x num_targets)
        :return:
        """

        self.simulated_features = simulated_features
        self.simulated_targets = simulated_targets
        self.num_features = simulated_features.shape[1]
        self.num_targets = simulated_targets.shape[1]

    def setup(self, device=None):
        try:
            self.setup_mapping[self.flow_type]()
        except KeyError:
            raise NotImplementedError("Flow type {} not implemented".format(self.flow_type))

        if device is not None:
            self.to(device)

    def train_model(self, num_epochs=100, lr=1e-4, batch_size=32):
        """
        Trains the conditional flow model on the simulated data. The model is trained to minimize the KL divergence
        :param num_epochs: the number of epochs to train for
        :param lr: the optimizer learning rate
        :param batch_size: the batch size for training
        :return:
        """

        assert self.simulated_features is not None, "Simulations must be added before training"
        assert self.model is not None, "Model must be setup before training"

        # set up the data loaders
        self._make_dataloaders(batch_size)

        self.train_losses = []
        self.val_losses = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print(f"Beginning training for {num_epochs} epochs\n")
        print_every = max(1, num_epochs // 10)

        for epoch in range(num_epochs):
            train_loss = self._train_step()
            val_loss = self._val_step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            if epoch % print_every == 0:
                print(f"Epoch {epoch+1}\t| Train Loss: {train_loss:.3e}\t| Val Loss: {val_loss:.3e}")

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.3e}")

    def load_best_model(self):
        """
        Loads the model state corresponding to the best validation loss observed during training.
        """
        if self.best_model_state is None:
            raise ValueError("No saved model state found. Have you run train_model()?")
        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        print(f"Loaded best model with val loss {self.best_val_loss:.3e}")

    def sample(self, num_samples, condition):
        """
        Samples the learned posterior distribution.
        :param condition: the condition for the posterior, i.e. the observational data vector (1 x num_features)
        :param num_samples: the number of samples to draw from the posterior
        :return: a tensor of samples from the posterior (num_samples x num_targets)
        """

        # I could place a prior by having a function that returns a prior probability, and rejecting samples based on
        # the prior probability. This would be a rejection sampler and might not be the most efficient option.
        # that being said, sampling is really quick that it probably doesn't matter. But is this stat still?

        return self.model.sample(num_samples, condition)

    def log_prob(self, condition, samples):
        """
        Computes the log probability for given features (conditions) and targets (samples)
        :param condition: the condition for the posterior, i.e. the observational data vector (1 x num_features)
        :param samples: the samples from the posterior (num_samples x num_targets)
        :return: the log probability of the posterior (num_samples x 1)
        """
        return self.model.log_prob(condition, samples)

    def to(self, device):
        """
        Moves the model to the specified device
        :param device: the device to move the model to
        :return:
        """
        assert self.model is not None, "Model must be setup before moving to device"
        super(InferenceModel, self).to(device)
        self.model.to(device)
        self.device = device

    def show_loss(self):
        """
        Plots the loss over training for the flow model
        :return:
        """
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def _neural_spline_flow_setup(self):
        # todo: pull out common parts in setup methods
        """
        Sets up the neural spline flow model
        :return:
        """

        # defaults
        if self.num_flow_layers is None:
            self.num_flow_layers = 2
        if self.num_hidden_units is None:
            self.num_hidden_units = 128
        if self.num_hidden_layers is None:
            self.num_hidden_layers = 2

        # flow settings
        latent_size = self.num_targets
        context_size = self.num_features

        # add nsf layers
        self.flows = []
        for i in range(self.num_flow_layers):
            self.flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, self.num_hidden_layers,
                                                                          self.num_hidden_units,
                                                                          num_context_channels=context_size)]
            self.flows += [nf.flows.LULinearPermute(latent_size)]

        # set base distribution
        self.base_distribution = nf.distributions.DiagGaussian(self.num_targets, trainable=False)

        # construct flow model
        self.model = nf.ConditionalNormalizingFlow(self.base_distribution, self.flows, self.simulated_targets)

    def _masked_autoregressive_flow_setup(self):
        # defaults
        if self.num_flow_layers is None:
            self.num_flow_layers = 2
        if self.num_hidden_units is None:
            self.num_hidden_units = 128
        if self.num_blocks is None:
            self.num_blocks = 2

        latent_size = self.num_targets
        context_size = self.num_features

        self.flows = []
        for i in range(self.num_flow_layers):
            self.flows += [nf.flows.MaskedAffineAutoregressive(latent_size, self.num_hidden_units,
                                                          context_features=context_size,
                                                          num_blocks=self.num_blocks)]
            self.flows += [nf.flows.LULinearPermute(latent_size)]

        # Set base distribution
        self.base_distribution = nf.distributions.DiagGaussian(self.num_targets, trainable=False)

        # Construct flow model
        self.model = nf.ConditionalNormalizingFlow(self.base_distribution, self.flows, self.simulated_targets)

    def _make_dataloaders(self, batch_size, val_fraction=0.2, shuffle_split=True, shuffle_dataloaders=True,
                          dataset_class=None):
        """
        Makes dataloaders for the simulated data
        :param batch_size: the batch size for the dataloaders
        :param val_fraction: the fraction of the data to use for validation
        :param shuffle_split: whether to shuffle the data before splitting into train and val
        :param shuffle_dataloaders: whether to shuffle the dataloaders
        :param dataset_class: the class to use for the dataset. Defaults to TensorDataset
        :return: train_loader, val_loader
        """
        # default dataset class
        if dataset_class is None:
            dataset_class = TensorDataset

        # determine split sizes
        num_data = self.simulated_features.shape[0]
        train_fraction = 1 - val_fraction
        num_train = int(num_data * train_fraction)

        # compute shuffle indices for split
        shuffle_idx = torch.randperm(num_data) if shuffle_split else torch.arange(num_data)
        train_idx = shuffle_idx[:num_train]
        val_idx = shuffle_idx[num_train:]

        # slice features and targets
        train_data = self.simulated_features[train_idx]
        train_targets = self.simulated_targets[train_idx]
        val_data = self.simulated_features[val_idx]
        val_targets = self.simulated_targets[val_idx]

        # create datasets
        train_dataset = dataset_class(train_data, train_targets)
        val_dataset = dataset_class(val_data, val_targets)

        # create dataloaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle_dataloaders)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle_dataloaders)

        return train_loader, val_loader

    def _train_step(self):
        """
        Performs a training step on the model
        :return:
        """

        # set model to train mode
        self.model.train()

        # train model
        train_loss = 0
        for batch_features, batch_targets in self.train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)

            # compute loss for the batch
            loss = self.model.forward_kld(batch_targets, batch_features)
            train_loss += loss.item()

            # backprop and perform Adam optimisation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return train_loss / len(self.train_loader)

    def _val_step(self):
        """
        Performs a validation step on the model
        :return:
        """

        # set model to eval mode
        self.model.eval()

        # evaluate model
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Compute loss for the batch
                loss = self.model.forward_kld(batch_targets, batch_features)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)
