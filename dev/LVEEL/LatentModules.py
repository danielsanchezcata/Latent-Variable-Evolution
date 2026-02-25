import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaTCVAE(LatentModule):
    """
    Beta-TC VAE with decomposed KL divergence.
    Loss = Recon + alpha*MI + beta*TC + gamma*KL
    Setting alpha=beta=gamma=1 gives standard VAE.
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dim=32, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__(input_dim, latent_dim)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Encoder outputs mu and logvar
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        """Returns mu (for inference/evolution use)."""
        h = self.encoder_net(x)
        return self.fc_mu(h)
    
    def encode_dist(self, x):
        """Returns mu and clamped logvar for training."""
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-20, max=2)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder_net(z)
    
    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def log_density_gaussian(self, x, mu, logvar):
        """Log density of Gaussian."""
        norm = -0.5 * (np.log(2 * np.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 / torch.exp(logvar))
        return log_density
    
    def loss(self, x, x_recon, mu=None, logvar=None, z=None, return_parts=False, **kwargs):
        """
        Compute BetaTCVAE loss with decomposed KL.
        Uses minibatch weighted sampling for TC estimation.
        
        Args:
            return_parts: if True, return dict with individual loss terms (unscaled)
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (mean over all elements)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Log q(z|x) - log density under encoder
        log_qz_x = self.log_density_gaussian(z, mu, logvar).sum(dim=1)
        
        # Log p(z) - log density under prior N(0,1)
        zeros = torch.zeros_like(z)
        log_pz = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)
        
        # Log q(z) - marginal (minibatch weighted sampling)
        _logqz = self.log_density_gaussian(
            z.unsqueeze(1),
            mu.unsqueeze(0),
            logvar.unsqueeze(0)
        )
        
        log_qz = torch.logsumexp(_logqz.sum(dim=2), dim=1) - np.log(batch_size)
        log_qz_product = (torch.logsumexp(_logqz, dim=1) - np.log(batch_size)).sum(dim=1)
        
        # Decomposed KL terms (unscaled)
        mi_loss = (log_qz_x - log_qz).mean()
        tc_loss = (log_qz - log_qz_product).mean()
        kl_loss = (log_qz_product - log_pz).mean()
        
        # Total loss (scaled)
        total_loss = recon_loss + self.alpha * mi_loss + self.beta * tc_loss + self.gamma * kl_loss
        
        if return_parts:
            return {
                'total': total_loss,
                'recon': recon_loss,
                'mi': mi_loss,
                'tc': tc_loss,
                'kl': kl_loss
            }
        return total_loss
    
    def fit(self, dataset, epochs=100, batch_size=32, lr=1e-3, device='cpu', verbose=True, val_split=0.2):
        """
        Training loop with validation.
        
        Args:
            dataset: list of numpy arrays
            epochs: number of training epochs
            batch_size: batch size
            lr: learning rate
            device: 'cpu' or 'cuda'
            verbose: print loss during training
            val_split: fraction for validation set
        
        Returns:
            dict with losses per epoch
        """
        self.to(device)
        
        # Convert dataset to tensor and split
        data = torch.tensor(np.array(dataset), dtype=torch.float32)
        n_val = int(len(data) * val_split)
        n_train = len(data) - n_val
        
        indices = torch.randperm(len(data))
        train_data = data[indices[:n_train]]
        val_data = data[indices[n_train:]]
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        history = {
            'train_total': [], 'train_recon': [], 'train_mi': [], 'train_tc': [], 'train_kl': [],
            'val_total': [], 'val_recon': [], 'val_mi': [], 'val_tc': [], 'val_kl': []
        }
        
        for epoch in range(epochs):
            # Training
            self.train()
            epoch_losses = {'total': 0, 'recon': 0, 'mi': 0, 'tc': 0, 'kl': 0}
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                x_recon, mu, logvar, z = self.forward(batch)
                loss_dict = self.loss(batch, x_recon, mu=mu, logvar=logvar, z=z, return_parts=True)
                
                loss_dict['total'].backward()
                optimizer.step()
                
                for k in epoch_losses:
                    epoch_losses[k] += loss_dict[k].item()
            
            for k in epoch_losses:
                history[f'train_{k}'].append(epoch_losses[k] / len(train_loader))
            
            # Validation
            self.eval()
            epoch_losses = {'total': 0, 'recon': 0, 'mi': 0, 'tc': 0, 'kl': 0}
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    x_recon, mu, logvar, z = self.forward(batch)
                    loss_dict = self.loss(batch, x_recon, mu=mu, logvar=logvar, z=z, return_parts=True)
                    
                    for k in epoch_losses:
                        epoch_losses[k] += loss_dict[k].item()
            
            for k in epoch_losses:
                history[f'val_{k}'].append(epoch_losses[k] / len(val_loader))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Train - Total: {history['train_total'][-1]:.4f}, Recon: {history['train_recon'][-1]:.4f}, "
                      f"MI: {history['train_mi'][-1]:.4f}, TC: {history['train_tc'][-1]:.4f}, KL: {history['train_kl'][-1]:.4f} | "
                      f"Val - Total: {history['val_total'][-1]:.4f}")
        
        self.eval()
        return history



class BinaryBetaVAE(LatentModule):
    """
    Beta-VAE for binary data.
    Sigmoid output with BCE reconstruction loss.
    Loss = BCE + beta * KL
    
    Args:
        input_dim: number of binary loci
        latent_dim: latent space dimension
        hidden_dim: hidden layer size
        beta: KL weight
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dim=32, beta=1.0):
        super().__init__(input_dim, latent_dim)
        self.beta = beta
        
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        """Returns mu."""
        h = self.encoder_net(x)
        return self.fc_mu(h)
    
    def encode_dist(self, x):
        """Returns mu and clamped logvar."""
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-20, max=2)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    
    def decode(self, z):
        """Returns sigmoid probabilities."""
        return torch.sigmoid(self.decoder_net(z))
    
    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def loss(self, x, x_recon, mu=None, logvar=None, z=None, return_parts=False, **kwargs):
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        
        if return_parts:
            return {'total': total, 'recon': recon_loss, 'kl': kl_loss}
        return total
    
    def fit(self, dataset, epochs=100, batch_size=32, lr=1e-3, device='cpu', verbose=True, val_split=0.2):
        self.to(device)
        
        data = torch.tensor(np.array(dataset), dtype=torch.float32)
        n_val = int(len(data) * val_split)
        n_train = len(data) - n_val
        
        indices = torch.randperm(len(data))
        train_data = data[indices[:n_train]]
        val_data = data[indices[n_train:]]
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        history = {
            'train_total': [], 'train_recon': [], 'train_kl': [],
            'val_total': [], 'val_recon': [], 'val_kl': []
        }
        
        for epoch in range(epochs):
            self.train()
            epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                x_recon, mu, logvar, z = self.forward(batch)
                loss_dict = self.loss(batch, x_recon, mu=mu, logvar=logvar, z=z, return_parts=True)
                loss_dict['total'].backward()
                optimizer.step()
                for k in epoch_losses:
                    epoch_losses[k] += loss_dict[k].item()
            for k in epoch_losses:
                history[f'train_{k}'].append(epoch_losses[k] / len(train_loader))
            
            self.eval()
            epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    x_recon, mu, logvar, z = self.forward(batch)
                    loss_dict = self.loss(batch, x_recon, mu=mu, logvar=logvar, z=z, return_parts=True)
                    for k in epoch_losses:
                        epoch_losses[k] += loss_dict[k].item()
            for k in epoch_losses:
                history[f'val_{k}'].append(epoch_losses[k] / len(val_loader))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train - Total: {history['train_total'][-1]:.4f}, Recon: {history['train_recon'][-1]:.4f}, KL: {history['train_kl'][-1]:.4f} | "
                      f"Val - Total: {history['val_total'][-1]:.4f}")
        
        self.eval()
        return history



class BetaVAE(LatentModule):
    """
    Beta-VAE for continuous data.
    MSE reconstruction + beta * KL divergence.
    Supports evolvability loss.

    Args:
        input_dim: input dimension
        latent_dim: latent space dimension
        hidden_dim: hidden layer size
        beta: KL weight
    """

    def __init__(self, input_dim, latent_dim, hidden_dim=32, beta=1.0):
        super().__init__(input_dim, latent_dim)
        self.beta = beta

        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder_net(x)
        return self.fc_mu(h)

    def encode_dist(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-20, max=2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def loss(self, x, x_recon, mu=None, logvar=None, z=None, return_parts=False, **kwargs):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        if return_parts:
            return {'total': total, 'recon': recon_loss, 'kl': kl_loss}
        return total

    def fit(self, dataset, epochs=100, batch_size=32, lr=1e-3, device='cpu',
            verbose=True, val_split=0.2, evolvability_loss=None):
        self.to(device)

        data = torch.tensor(np.array(dataset), dtype=torch.float32)
        n_val = int(len(data) * val_split)
        n_train = len(data) - n_val

        indices = torch.randperm(len(data))
        train_data = data[indices[:n_train]]
        val_data = data[indices[n_train:]]

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, drop_last=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = {
            'train_total': [], 'train_recon': [], 'train_kl': [],
            'val_total': [], 'val_recon': [], 'val_kl': []
        }
        if evolvability_loss is not None:
            history['train_evol'] = []
            history['train_evol_f'] = []
            history['train_evol_n'] = []
            history['train_evol_c'] = []

        for epoch in range(epochs):
            self.train()
            epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
            if evolvability_loss is not None:
                epoch_losses['evol'] = 0
                epoch_losses['evol_f'] = 0
                epoch_losses['evol_n'] = 0
                epoch_losses['evol_c'] = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                x_recon, mu, logvar, z = self.forward(batch)
                loss_dict = self.loss(batch, x_recon, mu=mu, logvar=logvar,
                                      z=z, return_parts=True)
                total = loss_dict['total']

                if evolvability_loss is not None:
                    evol_dict = evolvability_loss(z, batch, self.decode)
                    total = total + evol_dict['total']
                    epoch_losses['evol'] += evol_dict['total'].item()
                    epoch_losses['evol_f'] += evol_dict['f'].item()
                    epoch_losses['evol_n'] += evol_dict['n'].item()
                    epoch_losses['evol_c'] += evol_dict['c'].item()

                total.backward()
                optimizer.step()

                for k in ['total', 'recon', 'kl']:
                    epoch_losses[k] += loss_dict[k].item()

            for k in ['total', 'recon', 'kl']:
                history[f'train_{k}'].append(epoch_losses[k] / len(train_loader))
            if evolvability_loss is not None:
                history['train_evol'].append(epoch_losses['evol'] / len(train_loader))
                history['train_evol_f'].append(epoch_losses['evol_f'] / len(train_loader))
                history['train_evol_n'].append(epoch_losses['evol_n'] / len(train_loader))
                history['train_evol_c'].append(epoch_losses['evol_c'] / len(train_loader))

            self.eval()
            epoch_losses_val = {'total': 0, 'recon': 0, 'kl': 0}
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    x_recon, mu, logvar, z = self.forward(batch)
                    loss_dict = self.loss(batch, x_recon, mu=mu, logvar=logvar,
                                         z=z, return_parts=True)
                    for k in ['total', 'recon', 'kl']:
                        epoch_losses_val[k] += loss_dict[k].item()

            for k in ['total', 'recon', 'kl']:
                history[f'val_{k}'].append(epoch_losses_val[k] / len(val_loader))

            if verbose and (epoch + 1) % 10 == 0:
                msg = (f"Epoch {epoch+1}/{epochs} | "
                       f"Train Total: {history['train_total'][-1]:.4f}, "
                       f"Recon: {history['train_recon'][-1]:.4f}, "
                       f"KL: {history['train_kl'][-1]:.4f}")
                if evolvability_loss is not None:
                    msg += (f", Evol: {history['train_evol'][-1]:.4f}"
                            f" (F: {history['train_evol_f'][-1]:.4f}"
                            f", N: {history['train_evol_n'][-1]:.4f}"
                            f", C: {history['train_evol_c'][-1]:.4f})")
                msg += f" | Val Total: {history['val_total'][-1]:.4f}"
                print(msg)

        self.eval()
        return history

