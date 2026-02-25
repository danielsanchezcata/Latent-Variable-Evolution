import torch
import numpy as np


class EvolvabilityLoss:
    """
    Differentiable evolvability loss using reparameterized mutation in latent space.

    For each z in batch:
      - Sample n_offspring via z' = z + sigma * eps
      - Compute behavior b(z') = decode(z')
      - F(z') = -sphere_fitness(b(z'))  [negate: higher = better]
      - N(z') = mean over k archive members of ||b_archive - b(z')||^2
      - C(z') = ||b(z) - b(z')||^2

    Loss = mean over offspring of [ -theta * F(z') - phi * N(z') + eta * C(z') ]
    Minimizing:
      - -theta * F(z'): maximize fitness
      - -phi * N(z'):   maximize novelty (push offspring away from archive)
      +  eta * C(z'):   minimize locality violation (keep offspring close to parent)

    Args:
        archive_behaviors: torch.Tensor of shape (M, input_dim), behaviors of dataset
        fitness_fn: callable(x: Tensor) -> Tensor of shape (batch,), raw fitness to minimize
        sigma: mutation noise scale
        theta: fitness weight
        phi: novelty weight
        eta: locality weight
        k: number of archive members subsampled per offspring for novelty
        n_offspring: number of offspring sampled per z
    """

    def __init__(self, archive_behaviors, fitness_fn, sigma=0.1,
                 theta=1.0, phi=1.0, eta=1.0, k=10, n_offspring=5):
        self.archive_behaviors = archive_behaviors  # (M, D)
        self.fitness_fn = fitness_fn
        self.sigma = sigma
        self.theta = theta
        self.phi = phi
        self.eta = eta
        self.k = k
        self.n_offspring = n_offspring

    def __call__(self, z, x, decode_fn):
        """
        Compute evolvability loss for a batch of latent vectors.

        Args:
            z: Tensor of shape (batch, latent_dim)
            x: Tensor of shape (batch, input_dim), normalized input (parent behavior)
            decode_fn: callable(z: Tensor) -> Tensor of shape (batch, input_dim)

        Returns:
            scalar loss
        """
        batch_size = z.size(0)
        device = z.device
        archive = self.archive_behaviors.to(device)  # (M, D)
        M = archive.size(0)

        b_parent = x  # (batch, D), no decoder call needed

        f_total = torch.zeros(1, device=device)
        n_total = torch.zeros(1, device=device)
        c_total = torch.zeros(1, device=device)

        for _ in range(self.n_offspring):
            eps = torch.randn_like(z)
            z_prime = z + self.sigma * eps
            b_prime = decode_fn(z_prime)

            f_total = f_total + self.fitness_fn(b_prime).mean()

            idx = torch.randperm(M, device=device)[:self.k]
            b_archive_k = archive[idx]
            diff = b_prime.unsqueeze(1) - b_archive_k.unsqueeze(0)
            n_total = n_total + (1 + diff.pow(2).mean(dim=2).mean(dim=1).mean()).log()

            c_total = c_total + (b_parent - b_prime).pow(2).mean(dim=1).mean()

        f_mean = f_total / self.n_offspring
        n_mean = n_total / self.n_offspring
        c_mean = c_total / self.n_offspring

        total_loss = -self.theta * f_mean - self.phi * n_mean + self.eta * c_mean

        return {'total': total_loss, 'f': f_mean, 'n': n_mean, 'c': c_mean}
