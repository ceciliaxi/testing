import torch
import numpy as np
from merlion.models.anomaly.vae import VAE

class MLVAE(VAE):
    """
    """
    
    def get_batch_loss(self, model, batch, loss_func):             
        x = torch.tensor(batch, dtype=torch.float, device=self.device)
        recon_x, mu, log_var, _ = model(x, None)
        recon_loss = loss_func(x, recon_x)
        kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        loss = recon_loss + kld_loss * self.kld_weight
        return loss
    

    def get_anomaly_score(self, model, data): 
        y = torch.tensor(data, dtype=torch.float, device=self.device)
        r = np.zeros(y.shape)
        for _ in range(self.num_eval_samples):
            recon_y, _, _, _ = model(y, None)
            r += recon_y.cpu().data.numpy()
        r /= self.num_eval_samples
        # scores = np.sum(np.abs(np.concatenate(r) - np.concatenate(data)), axis=1)
        scores = np.sum(np.abs(r - data), axis=1)
        return scores
