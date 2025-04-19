# model.py

import torch
import torch.nn as nn
import math

################################
# 1) PositionalEncoding
################################
class PositionalEncoding(nn.Module):
    """
    Implementazione sinusoidale classica. 
    Se i blocchi hanno seq_len = 10, qui max_len=500 è ampiamente sufficiente.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1,max_len,d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, d_model)
        Ritorna x + positional_encoding.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

################################
# 2) MiniTransformer
################################
class MiniTransformer(nn.Module):
    """
    Un piccolo encoder Transformer:
      - proiezione input_dim -> d_model
      - positional encoding
      - n strati di TransformerEncoder
      - pooling (mean) finale => (batch_size, d_model)
    """
    def __init__(self, d_input=12, d_model=32, nhead=4, num_layers=1, dim_feedforward=64):
        super().__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_input)
        """
        b, s, d = x.shape
        # Proiezione
        x = self.input_linear(x)   # (b,s,d_model)
        # Positional
        x = self.pos_encoder(x)    # (b,s,d_model)
        # Pytorch transformer richiede shape (s,b,d_model)
        x = x.transpose(0,1)       # (s,b,d_model)
        out = self.transformer_encoder(x)  # (s,b,d_model)
        out = out.transpose(0,1)   # (b,s,d_model)
        out = self.norm(out)
        # riduciamo con una media
        seq_embed = out.mean(dim=1)
        return seq_embed

################################
# 3) DPVaDE
################################
class DPVaDE(nn.Module):
    """
    VAE con prior GMM (VaDE) e "spawn_new_component" per scoprire dinamicamente
    nuovi cluster. Implementazione dimostrativa.
    """
    def __init__(self, input_dim=32, latent_dim=16, hidden_dim=64, n_components=9, alpha_new=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.alpha_new = alpha_new  # peso iniziale per la nuova componente

        # ENC
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # DEC
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Parametri GMM
        self.pi = nn.Parameter(torch.ones(n_components) / n_components)
        self.gmm_mu = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.gmm_logvar = nn.Parameter(torch.zeros(n_components, latent_dim))

    def encode(self, x):
        h = self.enc(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def compute_loss(self, x):
        """
        Ritorna (total_loss, recon_loss, kl_loss, z)
        """
        recon, mu, logvar, z = self.forward(x)
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        # KL VaDE
        log_qz_x = self.log_normal_pdf(z, mu, logvar)
        log_pz = self.log_p_z(z)
        kl = (log_qz_x - log_pz).mean()
        total = recon_loss + kl
        return total, recon_loss, kl, z

    def log_normal_pdf(self, z, mu, logvar):
        log2pi = math.log(2 * math.pi)
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        return -0.5 * (
            log2pi * z.size(1) + torch.sum(logvar_clamped + (z - mu) ** 2 / torch.exp(logvar_clamped), dim=1)
        )

    def log_p_z(self, z):
        """
        log ( sum_k pi_k * N(z|gmm_mu_k, var_k) )
        """
        log_pi = torch.log_softmax(self.pi, dim=0)  # (K,)
        var_k = torch.exp(self.gmm_logvar)          # (K,latent_dim)
        log2pi = math.log(2*math.pi)
        B = z.size(0)
        K = self.n_components
        z_ = z.unsqueeze(1)        # (B,1,ld)
        mu_ = self.gmm_mu.unsqueeze(0)
        logvar_ = self.gmm_logvar.unsqueeze(0)
        logvar_ = torch.clamp(logvar_, min=-10.0, max=10.0)
        # log N
        log_prob = -0.5 * torch.sum(log2pi + logvar_ + (z_ - mu_)**2 / torch.exp(logvar_), dim=2)
        log_mix = log_prob + log_pi.unsqueeze(0)  # (B,K)
        return torch.logsumexp(log_mix, dim=1)

    # E-step
    def e_step(self, Z):
        """
        Gamma_ik = p(k|z_i).
        Z: (N,latent_dim).
        """
        with torch.no_grad():
            log_pi = torch.log_softmax(self.pi + 1e-6, dim=0)
            var_k = torch.exp(self.gmm_logvar)
            z_ = Z.unsqueeze(1)  # (N,1,ld)
            mu_ = self.gmm_mu.unsqueeze(0)
            logvar_ = self.gmm_logvar.unsqueeze(0)
            log2pi = math.log(2*math.pi)

            log_prob = -0.5 * torch.sum(log2pi + logvar_ + (z_ - mu_)**2/torch.exp(logvar_), dim=2)
            log_mix = log_prob + log_pi.unsqueeze(0)
            log_sum = torch.logsumexp(log_mix, dim=1, keepdim=True)
            gamma = torch.exp(log_mix - log_sum)
            return gamma

    # M-step
    def m_step(self, Z, gamma):
        Nk = gamma.sum(dim=0)  # (K,)
        self.pi.data = Nk / Nk.sum()
        mu_k = (gamma.unsqueeze(2)*Z.unsqueeze(1)).sum(dim=0)/Nk.unsqueeze(1)
        var_k = (gamma.unsqueeze(2)*((Z.unsqueeze(1)-mu_k)**2)).sum(dim=0)/Nk.unsqueeze(1)
        self.gmm_mu.data = mu_k
        self.gmm_logvar.data = torch.log(var_k + 1e-6)

    def spawn_new_component(self, Z, gamma, threshold=0.01, min_count=50, val_fraction=0.1):
        """
        Migliorato: controlla se c'è un gruppo di outlier stabili (bassa responsabilità).
         - threshold: se max gamma_i < threshold => outlier
         - min_count: numero minimo di outlier per spawnare
         - val_fraction: potresti tenerlo per validare la "bontà" del cluster
        """
        max_gamma = gamma.max(dim=1).values
        outlier_idx = (max_gamma < threshold).nonzero(as_tuple=True)[0]

        if len(outlier_idx) < min_count:
            return  # troppo pochi outlier

        # Esempio di "validazione semplificata": controlliamo la varianza dei new cluster
        # (volendo potresti implementare un check se outlier_idx si addensa in un cluster).
        print(f"*** Spawning new cluster! K={self.n_components} -> K={self.n_components+1} ***")
        K_new = self.n_components + 1

        # Espandiamo
        self.pi.data = torch.cat([self.pi.data, torch.tensor([self.alpha_new], device=self.pi.device)], dim=0)
        z_out = Z[outlier_idx]
        new_mu = z_out.mean(dim=0, keepdim=True)
        new_var = z_out.var(dim=0, keepdim=True)+1e-6
        self.gmm_mu.data = torch.cat([self.gmm_mu.data, new_mu], dim=0)
        self.gmm_logvar.data = torch.cat([self.gmm_logvar.data, torch.log(new_var)], dim=0)

        self.n_components = K_new
        # Rinormalizziamo
        self.pi.data = self.pi.data / self.pi.data.sum()
