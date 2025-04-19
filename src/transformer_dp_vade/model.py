# model.py

import torch
import torch.nn as nn
import math

################################
# 1) PositionalEncoding
################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # formula classica: sin/cos su dimensioni pari/dispari
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        Ritorna x + positional encoding (stessa shape).
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

################################
# 2) MiniTransformer
################################
class MiniTransformer(nn.Module):
    """
    TransformerEncoder con:
     - 2 strati (num_layers=2), dropout=0.1
     - LayerNorm finale
     - mean+max pooling => output shape = 2*d_model (quindi 64 se d_model=32).
    """
    def __init__(self, d_input=25, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Proiezione iniziale d_input -> d_model
        self.input_linear = nn.Linear(d_input, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)

        # Costruiamo il TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,  # dropout su attention e FC
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # LayerNorm finale sulla sequenza
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, seq_len, d_input)
        Ritorna un embedding di dimensione 2*d_model (mean+max pooling).
        """
        b, s, d = x.shape

        # (1) Proiezione
        x = self.input_linear(x)  # (B, s, d_model)

        # (2) Aggiunta del positional encoding
        x = self.pos_encoder(x)   # (B, s, d_model)

        # (3) Il TransformerEncoder, con batch_first=False, vuole (s, B, d_model)
        x = x.transpose(0, 1)     # (s, B, d_model)

        # (4) Passiamo i dati nel TransformerEncoder
        out = self.transformer_encoder(x)  # (s, B, d_model)

        # (5) Torniamo a (B, s, d_model)
        out = out.transpose(0, 1)

        # (6) LayerNorm sul canale d_model
        out = self.layernorm(out)

        # (7) Mean + Max pooling sul seq_len
        mean_pool = out.mean(dim=1)        # (B, d_model)
        max_pool, _ = out.max(dim=1)       # (B, d_model)
        seq_embed = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2*d_model)

        return seq_embed


################################
# ClassificationHead
################################
class ClassificationHead(nn.Module):
    """
    MLP a 2 layer:
     - FC(2*d_model, hidden) + ReLU + Dropout
     - FC(hidden, n_classes)
    d_in=64 se d_model=32 e usiamo mean+max pooling.
    """
    def __init__(self, d_in=64, n_classes=10, hidden=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        """
        x: (batch_size, d_in) -> (batch_size, n_classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

################################
# 3) DPVaDE
################################
class DPVaDE(nn.Module):
    """
    VaDE (VAE con prior GMM), usando LayerNorm invece di BatchNorm.
    Ora input_dim=64 (se la rete produce 2*d_model=64).
    """
    def __init__(self, input_dim=64, latent_dim=16, hidden_dim=64,
                 n_components=9, alpha_new=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.alpha_new = alpha_new

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_ln1 = nn.LayerNorm(hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_ln2 = nn.LayerNorm(hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_ln1 = nn.LayerNorm(hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_ln2 = nn.LayerNorm(hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

        # Parametri GMM
        self.pi = nn.Parameter(torch.ones(n_components)/n_components)
        self.gmm_mu = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.gmm_logvar = nn.Parameter(torch.zeros(n_components, latent_dim))

    def encode(self, x):
        """
        x: (B, input_dim=64)
        """
        h = self.enc_fc1(x)
        h = torch.relu(h)
        h = self.enc_ln1(h)

        h = self.enc_fc2(h)
        h = torch.relu(h)
        h = self.enc_ln2(h)

        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc1(z)
        h = torch.relu(h)
        h = self.dec_ln1(h)

        h = self.dec_fc2(h)
        h = torch.relu(h)
        h = self.dec_ln2(h)

        out = self.dec_out(h)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def compute_loss(self, x):
        recon, mu, logvar, z = self.forward(x)
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')

        log_qz_x = self.log_normal_pdf(z, mu, logvar)
        log_pz = self.log_p_z(z)
        kl = (log_qz_x - log_pz).mean()

        total = recon_loss + kl
        return total, recon_loss, kl, z

    def log_normal_pdf(self, z, mu, logvar):
        log2pi = math.log(2*math.pi)
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        return -0.5 * (
            log2pi * z.size(1) +
            torch.sum(logvar_clamped + (z - mu)**2 / torch.exp(logvar_clamped), dim=1)
        )

    def log_p_z(self, z):
        log_pi = torch.log_softmax(self.pi, dim=0)
        log2pi = math.log(2*math.pi)
        z_ = z.unsqueeze(1)               # (B,1,latent_dim)
        mu_ = self.gmm_mu.unsqueeze(0)    # (1,K,latent_dim)
        logvar_ = torch.clamp(self.gmm_logvar.unsqueeze(0), min=-10.0, max=10.0)

        diff2 = (z_ - mu_)**2 / torch.exp(logvar_)
        log_prob = -0.5 * torch.sum(log2pi + logvar_ + diff2, dim=2)  # (B,K)

        log_mix = log_prob + log_pi.unsqueeze(0)
        return torch.logsumexp(log_mix, dim=1)

    @torch.no_grad()
    def e_step(self, Z):
        log_pi = torch.log_softmax(self.pi + 1e-6, dim=0)
        log2pi = math.log(2*math.pi)
        z_ = Z.unsqueeze(1)
        mu_ = self.gmm_mu.unsqueeze(0)
        logvar_ = torch.clamp(self.gmm_logvar.unsqueeze(0), min=-10.0, max=10.0)

        diff2 = (z_ - mu_)**2 / torch.exp(logvar_)
        log_prob = -0.5 * torch.sum(log2pi + logvar_ + diff2, dim=2)
        log_mix = log_prob + log_pi.unsqueeze(0)
        log_sum = torch.logsumexp(log_mix, dim=1, keepdim=True)
        gamma = torch.exp(log_mix - log_sum)
        return gamma

    @torch.no_grad()
    def m_step(self, Z, gamma):
        Nk = gamma.sum(dim=0) + 1e-6
        self.pi.data = Nk / Nk.sum()

        mu_k = (gamma.unsqueeze(2) * Z.unsqueeze(1)).sum(dim=0) / Nk.unsqueeze(1)
        var_k = (gamma.unsqueeze(2) * ((Z.unsqueeze(1) - mu_k)**2)).sum(dim=0) / Nk.unsqueeze(1)
        var_k = var_k + 1e-6  # evita zero var

        self.gmm_mu.data = mu_k
        self.gmm_logvar.data = torch.log(var_k)

    @torch.no_grad()
    def spawn_new_component(self, Z, gamma, threshold=0.01, min_count=50):
        max_gamma = gamma.max(dim=1).values
        outlier_idx = (max_gamma < threshold).nonzero(as_tuple=True)[0]
        if len(outlier_idx) < min_count:
            return

        print(f"*** Spawning new cluster! K={self.n_components} -> K={self.n_components+1} ***")
        self.pi.data = torch.cat([self.pi.data, torch.tensor([self.alpha_new], device=self.pi.device)], dim=0)
        z_out = Z[outlier_idx]
        new_mu = z_out.mean(dim=0, keepdim=True)
        new_var = z_out.var(dim=0, keepdim=True) + 1e-6

        self.gmm_mu.data = torch.cat([self.gmm_mu.data, new_mu], dim=0)
        self.gmm_logvar.data = torch.cat([self.gmm_logvar.data, torch.log(new_var)], dim=0)

        self.n_components += 1
        self.pi.data = self.pi.data / self.pi.data.sum()
