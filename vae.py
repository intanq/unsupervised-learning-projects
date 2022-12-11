import torch
from torch import nn
class VAE(nn.Module):
    def __init__(self, d_in=28*28, d_z=100, d_h=512):
        super(VAE, self).__init__()
        
        self.d_z = d_z
        
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z * 2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_in),
            nn.Sigmoid()
        )
        
    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu_logvar = mu_logvar.reshape(-1, 2, self.d_z)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        xr = self.decode(z)
        return xr, mu, logvar

class VAE_FT(nn.Module):
    def __init__(self, d_in=28*28, d_z=100, d_h=512):
        super(VAE_FT, self).__init__()
        
        self.d_z = d_z
        
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z * 2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_in),
            # nn.Sigmoid()
        )
        
    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu_logvar = mu_logvar.reshape(-1, 2, self.d_z)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        xr = self.decode(z)
        return xr, mu, logvar
    

class VAE_FT_sigm(nn.Module):
    def __init__(self, d_in=28*28, d_z=100, d_h=512):
        super(VAE_FT_sigm, self).__init__()
        
        self.d_z = d_z
        
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z * 2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_in),
            nn.Sigmoid()
        )
        
    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu_logvar = mu_logvar.reshape(-1, 2, self.d_z)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        xr = self.decode(z)
        return xr, mu, logvar
    