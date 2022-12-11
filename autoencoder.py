from torch import nn

class MLPAutoEnc(nn.Module):
    def __init__(self, d_in=28*28, d_z=100, d_h=512):
        super(MLPAutoEnc, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_rep = self.encoder(x)
        xr = self.decoder(latent_rep)
        return xr

class AutoEncSigm(nn.Module):
    def __init__(self, d_in=28*28, d_z=100, d_h=512):
        super(AutoEncSigm, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_rep = self.encoder(x)
        xr = self.decoder(latent_rep)
        return xr

class AutoEncFT(nn.Module):
    def __init__(self, d_in=28*28, d_z=100, d_h=512):
        super(AutoEncFT, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_in),
            # nn.Sigmoid()
        )

    def forward(self, x):
        latent_rep = self.encoder(x)
        xr = self.decoder(latent_rep)
        return xr