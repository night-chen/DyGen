"""
Architecture of conditional VAE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

class text_CVAE(nn.Module):
    def __init__(self, args, hidden_dim):
        super(text_CVAE, self).__init__()
        self.args = args
        self.encoder = nn.Linear(hidden_dim+args.num_classes, args.num_classes)
        self.decoder = nn.Linear(hidden_dim+args.num_classes, args.num_classes)

        self.softplus = nn.Softplus(self.args.softplus_beta)

    def Encoder(self,x,y):
        input_xy = torch.cat([x, y], 1)
        alpha = self.softplus(self.encoder(input_xy)) + 1.0 + 1 / self.args.num_classes # to make probability

        return alpha

    def reparameterize(self, alpha):
        return ((torch.ones_like(alpha) / alpha) * (torch.log(torch.rand_like(alpha) * alpha) + torch.lgamma(alpha))).exp()

    def Decoder(self, x, y):
        y_tilde = self.decoder(torch.cat([x,y],1))

        return y_tilde

    def forward(self,x,ytilde):
        alpha = self.Encoder(x,ytilde)
        z = self.reparameterize(alpha)
        y_tilde = self.Decoder(x,z)

        return y_tilde, alpha