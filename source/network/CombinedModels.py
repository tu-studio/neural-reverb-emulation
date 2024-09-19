import torch
import torch.nn as nn

class CombinedEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        if self.encoder.use_kl:
            mu, logvar, encoder_outputs = self.encoder(x)
            encoder_outputs.pop()
            z, _ = self.encoder.reparameterize(mu, logvar)
        else:
            encoder_outputs = self.encoder(x)
            z = encoder_outputs.pop()

        encoder_outputs = encoder_outputs[::-1]
        output = self.decoder(z, encoder_outputs)
        return output