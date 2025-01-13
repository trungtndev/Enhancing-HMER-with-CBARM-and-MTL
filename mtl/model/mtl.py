from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from mtl.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder


class MTL(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)

        exp_out, imp_out = self.decoder(feature, mask, tgt)

        return exp_out, imp_out

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )

if __name__ == "__main__":
    img = torch.randn(4, 1, 256, 256)
    img_mask = torch.ones(4, 256, 256, dtype=torch.long)

    # Create a sample target tensor
    tgt = torch.randint(0, 50, size=(8, 30))
    model = MTL(
        d_model=256,
        growth_rate=32,
        num_layers=6,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        dc=32,
        cross_coverage=True,
        self_coverage=True,
    )
    output = model(img, img_mask, tgt)
    print(output)