import zipfile
from typing import List
import time
import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor

from mtl.datamodule import Batch, vocab
from mtl.model.mtl import MTL
from mtl.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                             to_bi_tgt_out)


class LitMTL(pl.LightningModule):
    def __init__(
            self,
            d_model: int,
            # encoder
            growth_rate: int,
            num_layers: int,
            # decoder
            nhead: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            dc: int,
            cross_coverage: bool,
            self_coverage: bool,
            lambda_1: float,
            lambda_2: float,
            # beam search
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            # training
            learning_rate: float,
            patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.comer_model = MTL(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.exprate_recorder = ExpRateRecorder()

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
        return self.comer_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        imp_tgt, imp_out = to_bi_tgt_out(batch.indices, self.device, is_implicit=True)

        out_hat, imp_hat = self(batch.imgs, batch.mask, tgt)

        out_loss = ce_loss(out_hat, out)
        imp_loss = ce_loss(imp_hat, imp_out)
        loss = out_loss * self.hparams.lambda_1 + imp_loss * self.hparams.lambda_2
        self.log("train_out_loss", out_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_imp_loss", imp_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        imp_tgt, imp_out = to_bi_tgt_out(batch.indices, self.device, is_implicit=True)
        out_hat, imp_hat = self(batch.imgs, batch.mask, tgt)

        out_loss = ce_loss(out_hat, out)
        imp_loss = ce_loss(imp_hat, imp_out)
        loss = out_loss * self.hparams.lambda_1 + imp_loss * self.hparams.lambda_2

        self.log("val_out_loss", out_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, )
        self.log("val_imp_loss", imp_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        start_time = time.time()
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        inference_time = time.time() - start_time
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps], inference_time

    def test_epoch_end(self, test_outputs) -> None:
        time_pack = test_outputs[2]
        total_time = sum(time_pack)
        n_samples = len(time_pack)
        print(f"Total inference time: {total_time:.5f} seconds.")
        print(f"Inference time: {total_time / n_samples:.5f} seconds.")

        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    def approximate_joint_search(
            self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.comer_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
