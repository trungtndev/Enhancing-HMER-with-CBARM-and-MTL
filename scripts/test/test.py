import os

import typer
from mtl.datamodule import CROHMEDatamodule
from mtl.lit_mtl import LitMTL
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


def main(version: str, test_year: str):
    # generate output latex in result.zip
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")

    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size=4)

    model = LitMTL.load_from_checkpoint(ckp_path, lambda_1=1.0, lambda_2=1.0)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
