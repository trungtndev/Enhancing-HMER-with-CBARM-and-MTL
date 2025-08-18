import os

import typer
from mtl.datamodule import CROHMEDatamodule
from mtl.lit_mtl import LitMTL
from pytorch_lightning import Trainer, seed_everything
import zipfile

seed_everything(7)
def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m * n == 0:
        return m + n
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a = dp[i - 1][j] + 1
            b = dp[i][j - 1] + 1
            c = dp[i - 1][j - 1]
            if word1[i - 1] != word2[j - 1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]
path ={
    "0": "/home/trung/Documents/Weight/origin/6040.ckpt",
    "1": "/home/trung/Documents/Weight/origin/0.5980.ckpt",
    "2": "/home/trung/Documents/Weight/origin/0.6010.ckpt",
    "3": "/home/trung/Documents/Weight/origin/0.6111.ckpt",
    "4": "/home/trung/Documents/Weight/origin/epoch=83-step=63083-val_ExpRate=0.6132.ckpt",
}

def main(version: str, test_year: str):
    # generate output latex in result.zip
    # ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    # fnames = os.listdir(ckp_folder)
    # assert len(fnames) == 1
    # ckp_path = os.path.join(ckp_folder, fnames[0])
    # print(f"Test with fname: {fnames[0]}")
    ckp_path = path[version]
    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size=4, num_workers=16)

    model = LitMTL.load_from_checkpoint(ckp_path, lambda_1=1.0, lambda_2=1.0)

    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    typer.run(main)
