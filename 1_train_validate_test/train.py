import sys
sys.path.append("src")

import torch
import lightning as L
from data import MNISTDataModule
from models import LitAutoEncoder, Encoder, Decoder


torch.set_float32_matmul_precision('medium')

# load data
data = MNISTDataModule(data_dir="data/", batch_size=64, num_workers=11)

# define model
model = LitAutoEncoder(Encoder(), Decoder())

# setup trainer
trainer = L.Trainer(max_epochs=5, accelerator="cuda", default_root_dir="checkpoint/")

if __name__ == "__main__":
    trainer.fit(model, data)
    trainer.test(model, data)
