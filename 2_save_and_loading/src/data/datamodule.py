import torch
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size:int = 64, num_workers: int = 11):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.ToTensor()
        self.seed = torch.Generator().manual_seed(42)

    def prepare_data(self) -> None:
        
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str=None) -> None:
        if stage == "fit" or stage is None:
            self.mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.class_dict = self.mnist_full.class_to_idx
            self.mnist_train, self.mnist_val = random_split(self.mnist_full, [55000, 5000], generator=self.seed)

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.class_dict = self.mnist_test.class_to_idx

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
