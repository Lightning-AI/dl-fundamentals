import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=200)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=200)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=200)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


def plot_csv_logger(
    csv_path, model_name, loss_names=["train_loss", "val_loss"], eval_names=["train_acc", "val_acc"]
):

    metrics = pd.read_csv(csv_path)

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)

    df_metrics[loss_names].plot(grid=True, legend=True, xlabel="Epoch", ylabel="Loss")
    plt.savefig(f'{model_name}-loss.png')

    df_metrics[eval_names].plot(grid=True, legend=True, xlabel="Epoch", ylabel="ACC")
    plt.savefig(f'{model_name}-acc.png')

class TinyImageNetDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.build_label_index(img_dir)
        self.init_data(img_dir)

    def __getitem__(self, index):
        # print(f'getting image {self.images[index]} with label {self.labels[index]}')

        img = Image.open(self.images[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        label = self.label_index[self.labels[index]]
        return img, label
    
    def __len__(self):
        return len(self.labels)

    def build_label_index(self, img_dir):
        base_dir = os.path.join(img_dir, 'train')
        self.label_index = {}
        for idx, label in enumerate(os.listdir(base_dir)):
            # print(f'index={idx}, label={label}')
            self.label_index[label] = idx
        # print(f'Total lables = {len(self.label_index)}')


class TinyImageNetTrainDataset(TinyImageNetDataset):
    def init_data(self, img_dir):
        self.images = []
        self.labels = []

        base_dir = os.path.join(img_dir, 'train')

        for label in os.listdir(base_dir):
            label_dir = os.path.join(base_dir, label, 'images')
            for img in os.listdir(label_dir):
                self.images.append(os.path.join(label_dir, img))
                self.labels.append(label)

class TinyImageNetValDataset(TinyImageNetDataset):
    def init_data(self, img_dir):

        base_dir = os.path.join(img_dir, 'val')
        df = pd.read_csv(os.path.join(base_dir, 'val_annotations.txt'), sep=r"\s+", header=None)

        self.images = [os.path.join(base_dir, 'images', file) for file in df[0]]
        self.labels = df[1]

class TinyImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path='./tiny-imagenet-200',
        batch_size=64,
        height_width=None,
        num_workers=0,
        augment_data=False,
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.height_width = height_width
        self.num_workers = num_workers


        if augment_data:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((250, 250)),
                    transforms.RandomCrop(self.height_width),
                    transforms.RandomHorizontalFlip(p=0.2),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                ]
            )

            self.test_transform = transforms.Compose(
                [
                    transforms.Resize((250, 250)),
                    transforms.CenterCrop(self.height_width),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )

            self.test_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )


    def setup(self, stage=None):
        train = TinyImageNetTrainDataset(
            img_dir=self.data_path,
            transform=self.train_transform,
        )
        self.test = TinyImageNetValDataset(
            img_dir=self.data_path,
            transform=self.test_transform,
        )

        self.train, self.valid = random_split(train, lengths=[90000, 10000])
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )

def get_model_list():
    model_list = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    # entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    # for e in entrypoints:
    #     if e.startswith("resnet"):
    #         model_list.append(e)
    return model_list