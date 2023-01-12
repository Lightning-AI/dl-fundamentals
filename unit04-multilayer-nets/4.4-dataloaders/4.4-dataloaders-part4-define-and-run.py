import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from watermark import watermark


class MyDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # based on DataFrame columns
        self.img_names = df["filepath"]
        self.labels = df["label"]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.labels.shape[0]


def viz_batch_images(batch):

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(batch[0][:64], padding=2, normalize=True), (1, 2, 0)
        )
    )
    plt.show()


if __name__ == "__main__":

    print(watermark(packages="torch", python=True))

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop((28, 28)),
                transforms.ToTensor(),
                # normalize images to [-1, 1] range
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop((28, 28)),
                transforms.ToTensor(),
                # normalize images to [-1, 1] range
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    }

    train_dataset = MyDataset(
        csv_path="mnist-pngs/new_train.csv",
        img_dir="mnist-pngs/",
        transform=data_transforms["train"],
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,  # want to shuffle the dataset
        num_workers=2,  # number processes/CPUs to use
    )

    val_dataset = MyDataset(
        csv_path="mnist-pngs/new_val.csv",
        img_dir="mnist-pngs/",
        transform=data_transforms["test"],
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    test_dataset = MyDataset(
        csv_path="mnist-pngs/test.csv",
        img_dir="mnist-pngs/",
        transform=data_transforms["test"],
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2
    )

    num_epochs = 1
    for epoch in range(num_epochs):

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= 3:
                break
            print(" Batch index:", batch_idx, end="")
            print(" | Batch size:", y.shape[0], end="")
            print(" | x shape:", x.shape, end="")
            print(" | y shape:", y.shape)

    print("Labels from current batch:", y)

    # Uncomment to visualize a data batch:
    # batch = next(iter(train_loader))
    # viz_batch_images(batch[0])
