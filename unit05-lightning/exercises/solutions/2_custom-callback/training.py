import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
import torch
from shared_utilities import LightningModel, MNISTDataModule, PyTorchMLP
from watermark import watermark

train_val_diff = []


class CustomCallback(Callback):
    def on_validation_epoch_end(self, trainer, lightning_module):
        print(lightning_module.train_acc.compute(), lightning_module.val_acc.compute())
        diff = lightning_module.train_acc.compute() - lightning_module.val_acc.compute()
        train_val_diff.append(diff.item())


if __name__ == "__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    dm = MNISTDataModule()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cpu", devices=1, deterministic=True,
        logger=CSVLogger(save_dir="logs/", name="my-model"),
        callbacks=[CustomCallback()]
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    train_acc = trainer.test(dataloaders=dm.train_dataloader())[0]["test_acc"]
    val_acc = trainer.test(dataloaders=dm.val_dataloader())[0]["test_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

for diff in train_val_diff:
    print(f"Train-Validation accuracy difference: {diff*100:.2f}%", )