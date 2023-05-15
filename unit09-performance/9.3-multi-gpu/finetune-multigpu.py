import os
import os.path as op
import time

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchmetrics
from datasets import load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from local_dataset_utilities import (
    IMDBDataset,
    download_dataset,
    load_dataset_into_to_dataframe,
    partition_dataset,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from watermark import watermark


# pip install -U deepspeed

def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)


def plot_logs(log_dir):
    metrics = pd.read_csv(op.join(log_dir, "metrics.csv"))

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )
    plt.savefig(op.join(log_dir, "loss.pdf"))

    df_metrics[["train_acc", "val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Accuracy"
    )
    plt.savefig(op.join(log_dir, "acc.pdf"))


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate=5e-5):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        self.log("train_loss", outputs["loss"])
        with torch.no_grad():
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            self.train_acc(predicted_labels, batch["label"])
            self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        return outputs["loss"]  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        self.log("val_loss", outputs["loss"], prog_bar=True)

        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.val_acc(predicted_labels, batch["label"])
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )

        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.test_acc(predicted_labels, batch["label"])
        self.log("accuracy", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.trainer.model.parameters(), lr=self.learning_rate
        )
        return optimizer


if __name__ == "__main__":

    print(watermark(packages="torch,lightning,transformers,deepspeed", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    # #########################
    # ## 1 Loading the Dataset
    # #########################
    download_dataset()
    df = load_dataset_into_to_dataframe()
    if not (op.exists("train.csv") and op.exists("val.csv") and op.exists("test.csv")):
        partition_dataset(df)

    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        },
    )

    # ########################################
    # ## 2 Tokenization and Numericalization
    # #######################################

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("Tokenizer input max length:", tokenizer.model_max_length)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    print("Tokenizing ...")
    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    del imdb_dataset
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ########################################
    # ## 3 Set Up DataLoaders
    # ########################################

    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=64, num_workers=4)

    test_loader = DataLoader(dataset=test_dataset, batch_size=64, num_workers=4)

    # ########################################
    # ## 4 Initializing the Model
    # ########################################

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # ########################################
    # ## 5 Finetuning
    # ########################################

    lightning_model = LightningModel(model)

    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc")  # save top 1 model
    ]
    logger = CSVLogger(save_dir="logs/", name="my-model")

    trainer = L.Trainer(
        max_epochs=3,
        callbacks=callbacks,
        precision="16-mixed",
        accelerator="gpu",
        strategy="ddp",
        devices=[0, 1, 2, 3],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    start = time.time()
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    test_acc = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
    print(test_acc)

    with open(op.join(trainer.logger.log_dir, "outputs.txt"), "w") as f:
        f.write((f"Time elapsed {elapsed/60:.2f} min\n"))
        f.write(f"Test acc: {test_acc}")

    # ########################################
    # ## 6 Plot logs
    # ########################################

    plot_logs(trainer.logger.log_dir)
