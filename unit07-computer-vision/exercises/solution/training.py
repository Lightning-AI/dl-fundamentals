import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torch
import torchvision
from local_utilities import LightningModel, TinyImageNetDataModule, plot_csv_logger, get_model_list

def train_model(resnet_type, augmentation):
    model_name = f"tiny-imagenet-{resnet_type}-{augmentation}"
    print(f"training model {model_name}")

    if augmentation == "augmented":
        augment_data = True
    else:
        augment_data = False   

    dm = TinyImageNetDataModule(height_width=(224, 224), batch_size=64, num_workers=4, augment_data=augment_data)
   
    pytorch_model = torch.hub.load('pytorch/vision', resnet_type, weights=None)
    L.pytorch.seed_everything(123)
    print(pytorch_model)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.1)

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        logger=CSVLogger(save_dir="logs/", name=model_name),
        deterministic=True,
    )

    trainer.fit(model=lightning_model, datamodule=dm)
    torch.save(pytorch_model.state_dict(), f"{model_name}.pt")
    plot_csv_logger(f"{trainer.logger.log_dir}/metrics.csv", model_name=model_name)

def train_models(model_list):
    for model in model_list:
        train_model(model, "baseline")
        train_model(model, "augmented")

def test_models(model_list):
    for model in model_list:
        test_model(model, "baseline")
        test_model(model, "augmented")

if __name__ == "__main__":
    model_list = get_model_list()
    train_models(model_list)