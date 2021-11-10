"""
Main python script. It initializes a Lightning Module of the original model
and fits it. The parameters can be changed in the constants of this file.
"""

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.module import LightningModuleMLP
from src.data import data_preprocess, SalesDataset

X_PATH = "data/datasets/x.csv"
Y_PATH = "data/datasets/y.csv"

TRAIN_SIZE = 0.8
NUM_LAYERS = 2
OUTPUT_DIM = 1
PROBABILITY = 0.9

SHUFFLE = True
NUM_WORKERS = 0
RANDOM_STATE = 42

grid = {
    "batch_size": [512],
    "hidden_dim": [8],
    "dropout": [0.05],
    "learning_rate":  [0.001]
}

seed_everything(RANDOM_STATE)

def main(args):
    
    X, y = data_preprocess(X_PATH, Y_PATH)

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X,
        y,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test,
        y_val_test,
        random_state=RANDOM_STATE,
        train_size=0.5
    )

    for b in grid["batch_size"]:
        batch_size = b

        train_dataset = SalesDataset(X_train, y_train)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE
        )

        val_dataset = SalesDataset(X_val, y_val)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE
        )

        test_dataset = SalesDataset(X_test, y_test)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE
        )

        for h in grid["hidden_dim"]:
            for d in grid["dropout"]:
                for lr in grid["learning_rate"]:

                    model = LightningModuleMLP(
                        input_dim=X.shape[-1],
                        hidden_dim=h,
                        output_dim=OUTPUT_DIM,
                        dropout=d,
                        learning_rate=lr,
                        num_layers=NUM_LAYERS,
                        pos_weight=train_dataset.pos_weight,
                        probability=PROBABILITY
                    )

                    tb_logger = TensorBoardLogger(
                        "data/logs/",
                        name=f"model_{batch_size}_{h}_{d}_{lr}"
                    )
    
                    val_checkpoint = ModelCheckpoint(
                        dirpath='../data/checkpoints/',
                        filename=f"val_model_{batch_size}_{h}_{d}_{lr}"+"_{epoch}_{val_loss:.2f}",
                        monitor='val_loss',
                        mode='min',
                        verbose=True,
                        save_top_k=1
                    )

                    train_checkpoint = ModelCheckpoint(
                        dirpath='../data/checkpoints/',
                        filename=f"train_model_{batch_size}_{h}_{d}_{lr}"+"_{epoch}_{val_loss:.2f}",
                        monitor='train_loss',
                        mode='min',
                        verbose=True,
                        save_top_k=1
                    )

                    trainer = Trainer.from_argparse_args(
                        args,
                        gpus=1,
                        val_check_interval=1.0,
                        log_every_n_steps=1,
                        default_root_dir="data/logs",
                        logger=tb_logger,
                        callbacks=[train_checkpoint, val_checkpoint]
                    )
                    trainer.fit(model, train_dataloader, val_dataloader)
                    trainer.test(model, test_dataloader)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
