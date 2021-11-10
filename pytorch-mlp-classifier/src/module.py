import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.model import MLP

from sklearn.metrics import roc_auc_score


class LightningModuleMLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        learning_rate: float = 0.01,
        pos_weight: float = 0.5,
        probability: float = 0.5
    ):
        super().__init__()

        self.model = MLP(
            input_dim,
            hidden_dim,
            output_dim,
            dropout,
            num_layers
        )

        self.lr = learning_rate
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.sig = nn.Sigmoid()
        self.prob = probability

        self.training_losses = []
        self.validation_losses = []
        self.test_losses = []
        self.training_acc = []
        self.validation_acc = []
        self.test_acc = []
        self.training_auc = []
        self.validation_auc = []
        self.test_auc = []

    def forward(self, x):
        return self.model(x)

    def calculate_and_log_metrics(
        self,
        out: torch.tensor,
        y: torch.tensor,
        loss_storage: list,
        loss_name: str,
        acc_storage: list,
        acc_name: str,
        auc_storage: list,
        auc_name: str
    ):
        sig_out = self.sig(out)

        loss = self.criterion(out, y)
        loss_storage.append(loss.item())
        self.log(loss_name, loss)

        acc = torch.count_nonzero(sig_out > self.prob, dim=0).item() / len(out)
        acc_storage.append(acc)
        self.log(acc_name, acc)

        auc = roc_auc_score(y.detach().cpu().numpy(), sig_out.detach().cpu().numpy())
        auc_storage.append(auc)
        self.log(auc_name, auc)

        return loss, acc, auc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer': optimizer}

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss, acc, auc = self.calculate_and_log_metrics(
            out,
            y,
            self.training_losses,
            "train_loss",
            self.training_acc,
            "train_acc",
            self.training_auc,
            "train_auc"
        )

        return {'loss': loss, "acc": acc, "auc": auc}

    def training_epoch_end(self, training_step_outputs):
        avg_train_loss = torch.mean(torch.tensor(self.training_losses))
        self.logger.experiment.add_scalar(
            'avg_training_loss',
            avg_train_loss,
            global_step=self.current_epoch
        )
        self.training_losses = []
        
        avg_train_acc = torch.mean(torch.tensor(self.training_acc))
        self.logger.experiment.add_scalar(
            'avg_training_acc',
            avg_train_acc,
            global_step=self.current_epoch
        )
        self.training_acc = []
        
        avg_train_auc = torch.mean(torch.tensor(self.training_auc))
        self.logger.experiment.add_scalar(
            'avg_training_auc',
            avg_train_auc,
            global_step=self.current_epoch
        )
        self.training_auc = []

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss, acc, auc = self.calculate_and_log_metrics(
            out,
            y,
            self.validation_losses,
            "val_loss",
            self.validation_acc,
            "val_acc",
            self.validation_auc,
            "val_auc"
        )

        return {'loss': loss, "acc": acc, "auc": auc}

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
        self.logger.experiment.add_scalar(
            'avg_validation_loss',
            avg_val_loss,
            global_step=self.current_epoch
        )
        self.validation_losses = []
        
        avg_val_acc = torch.mean(torch.tensor(self.validation_acc))
        self.logger.experiment.add_scalar(
            'avg_validation_acc',
            avg_val_acc,
            global_step=self.current_epoch
        )
        self.validation_acc = []
        
        avg_val_auc = torch.mean(torch.tensor(self.validation_auc))
        self.logger.experiment.add_scalar(
            'avg_validation_auc',
            avg_val_auc,
            global_step=self.current_epoch
        )
        self.validation_auc = []

    def test_step(self, test_batch, test_batch_idx):
        x, y = test_batch
        out = self.forward(x)
        loss, acc, auc = self.calculate_and_log_metrics(
            out,
            y,
            self.test_losses,
            "test_loss",
            self.test_acc,
            "test_acc",
            self.test_auc,
            "test_auc"
        )

        return {'loss': loss, "acc": acc, "auc": auc}

    def test_epoch_end(self, test_step_outputs):
        avg_test_loss = torch.mean(torch.tensor(self.test_losses))
        self.logger.experiment.add_scalar(
            'avg_test_loss',
            avg_test_loss,
            global_step=self.current_epoch
        )
        self.test_losses = []
        
        avg_test_acc = torch.mean(torch.tensor(self.test_acc))
        self.logger.experiment.add_scalar(
            'avg_test_acc',
            avg_test_acc,
            global_step=self.current_epoch
        )
        self.test_acc = []
        
        avg_test_auc = torch.mean(torch.tensor(self.test_auc))
        self.logger.experiment.add_scalar(
            'avg_test_auc',
            avg_test_auc,
            global_step=self.current_epoch
        )
        self.test_auc = []