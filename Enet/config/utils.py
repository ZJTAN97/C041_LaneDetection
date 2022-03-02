from config import config
from torchvision import transforms
import torch
from prettytable import PrettyTable

transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
        ),
        transforms.ToTensor(),
    ]
)


class LRScheduler:
    """
    Learning rate scheduler. If validation loss does not decrease given number of `patience`
    epochs, learning rate will be increased by a certain factor
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        print("[INFO] Learning rate changed..")
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    """
    Stops training when loss not dropping
    """

    def __init__(self, patience=5, min_delta=0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}"
            )
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
