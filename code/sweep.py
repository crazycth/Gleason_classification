from main import *
import math
from net_work import *
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sweep_config = {
    'method': 'random'
}

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
    },
    'lr': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.01
    },
    'weight_decay': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.03
    },
    'batch_size': {
        'distribution': 'q_log_uniform',
        'q': 1,
        'min': math.log(8),
        'max': math.log(64),
    },
    'epochs': {
        'value': 40
    },
    'model': {
        'values': ['resnet152', 'small_swin', "large_swin", "convnext_base", "resnet34"]
    },
    'root': {
        'values': ["./pic_trans_1", "./pic_save_1"]
    }
}

sweep_config['parameters'] = parameters_dict


def build_optimizer(network, optimizer, lr, weight_decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD((para for para in network.parameters() if para.requires_grad), lr=lr, momentum=0.9,
                                    weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam((para for para in network.parameters() if para.requires_grad), lr=lr, betas=(0.5, 0.999),
                               weight_decay=weight_decay)
    return optimizer


def build_model(network):
    if network == "small_swin":
        return get_swin_transformer(2)
    elif network == "large_swin":
        return get_large_transformer(2)
    elif network == "resnet34":
        return get_resnet_34(2)
    elif network == 'resnet152':
        return get_resnet_152(2)
    elif network == 'resnet101':
        return get_resnet_101(2)
    elif network == 'convnext_base':
        return get_convnext_base(2)


def train_sweep(model, optimizer, loader_train, loader_val, epochs=1, print_every=100, device=torch.device('cpu'),
                scheduler=None):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    wandb.watch(model)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    count = 0
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            count = count + 1

            if count % print_every == 0:
                print('Iteration %d, loss = %.4f' % (count, loss.item()))
                acc_val = check_accuracy_part34(loader_val, model)
                acc_train = check_accuracy_part34(loader_train, model)
                wandb.log({"val_acc": acc_val, "train_acc": acc_train, "loss": loss, "epo": e})
                print()

        if scheduler is not None:
            scheduler.step()


def train(Config=None):
    with wandb.init(config=Config):
        config = wandb.config
        loader_train, loader_val = get_loader(config["batch_size"], config["root"])
        network = build_model(config["model"])
        optimizer = build_optimizer(network, config["optimizer"], config["lr"], config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
        train_sweep(network, optimizer, loader_train, loader_val, config["epochs"], 100, device, scheduler)
        wandb.save(network, "model")


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, train, count=50)