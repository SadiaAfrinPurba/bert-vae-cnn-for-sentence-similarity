from torch import optim


def get_adam_optimizer(model, lr: float):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer
