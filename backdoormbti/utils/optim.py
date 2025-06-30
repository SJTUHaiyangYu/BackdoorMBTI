import torch


def get_optimizer(client_optimizer, params, **kwargs):
    if client_optimizer == "adam":
        optimizer = torch.optim.Adam(params, **kwargs)
    elif client_optimizer == "sgd":
        optimizer = torch.optim.SGD(params, **kwargs, momentum=0.9)
    elif client_optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, **kwargs)
    else:
        raise NotImplementedError("Optimizer %s not supported." % client_optimizer)
    return optimizer


def get_lr_scheduler(lr_scheduler, optimizer, args):
    if lr_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        raise NotImplementedError(
            "LearningRate Scheduler %s not supported." % lr_scheduler
        )
    return scheduler
