

def increase_head_lr(model, lr, increase_koeff):
    params = list(model.named_parameters())

    def _is_backbone(n):
        return "bert" in n

    optimizer_grouped_parameters = [
        {"params": [p for n, p in params if _is_backbone(n)], "lr": lr},
        {"params": [p for n, p in params if not _is_backbone(n)], "lr": lr * increase_koeff},
    ]

    return optimizer_grouped_parameters