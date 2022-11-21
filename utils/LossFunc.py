import torch


def tversky_loss(preds, targets, beta=0.7):
    assert preds.size() == targets.size(), "the size of predict and target must be equal."

    preds = torch.clamp(preds, min=1e-7, max=1 - 1e-7).to(torch.float32)

    batch_size = preds.size(0)
    channels = preds.size(1)

    assert 0 < channels < 3, "channel should more than 0 and less than 3"

    loss = 0.
    epsilon = 1e-5
    alpha = 1.0 - beta

    if channels == 1:
        for i in range(batch_size):
            pred = preds[i]
            target = targets[i]

            tp = (target * pred).sum()
            fp = ((1 - target) * pred).sum()
            fn = (target * (1 - pred)).sum()
            tversky = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
            loss += (1 - tversky)
    else:
        ratio = [0.05, 1.95]
        for i in range(batch_size):
            for j in range(channels):
                pred = preds[i, j]
                target = targets[i, j]

                tp = (target * pred).sum()
                fp = ((1 - target) * pred).sum()
                fn = (target * (1 - pred)).sum()
                tversky = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
                loss += ratio[j] * (1 - tversky)

    loss = loss / (batch_size * channels)

    return loss


def dice_loss(preds, targets):
    assert preds.size() == targets.size(), "the size of predict and target must be equal."

    preds = torch.clamp(preds, min=1e-7, max=1 - 1e-7).to(torch.float32)

    batch_size = preds.size(0)
    loss = 0.
    epsilon = 1e-5

    for i in range(batch_size):
        pred = preds[i].flatten()
        target = targets[i].flatten()

        intersection = (pred * target).sum()
        union = (pred + target).sum()
        loss += 1 - 2 * (intersection + epsilon) / (union + epsilon)
    return loss / batch_size


if __name__ == "__main__":
    pred = torch.Tensor([[[
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0]]]],
        [[[[0, 1, 1, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 1, 0]]]]])

    gt = torch.Tensor([[[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]]]],
        [[[[0, 0, 0, 0],
           [0, 1, 1, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 0]]]]])

    print(pred.shape, gt.shape)
    print(dice_loss(pred, gt))
