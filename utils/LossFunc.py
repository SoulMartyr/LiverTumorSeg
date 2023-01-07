import torch
import torch.nn.functional as F


def tversky_loss(preds, targets, beta=0.7):
    assert preds.size() == targets.size(), "the size of predict and target must be equal."

    batch_size = preds.size(0)
    channels = preds.size(1)

    assert channels == 2 or channels == 3, "channel should be 2 or 3"

    loss = 0.
    epsilon = 1e-5
    alpha = 1.0 - beta

    if channels == 2:
        ratio = [0.78, 9.22]
    else:
        ratio = [0.78, 0.65, 8.57]

    preds = F.softmax(preds, dim=1)
    preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

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


def weighted_cross_entropy_loss(preds, targets):
    assert preds.size() == targets.size(), "the size of predict and target must be equal."

    batch_size = preds.size(0)
    channels = preds.size(1)

    assert channels == 2 or channels == 3, "channel should be 2 or 3"

    # loss = torch.FloatTensor([0]).sum().to(preds.device)
    #
    # if channels == 2:
    #     ratio = [0.78, 9.22]
    # else:
    #     ratio = [0.78, 0.65, 8.57]
    #
    # preds = F.softmax(preds, dim=1)
    # print(preds)
    # preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)
    #
    # for i in range(batch_size):
    #     for j in range(channels):
    #         pred = preds[i, j].flatten()
    #         target = targets[i, j].flatten()
    #
    #         pred = pred[target == 1]
    #         if len(pred) == 0:
    #             tmp_loss = torch.FloatTensor([0]).sum().to(pred.device)
    #         else:
    #             tmp_loss = - torch.log(pred).mean()
    #
    #         loss += ratio[j] * tmp_loss
    #
    # loss = loss / (batch_size * channels)
    # return loss
    if channels == 2:
        return F.cross_entropy(input=preds, target=targets, weight=torch.tensor((0.78, 9.22), device=preds.device),
                               reduction='mean')
    else:
        return F.cross_entropy(input=preds, target=targets,
                               weight=torch.tensor((0.78, 0.65, 8.57), device=preds.device), reduction='mean')


def dice_loss(preds, targets):
    assert preds.size() == targets.size(), "the size of predict and target must be equal."

    batch_size = preds.size(0)
    loss = 0.
    epsilon = 1e-5

    preds = F.softmax(preds, dim=1)
    preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

    for i in range(batch_size):
        pred = preds[i].flatten()
        target = targets[i].flatten()

        intersection = (pred * target).sum()
        union = (pred + target).sum()
        loss += 1 - 2 * (intersection + epsilon) / (union + epsilon)
    return loss / batch_size


def liver_dice(preds, targets, threshold=0.5):
    assert preds.size() == targets.size(), "the size of predict and target must be equal."

    batch_size = preds.size(0)
    tot_dice = 0.
    epsilon = 1e-5

    preds = F.softmax(preds, dim=1)
    preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0
    for i in range(batch_size):
        pred = preds[i, 1].flatten()
        target = targets[i, 1].flatten()

        intersection = (pred * target).sum()
        union = (pred + target).sum()
        tot_dice += 2 * intersection / (union + epsilon)

    return tot_dice / batch_size


def tumor_dice(preds, targets, threshold=0.5):
    assert preds.size() == targets.size(), "the size of predict and target must be equal."

    batch_size = preds.size(0)
    tot_dice = 0.
    epsilon = 1e-5

    preds = F.softmax(preds, dim=1)
    preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0

    for i in range(batch_size):
        pred = preds[i, 2].flatten()
        target = targets[i, 2].flatten()

        intersection = (pred * target).sum()
        union = (pred + target).sum()
        tot_dice += 2 * (intersection + epsilon) / (union + epsilon)

    return tot_dice / batch_size


if __name__ == "__main__":
    pred = torch.Tensor([[[
        [[0, 1, 1, 0],
         [0.8, 0, 0, 0],
         [0.8, 0, 0, 0],
         [0, 1, 1, 0]]],
        [[[0, 0, 0, 0],
          [0.1, 0.9, 0, 0],
          [0.1, 0.9, 0, 0],
          [0, 1, 1, 0]]]],
        [[
            [[0, 1, 1, 0],
             [0.8, 0, 0, 0],
             [0.8, 0, 0, 0],
             [0, 1, 1, 0]]],
            [[[0, 0, 0, 0],
              [0.1, 0.9, 0, 0],
              [0.1, 0.9, 0, 0],
              [0, 1, 1, 0]]]]])

    gt = torch.Tensor([[[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]]],
        [[[1, 0, 0, 1],
          [0, 1, 1, 0],
          [0, 1, 1, 0],
          [1, 0, 0, 1]]]],
        [[
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]]],
            [[[1, 0, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [1, 0, 0, 1]]]]])

    print(pred.shape, gt.shape)
    print(weighted_cross_entropy_loss(pred, gt))
