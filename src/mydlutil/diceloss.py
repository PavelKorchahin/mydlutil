# dice loss
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss
    Args:
        smooth (float, optional): A float number to smooth loss to avoid NaN error
            Default is 1e-5
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient
    """
    def __init__(self, smooth: float = 1e-5, ignore_index: int = None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] Tensor of raw logits
            targets: [B, H, W] Tensor of targets
        Returns:
            dice loss
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)   # [B, C, H, W]

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)   # [B, H, W]

            safe_targets = targets.clone()

            safe_targets[~valid_mask] = 0

            targets_onehot = F.one_hot(safe_targets, num_classes=num_classes)  # [B, H, W, C]
            targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()        # [B, C, H, W]

            valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]

            probs = probs * valid_mask
            targets_onehot = targets_onehot * valid_mask

        else:
            targets_onehot = F.one_hot(targets, num_classes=num_classes)
            targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        union = probs.sum(dims) + targets_onehot.sum(dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

class DiceCELoss(nn.Module):
    """
    Dice Cross Entropy Loss
    Args:
        weight (list or tensor, optional): A manual rescaling weight given to each class.
            If given, has to be a Tensor of size C
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient
    """
    def __init__(self, weight=None, ignore_index=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] Tensor of raw logits
            targets: [B, H, W] Tensor of targets
        Returns:
            dice cross entropy loss
        """
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + dice_loss