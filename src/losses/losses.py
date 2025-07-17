import torch
import torch.nn as nn
import torch.nn.functional as F

def mse(y_pred:torch.Tensor, y_true:torch.Tensor):
    return F.mse_loss(y_pred, y_true)

def mae(y_pred:torch.Tensor, y_true:torch.Tensor):
    return F.l1_loss(y_pred, y_true)

def kl_div(logits:torch.Tensor, targets:torch.Tensor):
    return F.kl_div(F.softmax(logits, dim=-1), targets)

def cross_entropy(logits:torch.Tensor, targets:torch.Tensor):
    # Flatten the logits and the targets is optional
    num_classes = logits.shape[-1]
    logits = logits.view(-1, num_classes)
    targets = targets.view(-1)
    return F.cross_entropy(logits, targets)




def focal_loss(pred, target, alpha = 1.0, gamma = 2.0, reduction = 'mean'):
    '''
    pred: [B,C,H,W] tensor of raw model outputs (logits)
    target: [B,C,H,W] or [B,H,W] tensor of ground truth binary masks (as one - hot)
    alpha: weighting factor for rare classes (default 1.0 = no weighting)
    gamma: focusing parameter to reduce the relative loss for well-classified examples (default 2.0)
    reduction: specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    '''

    pred_soft = torch.softmax(pred, dim = 1) # [B,C,H,W] tensor of probabilities

    if target.dim() == 3:  # If target is [B, H, W], convert to [B, C, H, W]
        target_one_hot = F.one_hot(target, num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # Convert to [B, C, H, W]
    else:
        target_one_hot = target.float()

    # Clamp to prevent log(0)
    pred_soft = pred_soft.clamp(min=1e-7, max=1.0)

    # Compute the focal loss
    ce_loss = -target_one_hot * torch.log(pred_soft) 
    focal_term = (1 - pred_soft) ** gamma
    alpha = torch.tensor([38.2951, 35.3427, 45.6857, 45.2584,  1.4423], device = pred.device)
    alpha = alpha.view(1, -1, 1, 1)  # Reshape to match [B, C, H, W]
    loss = alpha * focal_term * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def dice_loss_multi_class(pred, target, smooth=1e-6):

    pred = torch.softmax(pred, dim=1)  # Apply softmax to get probabilities

    # Flatten the predictions and target masks per batch
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    # Calculate the intersection and union
    intersection = (pred_flat * target_flat).sum(2)
    union = pred_flat.sum(2) + target_flat.sum(2)

    # Compute the Dice coefficient and then the Dice loss
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    #weighted_dice = dice_score
    weighted_dice = 0.99*dice_score[:-1].mean() + 0.01*dice_score[-1].mean()
    loss = 1 - weighted_dice
    #loss = 1 - dice_score.mean()
    return loss.mean()



def dice_loss(pred, target, smooth=1e-6):
    """
    pred: predicted probabilities after sigmoid with shape [N, C, H, W]
    target: ground truth binary masks with shape [N, C, H, W]
    """
    # Apply sigmoid to obtain probabilities in the [0, 1] range
    pred = torch.sigmoid(pred)
    
    # Flatten the predictions and target masks per batch
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Calculate the intersection and union
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    
    # Compute the Dice coefficient and then the Dice loss
    dice_score = (2.0 * intersection + smooth) / (union + smooth)

    
    loss = 1 - dice_score.mean()
    return loss

'''def dice_loss(logits:torch.Tensor, labels:torch.Tensor, eps:float=1e-6):
    """
    Dice loss for multi-class segmentation
    Args:
        logits: Tensor of shape [D, H, W, num_classes] (or any shape where the last dim is classes).
        labels: Tensor of shape [D, H, W] with integer class labels.
        epsilon: Small constant to avoid division by zero.
        
    Returns:
        Mean Dice loss.
    """

    logits = logits.permute(0, 2, 3, 1)
    num_classes = logits.shape[-1]
    probs = F.softmax(logits, dim=-1)
    one_hot_labels = F.one_hot(labels, num_classes).float()


    # Sum over all dimensions except the classes dimension
    dims = list(range(len(probs.shape) - 1))
    intersection = torch.sum(probs * one_hot_labels, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(one_hot_labels, dim=dims)

    dice_per_class = (2.0 * intersection + eps) / (union + eps)
    mean_dice_loss = 1.0 - dice_per_class.mean()
    return mean_dice_loss
'''


def bce_loss(pred, target, smooth=1e-6):
    """
    logits: raw model outputs with shape [N, C, H, W]
    target: ground truth binary masks with shape [N, C, H, W]
    smooth: small constant to prevent log(0)
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Flatten per channel
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    # Clamp predictions to avoid numerical instability
    pred_flat = pred_flat.clamp(smooth, 1. - smooth)
    
    # Calculate BCE loss: -[y*log(p) + (1-y)*log(1-p)]
    loss = -(target_flat * torch.log(pred_flat) + (1. - target_flat) * torch.log(1. - pred_flat))
    
    # Average over all dimensions (pixels, channels, and batch)
    return loss.mean()

def get_loss_function(loss_fn:str):


    if loss_fn == "mse":
        return mse
    elif loss_fn == "mae":
        return mae
    elif loss_fn == "kl_div":
        return kl_div
    elif loss_fn == "cross_entropy":
        return cross_entropy
    elif loss_fn == "bce_loss":
        return bce_loss
    elif loss_fn == "focal_loss":
        return focal_loss
    elif loss_fn == "dice_loss":
        return dice_loss
    elif loss_fn == "dice_loss_multi_class":
        return dice_loss_multi_class
    else:
        raise ValueError(f"Loss function: {loss_fn} not supported")
