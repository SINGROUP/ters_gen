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



def focal_loss(logits:torch.Tensor, labels:torch.Tensor, gamma:float=2.0, alpha:float=0.25):
    """ 
    Focal loss for multi-class classification
    Args:
        logits: Tensor of shape [D, H, W, num_classes] (or any shape where the last dim is classes).
        labels: Tensor of shape [D, H, W] with integer class labels.
        gamma: Focusing parameter.
        alpha: Balancing parameter.
        
    Returns:
        Mean focal loss.
    """

    num_classes = logits.shape[-1]
    logits = logits.view(-1, num_classes)
    labels = labels.view(-1)


    # Compute the softmax probabilities
    probs = F.softmax(logits, dim=-1)

    # Create one-hot encoding of labels
    one_hot_labels = F.one_hot(labels, num_classes).float()

    # Get probabilities corresponding to the true class
    probs = (probs*one_hot_labels).sum(dim=1)

    # For numerical stability, clamp probabilities
    probs = probs.clamp(min=1e-9, max=1.0)

    loss = -alpha*(1-probs)**gamma*torch.log(probs)
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
    else:
        raise ValueError(f"Loss function: {loss_fn} not supported")
