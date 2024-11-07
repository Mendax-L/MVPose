import torch
import torch.nn as nn
from torch.nn import functional as F

def criterion_R(output, target, alpha=0.3):
    # Normalize both vectors to ensure that we're only comparing directions
    # cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
    # Compute the cosine similarity
    # Compute the cosine similarity
    cosine_sim_1 = torch.sum(output[:,:3] * target[:,:3], dim=1)
    cosine_sim_2 = torch.sum(output[:,3:6] * target[:,3:6], dim=1)
    # cosine_sim_3 = torch.sum(output[:,6:] * target[:,6:], dim=1)
    
    # Compute the angular distance (arccos of cosine similarity)
    angle_1 = torch.acos(torch.clamp(cosine_sim_1, -1 + 1e-7, 1 - 1e-7))  # Clamp to avoid numerical errors
    angle_2 = torch.acos(torch.clamp(cosine_sim_2, -1 + 1e-7, 1 - 1e-7))  # Clamp to avoid numerical errors
    # angle_3 = torch.acos(torch.clamp(cosine_sim_3, -1 + 1e-7, 1 - 1e-7))  # Clamp to avoid numerical errors


    # The loss is the mean of the angle (in radians)
    loss = alpha * torch.mean(angle_1+angle_2) + (1 - alpha) * F.l1_loss(output, target, reduction='mean')  # Mean angle across all samples
    return loss

criterion_uv = nn.L1Loss(reduction='mean')
