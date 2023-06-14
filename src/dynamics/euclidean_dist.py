"""
Compute the Euclidean distance for prior computation.
"""
import numpy
import torch
import time
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def euclidean_dist(args, train_embeds, train_labels, train_labels2=None):
    print(train_labels.shape)
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    l_max = torch.max(train_labels)
    cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    for i in range(l_max+1):
        cluster_centroids[i] = torch.mean(train_embeds[train_labels==i], 0)
    embeds1 = train_embeds.unsqueeze(1).repeat((1, cluster_centroids.shape[0], 1))
    embeds2 = cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0], 1, 1))
    dists = torch.sqrt(torch.sum((embeds1.to(embeds2.device) - embeds2) ** 2, -1)).to(embeds1.device)
    print(dists.shape)
    torch.cuda.empty_cache()
    return dists

def euclidean_dist_wos(args, train_embeds, train_labels, train_labels2=None):
    print(train_labels.shape)
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    l_max = torch.max(train_labels)
    # cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    dists = torch.zeros((l_max+1, train_embeds.shape[0]))
    for i in range(l_max+1):
        cluster_centroids = torch.mean(train_embeds[train_labels==i], 0)
        cluster_centroids = cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0], 1))
        dists[i] = torch.sqrt(torch.sum((train_embeds.to(cluster_centroids.device) - cluster_centroids) ** 2, -1)).squeeze().to(train_embeds.device)
    dists = dists.permute(1, 0)
    print(dists.shape)
    torch.cuda.empty_cache()
    return dists