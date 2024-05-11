import torch
import numpy
from numpy.random import randint


def pairwise_distance(data1, data2, metric='euclidean',
                      self_nearest=True, all_negative=False, p=2.0):
    """
    Return:
        a distance matrix [N1,N2] or [B,N1,N2]
    """
    if metric == 'euclidean':
        dis = torch.cdist(data1, data2, p=p)

    elif metric == 'cosine':
        A_norm = data1 / (data1.norm(dim=-1, keepdim=True) + 1e-6)
        B_norm = data2 / (data2.norm(dim=-1, keepdim=True) + 1e-6)
        if data1.ndim == 3:
            dis = 1.0 - torch.bmm(A_norm, B_norm.transpose(-2, -1))
        else:
            dis = 1.0 - torch.matmul(A_norm, B_norm.transpose(-2, -1))

    else:
        raise NotImplementedError("{} metric is not implemented".format(metric))

    if all_negative:
        dis = dis - torch.max(dis) - 1.0

    if self_nearest:
        # avoid two same points
        diag = torch.arange(dis.shape[-1], device=dis.device, dtype=torch.long)
        dis[..., diag, diag] -= 1.0

    return dis


def KKZ_init(X, distance_matrix, K, batch=False):
    """
    KKZ initilization for kmeans
    1. Choose the point with the maximum L2-norm as the first centroid.
    2. For j = 2, . . . ,K, each centroid μj is set in the following way: For
    any remaining data xi, we compute its distance di to the existing cen-
    troids. di is calculated as the distance between xi to its closest existing
    centroid. Then, the point with the largest di is selected as μj .

    Reference:
        I. Katsavounidis, C.-C. J. Kuo, and Z. Zhang. A new initialization tech-
        nique for generalized Lloyd iteration. IEEE Signal Processing Letters,
        1(10):144–146, 1994.

    """
    l2_norm = torch.norm(X, dim=-1)
    if not batch:
        medoids = torch.arange(K, device=distance_matrix.device, dtype=torch.long)
        _, medoids[0] = torch.max(l2_norm, dim=0)
        for i in range(1, K):
            sub_dis_matrix = distance_matrix[:, medoids[:i]]
            # print(sub_dis_matrix.shape)
            values, indices = torch.min(sub_dis_matrix, dim=1)
            medoids[i] = torch.argmax(values, dim=0)

        # import pdb; pdb.set_trace()
        return medoids

    else:
        # batch version
        batch_i = torch.arange(X.shape[0], dtype=torch.long, device=X.device).unsqueeze(1)
        medoids = torch.arange(K, device=distance_matrix.device, dtype=torch.long)
        medoids = medoids.unsqueeze(0).repeat(X.shape[0], 1)
        _, medoids[:, 0] = torch.max(l2_norm, dim=1)
        for i in range(1, K):
            sub_dis_matrix = distance_matrix[batch_i, medoids[:, :i], :]  # [B, i, N]
            values, indices = torch.min(sub_dis_matrix, dim=1)  # [B, N]
            values_, indices_ = torch.max(values, dim=1)  # [B]
            medoids[:, i] = indices_

        return medoids
