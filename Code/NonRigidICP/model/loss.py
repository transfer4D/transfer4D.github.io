from typing import Union
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
import matplotlib.pyplot as plt

def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals

def compute_truncated_chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        trunc=0.2,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # truncation
    x_mask[cham_x >= trunc] = True
    y_mask[cham_y >= trunc] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0


    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    # cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist,x_nn.idx[...,0],~x_mask


def arap_cost (R, t, g, e, w, lietorch=True):
    '''
    :param R:
    :param t:
    :param g:
    :param e:
    :param w:
    :return:
    '''

    R_i = R [:, None]
    g_i = g [:, None]
    t_i = t [:, None]

    g_j = g [e]
    t_j = t [e]

    # print(R.shape,t.shape,g.shape,e.shape,w.shape)
    # print(R_i.shape,g_i.shape,t_i.shape,g_j.shape,t_j.shape)

    if lietorch :
        # print((g_j - g_i).shape)
        # print( (g_i + t_i  - g_j - t_j).shape)
        # print((R_i * (g_j - g_i) + g_i + t_i  - g_j - t_j ).shape)
        e_ij = ((R_i * (g_j - g_i) + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)
    else :
        e_ij = (((R_i @ (g_j - g_i)[...,None]).squeeze() + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)

    o = (w * e_ij ).sum()

    return o


def projective_depth_cost(dx, dy):

    x_mask = dx> 0
    y_mask = dy> 0
    depth_error_image = (dx - dy) ** 2
    depth_error = depth_error_image[y_mask * x_mask]

    silh_loss = torch.mean(depth_error)

    return silh_loss,(dx - dy) ** 2

def projective_depth_inverse_cost(dx, dy):

    diff_img = torch.abs(dx - dy) 

    x_mask = dx > 0
    y_mask = dy > 0

    # If z > 0, inverse = 1/z else 0
    dx_inverse = dx
    dx_inverse[x_mask] = 1/dx[x_mask]

    dy_inverse = dy
    dy_inverse[y_mask] = 1/dy[y_mask]


    assert torch.isfinite(dx_inverse).any(), f"Depth Image is 0 for values of X, cannot calculate inverse" 
    assert torch.isfinite(dy_inverse).any(), f"Depth Image is 0 for values of X, cannot calculate inverse" 


    depth_error_image = (dx_inverse - dy_inverse) ** 2
    depth_error = torch.mean(depth_error_image)

    return depth_error,diff_img


def iou(x, y):

    x_mask = x[..., 0] > 0
    y_mask = y[..., 0] > 0


    silh_loss = torch.logical_and(x_mask, y_mask).sum()/(torch.logical_or(x_mask, y_mask).sum() + 1e-6)


    return silh_loss

def silhouette_cost(x, y):
    x_mask = x[..., 0] > 0
    y_mask = y[..., 0] > 0
    silh_error = (x - y) ** 2
    silh_error = silh_error[~y_mask]
    print(silh_error)
    if len(silh_error) == 0:
        return 1e-6
    silh_loss = torch.mean(silh_error)

    return silh_loss    

def landmark_cost(x, y, landmarks):
    x = x [ landmarks[0] ]
    y = y [ landmarks[1] ]
    loss = torch.sum(
        torch.sum( (x-y)**2, dim=-1 ))
    return loss

def landmark_cost_with_conf(x, y, landmarks,conf):
    x = x [ landmarks[0] ]
    y = y [ landmarks[1] ]
    loss = torch.sum(
        torch.sum( conf[:,None]*(x-y)**2, dim=-1 ))
    return loss


def chamfer_dist(src_pcd,tgt_pcd,samples=5000):
    '''
    :param src_pcd: warpped_pcd
    :param R: node_rotations
    :param t: node_translations
    :param data:
    :return:
    '''

    """chamfer distance"""
    src=torch.randperm(src_pcd.shape[0])
    tgt=torch.randperm(tgt_pcd.shape[0])
    s_sample = src_pcd[ src[:samples]]
    t_sample = tgt_pcd[ tgt[:samples]]
    cham_dist,corresp,valid_source_points = compute_truncated_chamfer_distance(s_sample[None], t_sample[None], trunc=0.3)

    corresp = corresp[0]
    valid_source_points = valid_source_points[0]

    corresp = torch.cat([src[:samples][:,None],tgt[:samples][corresp][:,None]],axis=-1)
    corresp = corresp[valid_source_points]

    return cham_dist,corresp


def occlusion_fusion_graph_motion_cost(nodes,t,
                    target_graph_node_location,
                    target_graph_node_confidence):

    per_node_loss =  target_graph_node_location[:,None]**2 * (nodes + t - target_graph_node_location)**2 

    # print("Per node motion loss:",per_node_loss)
    # print("Shape:",per_node_loss.shape)

    return torch.mean(per_node_loss)