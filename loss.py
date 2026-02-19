import torch
import torch.nn.functional as F


def zncc_loss(I_pred, I_actual, eps=1e-8):
    mean_pred = torch.mean(I_pred, dim=[2, 3], keepdim=True)
    mean_actual = torch.mean(I_actual, dim=[2, 3], keepdim=True)
    std_pred = torch.std(I_pred, dim=[2, 3], keepdim=True) + eps
    std_actual = torch.std(I_actual, dim=[2, 3], keepdim=True) + eps
    zncc_map = (I_pred - mean_pred) * (I_actual - mean_actual) / (std_pred * std_actual)
    zncc_score = torch.mean(zncc_map, dim=[2, 3])
    loss = 1.0 - zncc_score.mean()
    return loss


# ---------- finite difference ----------
def _diff_kernels(device, dtype):
    kx = torch.tensor([[-0.5, 0.0, 0.5]], device=device, dtype=dtype).view(1, 1, 1, 3)
    ky = torch.tensor([[-0.5], [0.0], [0.5]], device=device, dtype=dtype).view(1, 1, 3, 1)
    kxx = torch.tensor([[1.0, -2.0, 1.0]], device=device, dtype=dtype).view(1, 1, 1, 3)
    kyy = torch.tensor([[1.0], [-2.0], [1.0]], device=device, dtype=dtype).view(1, 1, 3, 1)
    return kx, ky, kxx, kyy


def d_dx(f):
    kx, _, _, _ = _diff_kernels(f.device, f.dtype)
    return F.conv2d(f, kx, padding=(0, 1))


def d_dy(f):
    _, ky, _, _ = _diff_kernels(f.device, f.dtype)
    return F.conv2d(f, ky, padding=(1, 0))


def d2_dx2(f):
    _, _, kxx, _ = _diff_kernels(f.device, f.dtype)
    return F.conv2d(f, kxx, padding=(0, 1))


def d2_dy2(f):
    _, _, _, kyy = _diff_kernels(f.device, f.dtype)
    return F.conv2d(f, kyy, padding=(1, 0))


def d2_dxdy(f):
    return d_dy(d_dx(f))


def _masked_mean(x, mask):
    # x, mask: [B,1,H,W] (mask can be float/bool)
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def split_pred(pred):
    """
    pred: [B,5,H,W]
    return u,v,exx,eyy,exy each [B,1,H,W]
    """
    if pred.shape[1] != 5:
        raise ValueError(f"Expect pred with 5 channels [u,v,exx,eyy,exy], got {pred.shape}")
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    exx = pred[:, 2:3]
    eyy = pred[:, 3:4]
    exy = pred[:, 4:5]
    return u, v, exx, eyy, exy


def physics_loss_eq13(pred, roi_mask=None):
    """
    MFPINN-DIC Eq.(13):
      mean_{(i,j) in Ω} [(exx_pred - du/dx)^2 + (eyy_pred - dv/dy)^2 + (exy_pred - 0.5(du/dy + dv/dx))^2]

    CNN version: derivatives computed by finite differences on pixel grid.
    """
    u, v, exx_p, eyy_p, exy_p = split_pred(pred)

    du_dx = d_dx(u)
    du_dy = d_dy(u)
    dv_dx = d_dx(v)
    dv_dy = d_dy(v)

    exx_bm = du_dx
    eyy_bm = dv_dy
    exy_bm = 0.5 * (du_dy + dv_dx)

    per_pixel = (exx_p - exx_bm) ** 2 + (eyy_p - eyy_bm) ** 2 + (exy_p - exy_bm) ** 2

    if roi_mask is None:
        return per_pixel.mean()
    return _masked_mean(per_pixel, roi_mask)


def compatibility_loss_eq3(pred, roi_mask=None):
    """
    MCNN-DIC Eq.(3):
      d2(exy)/dxdy = 0.5( d2(exx)/dy2 + d2(eyy)/dx2 )
    """
    _, _, exx, eyy, exy = split_pred(pred)
    left = d2_dxdy(exy)
    right = 0.5 * (d2_dy2(exx) + d2_dx2(eyy))
    res2 = (left - right) ** 2

    if roi_mask is None:
        return res2.mean()
    return _masked_mean(res2, roi_mask)


def compute_warmup_loss(I_pred, I_actual):
    return zncc_loss(I_pred, I_actual)


def compute_main_loss(I_pred, I_actual, pred, roi_mask=None, w_phy=1e-4, w_cmp=0.0):
    """
    total = zncc + w_phy * physics(Eq13) + w_cmp * compat(Eq3, optional)
    """
    L_img = zncc_loss(I_pred, I_actual)
    L_phy = physics_loss_eq13(pred, roi_mask=roi_mask)

    if w_cmp > 0:
        L_cmp = compatibility_loss_eq3(pred, roi_mask=roi_mask)
    else:
        L_cmp = torch.zeros((), device=I_pred.device, dtype=I_pred.dtype)

    return L_img + w_phy * L_phy + w_cmp * L_cmp
