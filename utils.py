import torch
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def add_coordinate_channels(tensor):

    B, C, H, W = tensor.shape
    device = tensor.device

    x_coords = torch.linspace(-1, 1, W, device=device)
    y_coords = torch.linspace(-1, 1, H, device=device)

    x_grid = x_coords.repeat(H, 1)
    y_grid = y_coords.unsqueeze(1).repeat(1, W)

    x_channel = x_grid.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)
    y_channel = y_grid.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)

    # return coords too (for compatibility with your call sites), but they are not used in loss anymore
    return torch.cat([tensor, x_channel, y_channel], dim=1), x_coords, y_coords


def _extract_uv(pred):

    if pred.shape[1] >= 2:
        u = pred[:, 0:1]
        v = pred[:, 1:2]
        return u, v
    raise ValueError(f"pred must have at least 2 channels, got {pred.shape}")


def warp_image_with_displacement(reference_img, pred):
    """
    Warp the reference image using predicted displacement u,v.
    Supports pred with 2ch or 5ch.
    padding_mode='zeros' for MPS compatibility.
    """
    B, C, H, W = reference_img.shape
    u, v = _extract_uv(pred)

    y_coords = torch.linspace(-1, 1, H, device=reference_img.device)
    x_coords = torch.linspace(-1, 1, W, device=reference_img.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    base_grid = torch.stack((x_grid, y_grid), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)

    # normalized flow for grid_sample
    norm_disp_u = u[:, 0] * (2.0 / W)
    norm_disp_v = -v[:, 0] * (2.0 / H)  # keep your sign convention

    flow = torch.stack([norm_disp_u, norm_disp_v], dim=-1)  # [B,H,W,2]
    warped_grid = base_grid + flow

    warped_img = F.grid_sample(
        reference_img,
        warped_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return warped_img


def save_displacement_to_csv(pred, filename="outputs/displacement.csv", roi_mask=None):
    """
    Save u,v only. pred can be [B,5,H,W] or [B,2,H,W].
    """
    if roi_mask is not None:
        pred = pred.clone()
        pred[:, 0:2] = pred[:, 0:2] * roi_mask.repeat(1, 2, 1, 1)

    u = pred[0, 0].detach().cpu().numpy()
    v = pred[0, 1].detach().cpu().numpy()
    H, W = u.shape
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "u", "v"])
        for i in range(H):
            for j in range(W):
                writer.writerow([i, j, u[i, j], v[i, j]])
    print(f"Displacement saved to {filename}")


def visualize_displacement(pred, cmap='viridis', save_path=None):
    """
    Visualize u,v only.
    """
    u = pred[0, 0].detach().cpu().numpy()
    v = pred[0, 1].detach().cpu().numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=300, constrained_layout=True)
    norm_u = mcolors.Normalize(vmin=u.min(), vmax=u.max())
    norm_v = mcolors.Normalize(vmin=v.min(), vmax=v.max())

    im0 = ax0.imshow(u, cmap=cmap, interpolation='bilinear', norm=norm_u)
    fig.colorbar(im0, ax=ax0)
    ax0.set_title("Displacement u")
    ax0.axis('off')

    im1 = ax1.imshow(v, cmap=cmap, interpolation='bilinear', norm=norm_v)
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("Displacement v")
    ax1.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    plt.close()


def plot_loss_curve(losses, save_path=None, smooth_window=1, title='Training Loss Curve'):
    losses = np.array(losses)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        losses_s = np.convolve(losses, kernel, mode='valid')
        x = np.arange(len(losses_s))
    else:
        losses_s = losses
        x = np.arange(len(losses_s))

    plt.figure(figsize=(8, 5))
    plt.plot(x, losses_s, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Loss curve saved to {save_path}")
    plt.close()


def compute_benchmark_strain_from_uv(pred):
    """
    Compute benchmark strain from predicted u,v using torch.gradient (sanity check).
    pred: [B,5,H,W] or [B,2,H,W]
    Return exx, eyy, exy as [H,W] (first batch).
    """
    u = pred[0, 0]
    v = pred[0, 1]

    du_dx = torch.gradient(u, dim=1)[0]
    du_dy = torch.gradient(u, dim=0)[0]
    dv_dx = torch.gradient(v, dim=1)[0]
    dv_dy = torch.gradient(v, dim=0)[0]

    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)
    return exx, eyy, exy


def save_all_results(pred, reference, save_prefix_csv, save_prefix_img, save_benchmark=True):
    """
    pred: [B,5,H,W] (recommended) or [B,2,H,W]
    Saves:
      - displacement csv (u,v)
      - strain npy (predicted exx,eyy,exy if available)
      - figure with u,v,exx,eyy,exy
      - optional: benchmark strain from u,v as another npy
    """
    save_displacement_to_csv(
        pred,
        filename=f"{save_prefix_csv}_displacement.csv"
    )

    has_pred_strain = pred.shape[1] >= 5
    if has_pred_strain:
        exx_p = pred[0, 2].detach().cpu()
        eyy_p = pred[0, 3].detach().cpu()
        exy_p = pred[0, 4].detach().cpu()
        strain_np = torch.stack([exx_p, eyy_p, exy_p]).numpy()
        np.save(f"{save_prefix_csv}_strain_pred.npy", strain_np)

    if save_benchmark:
        exx_b, eyy_b, exy_b = compute_benchmark_strain_from_uv(pred)
        strain_bm = torch.stack([exx_b, eyy_b, exy_b]).detach().cpu().numpy()
        np.save(f"{save_prefix_csv}_strain_benchmark.npy", strain_bm)

    u = pred[0, 0].detach().cpu()
    v = pred[0, 1].detach().cpu()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    def show(ax, data, title):
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axes[0], u, "U displacement")
    show(axes[1], v, "V displacement")

    if has_pred_strain:
        show(axes[2], exx_p, "εxx (pred)")
        show(axes[3], eyy_p, "εyy (pred)")
        show(axes[4], exy_p, "εxy (pred)")
        axes[5].axis("off")
    else:
        exx_b, eyy_b, exy_b = compute_benchmark_strain_from_uv(pred)
        show(axes[2], exx_b.detach().cpu(), "εxx (benchmark)")
        show(axes[3], eyy_b.detach().cpu(), "εyy (benchmark)")
        show(axes[4], exy_b.detach().cpu(), "εxy (benchmark)")
        axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_prefix_img}_displacement_strain.png", dpi=300)
    plt.close()
    print("Displacement & strain figure saved.")
