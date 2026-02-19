import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import save_displacement_to_csv


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

    # predicted strain (if 5ch)
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
