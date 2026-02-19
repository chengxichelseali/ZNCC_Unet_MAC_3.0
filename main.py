import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from cnn_model import DisplacementCNN
from utils import add_coordinate_channels, save_all_results
import train



reference_path = "/Users/chelseali/Downloads/UNET_ZNCC_NO_ROI-main/sin_0_imagesize512.jpg"
deformed_path = "/Users/chelseali/Downloads/UNET_ZNCC_NO_ROI-main/sin64_imagesize512_u_1.jpg"

loss_tag = "sin zncc"
output_dir = f"/Users/chelseali/Downloads/UNET_ZNCC_NO_ROI-main/outputs/{loss_tag}"
os.makedirs(output_dir, exist_ok=True)

output_csv_prefix = os.path.join(output_dir, "sin641e-4")
output_img_prefix = os.path.join(output_dir, "sin641e-4")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_grayscale_image(path):
    img = Image.open(path).convert("L")
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    return img_tensor.unsqueeze(0).unsqueeze(0)


def crop_center_to_multiple(tensor, factor=8):
    _, _, H, W = tensor.shape
    H_new = H - (H % factor)
    W_new = W - (W % factor)
    top = (H - H_new) // 2
    left = (W - W_new) // 2
    return tensor[:, :, top:top+H_new, left:left+W_new]


def plot_loss_curve(losses, save_path=None, smooth_window=1, title="Training Loss Curve"):
    losses = np.array(losses)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        losses_s = np.convolve(losses, kernel, mode="valid")
    else:
        losses_s = losses

    x = np.arange(len(losses_s))
    plt.figure(figsize=(8, 5))
    plt.plot(x, losses_s, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Loss curve saved to {save_path}")
    plt.close()


def save_loss_files(loss_array, name_prefix, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{name_prefix}.npy"), np.array(loss_array))
    pd.DataFrame({
        "epoch": np.arange(len(loss_array)),
        "loss": loss_array,
    }).to_csv(os.path.join(save_dir, f"{name_prefix}.csv"), index=False)


def predict_and_save(model, reference, save_prefix_csv, save_prefix_img):
    model.eval()
    with torch.no_grad():
        input_tensor, _, _ = add_coordinate_channels(reference)
        pred = model(input_tensor)

    pred_corrected = pred.clone()
    pred_corrected[0, 0] *= -1

    save_all_results(
        pred_corrected,
        reference,
        save_prefix_csv,
        save_prefix_img,
        save_benchmark=True
    )


if __name__ == "__main__":
    reference = load_grayscale_image(reference_path).to(device)
    deformed = load_grayscale_image(deformed_path).to(device)

    reference = crop_center_to_multiple(reference, factor=8)
    deformed = crop_center_to_multiple(deformed, factor=8)

    model = DisplacementCNN(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Training start.")
    start_time = time.time()

    warmup_losses = train.train(
        model,
        reference,
        deformed,
        optimizer,
        warmup=True,
        num_epochs=10,
    )
    save_loss_files(warmup_losses, f"{loss_tag}_warmup_loss", output_dir)
    print("Warm-up over.")

    print("Main training start.")
    main_losses = train.train(
        model,
        reference,
        deformed,
        optimizer,
        warmup=False,
        num_epochs=1,
        threshold=0.0001,
        w_phy=1e-4,   # 1e-5 ~ 1e-2
        w_cmp=1e-5    # 1e-5 or 1e-4
    )
    save_loss_files(main_losses, f"{loss_tag}_main_loss", output_dir)

    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    print(f"\nTotal training time: {int(mins)} min {secs:.2f} sec")

    loss_img_path = os.path.join(output_dir, f"{loss_tag}_loss_curve.png")
    plot_loss_curve(
        main_losses,
        save_path=loss_img_path,
        smooth_window=1,
        title="Training Loss Curve (ZNCC + Physics)"
    )

    predict_and_save(
        model,
        reference,
        save_prefix_csv=output_csv_prefix,
        save_prefix_img=output_img_prefix
    )

    print("All results saved successfully.")
