from utils import add_coordinate_channels, warp_image_with_displacement, EarlyStopping
from loss import compute_main_loss, compute_warmup_loss
import torch


def train(
    model,
    reference,
    deformed,
    optimizer,
    num_epochs=100,
    warmup=False,
    threshold=None,
    roi_mask=None,
    w_phy=1e-4,     # Eq.(13) weight
    w_cmp=0.0,      # Eq.(3) compatibility weight (optional)
):
    """
    Train the displacement CNN model.

    reference, deformed: [1,1,H,W] typically (you use single pair training)
    roi_mask: [1,1,H,W] or None
    """
    early_stopper = EarlyStopping(min_delta=1e-6, patience=500)
    loss_log = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for _ in range(100):
            optimizer.zero_grad()

            # Input with coordinate channels (kept your design)
            input_tensor, _, _ = add_coordinate_channels(reference)
            pred = model(input_tensor)  # [B,5,H,W]

            # Warp reference using predicted u,v
            I_pred = warp_image_with_displacement(reference, pred)

            # Apply ROI mask to images (same as your original)
            if roi_mask is not None:
                I_pred = I_pred * roi_mask
                deformed_masked = deformed * roi_mask
            else:
                deformed_masked = deformed

            # Loss
            if warmup:
                loss = compute_warmup_loss(I_pred, deformed_masked)
            else:
                loss = compute_main_loss(
                    I_pred, deformed_masked,
                    pred,
                    roi_mask=roi_mask,
                    w_phy=w_phy,
                    w_cmp=w_cmp
                )

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_log.append(avg_loss)
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

        if threshold is not None and avg_loss < threshold:
            print(f"Loss {avg_loss:.6f} < threshold {threshold}, stopping.")
            break

        early_stopper(avg_loss)
        if early_stopper.early_stop:
            stage = "warmup" if warmup else "main"
            print(f"Early stopping triggered during {stage}.")
            break

    return loss_log
