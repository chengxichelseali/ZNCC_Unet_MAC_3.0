from utils import add_coordinate_channels, warp_image_with_displacement, EarlyStopping
from loss import compute_main_loss, compute_warmup_loss


def train(
    model,
    reference,
    deformed,
    optimizer,
    num_epochs=100,
    warmup=False,
    threshold=None,
    w_phy=1e-4,
    w_cmp=1e-5,
):

    early_stopper = EarlyStopping(min_delta=1e-6, patience=500)
    loss_log = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for _ in range(100):
            optimizer.zero_grad()

            input_tensor, _, _ = add_coordinate_channels(reference)

            pred = model(input_tensor)

            I_pred = warp_image_with_displacement(reference, pred)

            deformed_masked = deformed

            if warmup:
                loss = compute_warmup_loss(I_pred, deformed_masked)
            else:
                loss = compute_main_loss(
                    I_pred,
                    deformed_masked,
                    pred,
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
