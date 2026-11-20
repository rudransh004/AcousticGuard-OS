from pathlib import Path

p = Path("src/train.py")
s = p.read_text(encoding="utf-8")

start = s.index("    # Align ground-truth target shape to [num_nodes, 1]")
end = s.index("\n    return {", start)

new_block = '''
    # Normalize inputs and target using training nodes only
    target_sound_speed = data.y
    if target_sound_speed.dim() == 1:
        target_sound_speed = target_sound_speed.unsqueeze(1)

    x_mean = data.x[train_mask].mean(dim=0, keepdim=True)
    x_std = data.x[train_mask].std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-8)

    y_mean = target_sound_speed[train_mask].mean(dim=0, keepdim=True)
    y_std = target_sound_speed[train_mask].std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-8)

    x_scaled = (data.x - x_mean) / x_std
    y_scaled = (target_sound_speed - y_mean) / y_std

    model.train()
    history = {
        "train_loss": [],
        "train_mse": [],
        "val_mse": [],
        "val_rmse": []
    }

    print("Beginning scaled training loop...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_scaled = model(x_scaled, data.edge_index, data.edge_attr)

        loss, train_mse_scaled, _ = criterion(
            pred=pred_scaled,
            target=y_scaled,
            edge_index=data.edge_index,
            mask=train_mask
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_scaled = model(
                x_scaled, data.edge_index, data.edge_attr
            )
            _, val_mse_scaled, _ = criterion(
                pred=val_pred_scaled,
                target=y_scaled,
                edge_index=data.edge_index,
                mask=val_mask
            )

            val_mse = val_mse_scaled * y_std.item() ** 2
            val_rmse = torch.sqrt(val_mse).item()
            train_rmse = torch.sqrt(train_mse_scaled).item() * y_std.item()

        history["train_loss"].append(loss.item())
        history["train_mse"].append(
            train_mse_scaled.item() * y_std.item() ** 2
        )
        history["val_mse"].append(val_mse.item())
        history["val_rmse"].append(val_rmse)

        if epoch % 20 == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"Epoch {epoch:03d} | Normalized Loss: {loss.item():.6f} | "
                f"Train RMSE: {train_rmse:.4f} m/s | "
                f"Validation RMSE: {val_rmse:.4f} m/s"
            )

    # Save model and normalization statistics together
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.detach().cpu(),
        "x_std": x_std.detach().cpu(),
        "y_mean": y_mean.detach().cpu(),
        "y_std": y_std.detach().cpu(),
        "feature_names": list(dataset.feature_cols),
        "target_name": dataset.target_col,
        "held_out_ratio": float(held_out_ratio),
        "seed": int(seed),
        "split_ranges": split_ranges,
        "num_nodes": int(num_nodes),
        "num_profiles": int(dataset.num_profiles),
    }

    print("\\\\nTraining completed. Verifying scaled checkpoint...")
    try:
        torch.save(checkpoint, str(temp_checkpoint_path))
        if not temp_checkpoint_path.exists() or temp_checkpoint_path.stat().st_size == 0:
            raise RuntimeError("Temporary checkpoint is missing or empty.")

        verified = torch.load(str(temp_checkpoint_path), map_location="cpu")
        required = {
            "format_version", "model_state_dict", "x_mean", "x_std",
            "y_mean", "y_std", "feature_names", "target_name",
            "held_out_ratio", "seed", "split_ranges", "num_nodes",
            "num_profiles"
        }
        if not isinstance(verified, dict) or not required.issubset(verified):
            raise RuntimeError("Checkpoint is missing required fields.")
        if verified["num_nodes"] != num_nodes or verified["num_profiles"] != dataset.num_profiles:
            raise RuntimeError("Checkpoint dataset metadata does not match.")
        if verified["feature_names"] != list(dataset.feature_cols):
            raise RuntimeError("Checkpoint feature names do not match.")
        if verified["target_name"] != dataset.target_col:
            raise RuntimeError("Checkpoint target name does not match.")
        for key in ("x_mean", "x_std", "y_mean", "y_std"):
            value = verified[key]
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise RuntimeError(f"Invalid scaling statistic: {key}")
        if verified["x_mean"].shape != (1, in_channels):
            raise RuntimeError("Input scaling shape is invalid.")
        if verified["y_mean"].shape != (1, 1):
            raise RuntimeError("Target scaling shape is invalid.")
        if torch.any(verified["x_std"] <= 0) or torch.any(verified["y_std"] <= 0):
            raise RuntimeError("Scaling standard deviations must be positive.")
        os.replace(temp_checkpoint_path, checkpoint_path)
        print(f"Verified scaled checkpoint saved to {checkpoint_path}")
    except Exception as e:
        if temp_checkpoint_path.exists():
            temp_checkpoint_path.unlink()
        raise RuntimeError(f"Checkpoint save aborted: {e}") from e
'''

s = s[:start] + new_block + s[end:]

old_return = '''        "split_ranges": split_ranges,
        "history": history
'''
new_return = '''        "split_ranges": split_ranges,
        "x_mean": x_mean.detach().cpu(),
        "x_std": x_std.detach().cpu(),
        "y_mean": y_mean.detach().cpu(),
        "y_std": y_std.detach().cpu(),
        "history": history
'''
if old_return not in s:
    raise RuntimeError("Expected return metadata block not found.")
s = s.replace(old_return, new_return, 1)

p.write_text(s, encoding="utf-8")
print("Scaled profile-aware training patch applied.")
