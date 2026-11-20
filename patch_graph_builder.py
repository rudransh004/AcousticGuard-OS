from pathlib import Path

path = Path("src/ml_core/graph_builder.py")
source = path.read_text(encoding="utf-8")

start = source.index("def build_vertical_edges(")
end = source.index("def build_spatial_edges(", start)

replacement = '''def build_vertical_edges(
    input_file: Union[Path, str],
    output_file: Union[Path, str],
) -> pd.DataFrame:
    """
    Build independent bidirectional vertical graphs for each ARGO profile.
    Node indices follow profile_id and pressure ordering in the sorted CSV.
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input acoustic features file not found: {input_path}")

    logger.info("Loading acoustic features from: %s", input_path.name)
    df = pd.read_csv(input_path)

    required_cols = ["profile_id", "depth", "pressure"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input features file contains no observations.")

    if df["profile_id"].isna().any():
        raise ValueError("Missing profile_id values found.")

    df["profile_id"] = df["profile_id"].astype(str)
    if (df["profile_id"].str.strip() == "").any():
        raise ValueError("Empty profile_id values found.")

    for col in ["depth", "pressure"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        values = df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values found in '{col}'.")

    df = df.sort_values(
        ["profile_id", "pressure"], kind="mergesort"
    ).reset_index(drop=True)

    edges: List[dict] = []
    total_profiles = 0

    for profile_id, group in df.groupby("profile_id", sort=False):
        node_indices = group.index.to_numpy(dtype=np.int64)
        depths = group["depth"].to_numpy(dtype=np.float64)
        pressures = group["pressure"].to_numpy(dtype=np.float64)

        if len(group) < 2:
            logger.warning("Profile %s has fewer than 2 observations; no edges created.", profile_id)
            continue

        if np.any(np.diff(pressures) <= 0) or np.any(np.diff(depths) <= 0):
            raise ValueError(
                f"Non-increasing or duplicate pressure/depth levels in profile {profile_id}."
            )

        total_profiles += 1

        for i in range(len(group) - 1):
            source_node = int(node_indices[i])
            target_node = int(node_indices[i + 1])
            dz = float(depths[i + 1] - depths[i])

            if not np.isfinite(dz) or dz <= 0:
                raise ValueError(f"Invalid depth difference in profile {profile_id}: {dz}")

            for src, dst in [
                (source_node, target_node),
                (target_node, source_node),
            ]:
                edges.append({
                    "source_node": src,
                    "target_node": dst,
                    "depth_diff_m": round(dz, 4),
                })

    edges_df = pd.DataFrame(
        edges, columns=["source_node", "target_node", "depth_diff_m"]
    )

    expected_edge_count = sum(
        2 * (len(group) - 1)
        for _, group in df.groupby("profile_id", sort=False)
        if len(group) >= 2
    )

    if len(edges_df) != expected_edge_count:
        raise ValueError(
            f"Edge count mismatch: expected {expected_edge_count}, got {len(edges_df)}"
        )

    if len(edges_df):
        num_nodes = len(df)
        src = edges_df["source_node"].to_numpy()
        dst = edges_df["target_node"].to_numpy()
        attrs = edges_df["depth_diff_m"].to_numpy(dtype=np.float64)

        if (
            np.any(src < 0) or np.any(src >= num_nodes)
            or np.any(dst < 0) or np.any(dst >= num_nodes)
        ):
            raise ValueError("Edge indices out of bounds.")

        if not np.isfinite(attrs).all() or np.any(attrs <= 0):
            raise ValueError("Invalid edge depth attributes.")

        profile_ids = df["profile_id"].to_numpy()
        if not np.all(profile_ids[src] == profile_ids[dst]):
            raise ValueError("Cross-profile edge detected.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(output_path, index=False)

    logger.info(
        "Graph generation complete: %d profiles, %d nodes, %d directed edges saved to %s",
        total_profiles, len(df), len(edges_df), output_path
    )
    return edges_df


'''

path.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
print("Patch applied to build_vertical_edges().")

