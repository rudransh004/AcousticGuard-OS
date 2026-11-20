from pathlib import Path

path = Path("src/dataset.py")
source = path.read_text(encoding="utf-8")

old = '''        # 3. Ensure consistent node ordering (strictly monotonic by depth / pressure)
        features_df = features_df.sort_values(by="pressure").reset_index(drop=True)
        num_nodes = len(features_df)
'''
new = '''        # 3. Match the profile-aware graph builder node ordering.
        if "profile_id" not in features_df.columns:
            raise ValueError("Missing required column: profile_id")
        if features_df["profile_id"].isna().any():
            raise ValueError("Missing profile_id values found.")

        features_df["profile_id"] = features_df["profile_id"].astype(str)
        features_df = features_df.sort_values(
            by=["profile_id", "pressure"], kind="mergesort"
        ).reset_index(drop=True)

        profile_ids = features_df["profile_id"].to_numpy()
        observation_times = (
            features_df["time"].astype(str).tolist()
            if "time" in features_df.columns else []
        )
        num_nodes = len(features_df)
'''
assert source.count(old) == 1, "Node ordering block not found exactly once."
source = source.replace(old, new)

old = '''        # Validate edge count: for 1D vertical lattice, expected edges = 2 * (num_nodes - 1)
        expected_edges = 2 * (num_nodes - 1)
        if len(edges_df) != expected_edges:
            raise ValueError(
                f"Incorrect edge count in {edges_file.name}: expected {expected_edges} directed edges for {num_nodes} nodes, found {len(edges_df)}"
            )
'''
new = '''        # Validate expected edges independently for each profile.
        expected_edges = sum(
            2 * (len(group) - 1)
            for _, group in features_df.groupby("profile_id", sort=False)
            if len(group) >= 2
        )
        if len(edges_df) != expected_edges:
            raise ValueError(
                f"Incorrect edge count in {edges_file.name}: "
                f"expected {expected_edges} directed edges across profiles, "
                f"found {len(edges_df)}"
            )

        # Ensure no edge crosses between separate ARGO profiles.
        if len(edges_df) and not np.all(
            profile_ids[src_indices] == profile_ids[dst_indices]
        ):
            raise ValueError("Cross-profile edge detected in edge CSV.")
'''
assert source.count(old) == 1, "Edge validation block not found exactly once."
source = source.replace(old, new)

old = '''        self.num_nodes = num_nodes
        self.num_edges = len(edges_df)
        self.target_col = self.TARGET_COL
'''
new = '''        self.profile_ids = profile_ids
        self.observation_times = observation_times
        self.num_profiles = int(features_df["profile_id"].nunique())

        self.num_nodes = num_nodes
        self.num_edges = len(edges_df)
        self.target_col = self.TARGET_COL
'''
assert source.count(old) == 1, "Metadata block not found exactly once."
source = source.replace(old, new)

old = '''            y=self.y,
            num_nodes=self.num_nodes,
'''
new = '''            y=self.y,
            profile_id=self.profile_ids.tolist(),
            observation_time=self.observation_times,
            num_nodes=self.num_nodes,
'''
assert source.count(old) == 1, "PyG Data block not found exactly once."
source = source.replace(old, new)

path.write_text(source, encoding="utf-8")
print("Dataset loader patch applied.")
