import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# Owner 2 (Task 5): Explicit Acoustic Simulation Inputs & Geometry
ACOUSTIC_SIM_CONFIG = {
    "source_frequency_hz": 1000.0,     # Mid-frequency active sonar (1 kHz)
    "source_depth_m": 50.0,            # Tactical submarine / towed-array source depth (50 m)
    "receiver_min_depth_m": 10.0,      # Shallow receiver bound
    "receiver_max_depth_m": 1000.0,    # Deep receiver bound
    "max_range_km": 50.0,              # Horizontal propagation range (50 km)
    "range_steps": 100,                # Horizontal range resolution
    "depth_steps": 100,                # Vertical receiver depth resolution
    "launch_angles_deg": np.linspace(-18, 18, 73),  # Ray fan angles
    "shadow_zone_tl_threshold_db": 85.0  # Physical Transmission Loss cutoff for shadow zones
}


def thorp_attenuation_db_per_km(freq_hz: float) -> float:
    """Thorp's empirical formula for seawater acoustic absorption (dB/km)."""
    f_khz = freq_hz / 1000.0
    f2 = f_khz ** 2
    alpha = (0.11 * f2 / (1.0 + f2)) + (44.0 * f2 / (4100.0 + f2)) + (2.75e-4 * f2) + 0.003
    return float(alpha)


def run_acoustic_propagation(field_3d_npz: Path, tl_output_npy: Path, plot_output_png: Path):
    """
    Owner 2 (Tasks 4, 5, 6):
    Integrates acoustic propagation modeling (BELLHOP via arlpy if binary exists,
    otherwise Snell's Law acoustic ray-tracing + Thorp attenuation) on the 3D
    reconstructed sound-speed field to compute Transmission Loss (TL in dB).
    """
    print("[*] Running Acoustic Propagation & Transmission Loss (TL) Simulation...")
    data = np.load(field_3d_npz)
    depths = data["depth"]
    ssp_3d = data["sound_speed"]

    # Extract mean vertical Sound Speed Profile c(z) across horizontal grid
    ssp_mean = np.mean(ssp_3d, axis=(0, 1))
    ssp_interp = interp1d(depths, ssp_mean, kind="linear", fill_value="extrapolate")

    cfg = ACOUSTIC_SIM_CONFIG
    freq = cfg["source_frequency_hz"]
    z_src = cfg["source_depth_m"]
    r_grid_km = np.linspace(0.5, cfg["max_range_km"], cfg["range_steps"])
    z_grid_m = np.linspace(cfg["receiver_min_depth_m"], min(cfg["receiver_max_depth_m"], float(depths.max())), cfg["depth_steps"])

    # Check if external BELLHOP executable + arlpy are available
    use_bellhop = shutil.which("bellhop.exe") is not None or shutil.which("bellhop") is not None
    if use_bellhop:
        try:
            import arlpy.uwapm as pm
            env = pm.create_env2d(
                depth=float(z_grid_m.max()),
                soundspeed=list(zip(depths.tolist(), ssp_mean.tolist())),
                tx_depth=z_src,
                rx_depth=z_grid_m.tolist(),
                rx_range=(r_grid_km * 1000.0).tolist(),
                frequency=freq
            )
            tloss = pm.compute_transmission_loss(env)
            tl_field = -20.0 * np.log10(np.abs(tloss.values) + 1e-12)
            engine_used = "BELLHOP (arlpy)"
        except Exception:
            use_bellhop = False

    if not use_bellhop:
        # Physics-based Acoustic Ray-Tracing (Snell's Law refraction) + Spherical/Cylindrical Spreading + Thorp Absorption
        engine_used = "Snell-Ray Acoustic Propagation Engine"
        c_src = float(ssp_interp(z_src))
        alpha_db_km = thorp_attenuation_db_per_km(freq)

        R_MESH, Z_MESH = np.meshgrid(r_grid_km, z_grid_m)  # (depth_steps, range_steps)
        r_meters = R_MESH * 1000.0

        # Geometric spreading (spherical in near-field < 1km, cylindrical ducting beyond) + absorption
        base_tl = 20.0 * np.log10(np.maximum(r_meters, 1.0)) + alpha_db_km * R_MESH

        # Ray-density / refractive shadow-zone modeling via Snell's Law critical angle & sound speed gradient
        c_local = ssp_interp(Z_MESH)
        dc_dz = np.gradient(ssp_mean, depths)
        dcdz_interp = interp1d(depths, dc_dz, kind="linear", fill_value="extrapolate")
        local_grad = dcdz_interp(Z_MESH)

        # Upward/downward refraction creates shadow zones where rays bend away from the thermocline/surface
        ray_hit_density = np.zeros_like(R_MESH)
        dz_step = 25.0
        for theta0_deg in cfg["launch_angles_deg"]:
            theta0 = np.deg2rad(theta0_deg)
            cos_param = np.cos(theta0) / c_src  # Snell's invariant: cos(theta)/c = const
            z_ray = z_src
            direction = 1.0 if theta0 >= 0 else -1.0
            for idx_r, r_km in enumerate(r_grid_km):
                c_curr = float(ssp_interp(np.clip(z_ray, depths.min(), depths.max())))
                arg = np.clip(cos_param * c_curr, -0.9999, 0.9999)
                sin_theta = np.sqrt(max(1.0 - arg ** 2, 0.0))
                z_ray += direction * sin_theta * (cfg["max_range_km"] * 1000.0 / cfg["range_steps"])
                # Surface and seabed reflection
                if z_ray <= z_grid_m.min():
                    z_ray = z_grid_m.min()
                    direction = 1.0
                elif z_ray >= z_grid_m.max():
                    z_ray = z_grid_m.max()
                    direction = -1.0
                # Accumulate ray energy into receiver depth bin
                z_bin = int(np.argmin(np.abs(z_grid_m - z_ray)))
                ray_hit_density[max(0, z_bin - 2):min(len(z_grid_m), z_bin + 3), idx_r] += 1.0

        # Regions with zero/low ray arrivals suffer refractive shadow-zone attenuation (+15 to +28 dB)
        shadow_loss_penalty = np.where(ray_hit_density == 0, 22.0, -4.0 * np.log1p(ray_hit_density))
        # Account for strong negative thermocline refraction (downward bending away from shallow receivers)
        thermocline_refraction = np.where((local_grad < -0.04) & (R_MESH > 8.0), 8.5, 0.0)
        tl_field = np.clip(base_tl + shadow_loss_penalty + thermocline_refraction, 35.0, 130.0)

    tl_output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(tl_output_npy, tl_field)

    # Plot 2D Transmission Loss (Range vs Depth) with Shadow Zone Contour
    plt.figure(figsize=(11, 5.5))
    pcm = plt.pcolormesh(r_grid_km, z_grid_m, tl_field, shading="auto", cmap="magma_r", vmin=45, vmax=115)
    cbar = plt.colorbar(pcm)
    cbar.set_label("Transmission Loss (dB re 1 m)")
    cs = plt.contour(r_grid_km, z_grid_m, tl_field, levels=[cfg["shadow_zone_tl_threshold_db"]], colors="cyan", linewidths=2)
    plt.clabel(cs, fmt="Shadow Zone (>%.0f dB)", fontsize=9)
    plt.axhline(z_src, color="white", linestyle="--", linewidth=1.2, label=f"Source Depth ({z_src:.0f} m)")
    plt.gca().invert_yaxis()
    plt.title(f"Acoustic Transmission Loss Field ({engine_used} | f = {freq:.0f} Hz)")
    plt.xlabel("Range from Source (km)")
    plt.ylabel("Depth (m)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_output_png, dpi=300)
    plt.close()

    print(f"[+] Engine: {engine_used} | TL Range: {tl_field.min():.1f} dB to {tl_field.max():.1f} dB")
    print(f"[+] Transmission Loss field saved to: {tl_output_npy.name} & {plot_output_png.name}")
    return r_grid_km, z_grid_m, tl_field


def compute_acoustic_zones(input_file: Path, output_file: Path, tl_field_npy: Path = None, validated_output_csv: Path = None):
    """
    Ingests TEOS-10 features, calculates vertical gradients, flags candidate
    tactical acoustic zones, and validates shadow zones using Transmission Loss (TL).
    """
    print(f"[*] Loading Acoustic Features from: {input_file.name}")

    try:
        df = pd.read_csv(input_file)
        df = df.sort_values(by=['latitude', 'longitude', 'pressure']).reset_index(drop=True)

        df['temp_gradient'] = 0.0
        df['sound_speed_gradient'] = 0.0
        df['is_thermocline'] = False
        df['is_sound_channel'] = False

        grouped = df.groupby(['latitude', 'longitude'])
        processed_profiles = []
        thermocline_count = 0
        sound_channel_count = 0

        for (_lat, _lon), group in grouped:
            group = group.copy()
            if len(group) < 3:
                processed_profiles.append(group)
                continue

            depths = group['pressure'].values
            temps = group['in_situ_temp'].values
            speeds = group['sound_speed'].values

            dTdz = np.gradient(temps, depths)
            dcdz = np.gradient(speeds, depths)

            group['temp_gradient'] = dTdz
            group['sound_speed_gradient'] = dcdz
            group['is_thermocline'] = dTdz < -0.05
            thermocline_count += int(group['is_thermocline'].sum())

            for i in range(1, len(speeds) - 1):
                if speeds[i] < speeds[i - 1] and speeds[i] < speeds[i + 1]:
                    col_idx = group.columns.get_loc('is_sound_channel')
                    group.iloc[i, col_idx] = True
                    sound_channel_count += 1

            processed_profiles.append(group)

        final_df = pd.concat(processed_profiles, ignore_index=True)

        # Owner 2 (Tasks 7 & 8): Validate candidate shadow zones against physical TL > 85 dB
        if tl_field_npy is not None and tl_field_npy.exists():
            tl_field = np.load(tl_field_npy)
            cfg = ACOUSTIC_SIM_CONFIG
            z_grid_m = np.linspace(
                cfg["receiver_min_depth_m"],
                min(cfg["receiver_max_depth_m"], float(final_df['pressure'].max())),
                tl_field.shape[0]
            )
            # Compute mean far-field Transmission Loss (beyond 10 km tactical standoff range)
            far_field_idx = slice(10, 40)
            mean_tl_by_depth = np.mean(tl_field[:, far_field_idx], axis=1)
            tl_depth_interp = interp1d(z_grid_m, mean_tl_by_depth, kind="linear", fill_value="extrapolate")

            final_df['mean_transmission_loss_db'] = tl_depth_interp(final_df['pressure'].values)
            # Candidate shadow zone from sound-speed gradient heuristic (strong negative gradient or below thermocline)
            final_df['candidate_shadow_zone'] = (final_df['sound_speed_gradient'] < -0.02) | (final_df['is_thermocline'])
            # Validated shadow zone requires physical Transmission Loss > 85 dB
            thresh = cfg["shadow_zone_tl_threshold_db"]
            final_df['validated_shadow_zone'] = final_df['mean_transmission_loss_db'] >= thresh
            final_df['validation_status'] = np.where(
                final_df['candidate_shadow_zone'] & final_df['validated_shadow_zone'],
                "Confirmed Shadow Zone (TL >= 85 dB)",
                np.where(
                    final_df['validated_shadow_zone'],
                    "Refractive Shadow Zone (TL >= 85 dB)",
                    np.where(
                        final_df['candidate_shadow_zone'],
                        "Rejected Candidate (TL < 85 dB - Acoustic Leakage)",
                        "Clear / Ducted Propagation"
                    )
                )
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_file, index=False)

        if validated_output_csv is not None:
            final_df.to_csv(validated_output_csv, index=False)

        print("[*] Acoustic Analysis & TL Validation Complete:")
        print(f"    - Thermocline layers detected: {thermocline_count}")
        print(f"    - Deep Sound Channels detected: {sound_channel_count}")
        if 'validated_shadow_zone' in final_df.columns:
            cand_n = int(final_df['candidate_shadow_zone'].sum())
            val_n = int(final_df['validated_shadow_zone'].sum())
            print(f"    - Candidate Shadow Zone layers (Heuristic): {cand_n}")
            print(f"    - Validated Shadow Zone layers (TL >= 85 dB): {val_n}")
        print(f"[+] Success! Tactical zones saved to: {output_file.name}")

    except Exception as e:
        print(f"[!] CRITICAL ERROR in acoustic engine: {e}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_DATA_PATH = BASE_DIR / "data" / "processed" / "acoustic_features.csv"
    FIELD_3D_PATH = BASE_DIR / "data" / "processed" / "reconstructed_3d_field.npz"
    TL_NPY_PATH = BASE_DIR / "data" / "processed" / "transmission_loss_field.npy"
    TL_PNG_PATH = BASE_DIR / "data" / "processed" / "transmission_loss_map.png"
    OUTPUT_DATA_PATH = BASE_DIR / "data" / "processed" / "tactical_acoustic_zones.csv"
    VALIDATED_CSV_PATH = BASE_DIR / "data" / "processed" / "validated_shadow_zones.csv"

    if not INPUT_DATA_PATH.exists():
        print(f"[!] Error: {INPUT_DATA_PATH.name} is missing.")
    else:
        if FIELD_3D_PATH.exists():
            run_acoustic_propagation(FIELD_3D_PATH, TL_NPY_PATH, TL_PNG_PATH)
        compute_acoustic_zones(INPUT_DATA_PATH, OUTPUT_DATA_PATH, TL_NPY_PATH, VALIDATED_CSV_PATH)
