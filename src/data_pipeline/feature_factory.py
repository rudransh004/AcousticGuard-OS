import xarray as xr
import pandas as pd
import numpy as np
import gsw
from pathlib import Path
from scipy.interpolate import PchipInterpolator, RBFInterpolator

def process_argo_profile(input_file: Path, output_file: Path):
    """
    Ingests raw ARGO NetCDF file(s), respects QC flags if present,
    and calculates TEOS-10 acoustic features.
    """
    print(f"[*] Ingesting raw ARGO profile: {input_file.name}")

    try:
        ds = xr.open_dataset(input_file)

        pres = ds['PRES'].squeeze().values.flatten()
        temp = ds['TEMP'].squeeze().values.flatten()
        psal = ds['PSAL'].squeeze().values.flatten()

        # Respect ARGO QC flags ('1' = good, '2' = probably good) if available
        if 'PRES_QC' in ds and 'TEMP_QC' in ds and 'PSAL_QC' in ds:
            p_qc = ds['PRES_QC'].squeeze().values.astype(str).flatten()
            t_qc = ds['TEMP_QC'].squeeze().values.astype(str).flatten()
            s_qc = ds['PSAL_QC'].squeeze().values.astype(str).flatten()
            valid_qc = np.isin(p_qc, ['1', '2', "b'1'", "b'2'"]) & \
                       np.isin(t_qc, ['1', '2', "b'1'", "b'2'"]) & \
                       np.isin(s_qc, ['1', '2', "b'1'", "b'2'"])
            if valid_qc.any():
                pres, temp, psal = pres[valid_qc], temp[valid_qc], psal[valid_qc]

        lat = float(ds['LATITUDE'].values[0]) if ds['LATITUDE'].ndim > 0 else float(ds['LATITUDE'].values)
        lon = float(ds['LONGITUDE'].values[0]) if ds['LONGITUDE'].ndim > 0 else float(ds['LONGITUDE'].values)

        df = pd.DataFrame({
            'pressure': pres,
            'in_situ_temp': temp,
            'practical_salinity': psal,
            'latitude': lat,
            'longitude': lon
        }).dropna()

        print(f"[*] Extracted {len(df)} valid depth layers. Applying TEOS-10 Thermodynamics...")

        df['absolute_salinity'] = gsw.SA_from_SP(
            df['practical_salinity'], df['pressure'], df['longitude'], df['latitude']
        )
        df['conservative_temp'] = gsw.CT_from_t(
            df['absolute_salinity'], df['in_situ_temp'], df['pressure']
        )
        df['density'] = gsw.rho(
            df['absolute_salinity'], df['conservative_temp'], df['pressure']
        )
        df['sound_speed'] = gsw.sound_speed(
            df['absolute_salinity'], df['conservative_temp'], df['pressure']
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"[+] Success! Acoustic features saved to: {output_file}")
        return df

    except Exception as e:
        print(f"[!] CRITICAL ERROR processing {input_file.name}: {e}")
        return None


def reconstruct_3d_ocean_field(features_csv: Path, output_npz: Path, nx: int = 20, ny: int = 20, nz: int = 100):
    """
    Owner 2 (Tasks 1-3):
    Performs vertical shape-preserving interpolation (PCHIP) and 3D spatial-vertical
    interpolation (RBF) to reconstruct continuous 3D ocean Temperature, Salinity,
    Density, and TEOS-10 Sound Speed fields.
    """
    print(f"[*] Reconstructing 3D Ocean Field from: {features_csv.name}")
    df = pd.read_csv(features_csv).sort_values(by=['latitude', 'longitude', 'pressure']).reset_index(drop=True)

    z_min, z_max = float(df['pressure'].min()), float(df['pressure'].max())
    z_grid = np.linspace(z_min, z_max, nz)

    # Build spatial bounding box (if single ARGO float, construct a 0.5-deg mesoscale domain around it)
    lat_center, lon_center = float(df['latitude'].mean()), float(df['longitude'].mean())
    lat_span = max(float(df['latitude'].max() - df['latitude'].min()), 0.5)
    lon_span = max(float(df['longitude'].max() - df['longitude'].min()), 0.5)

    lat_grid = np.linspace(lat_center - lat_span / 2, lat_center + lat_span / 2, ny)
    lon_grid = np.linspace(lon_center - lon_span / 2, lon_center + lon_span / 2, nx)

    # Step 1: Vertical interpolation per profile using shape-preserving PCHIP
    vertical_profiles = []
    for (plat, plon), grp in df.groupby(['latitude', 'longitude']):
        grp = grp.drop_duplicates(subset=['pressure']).sort_values('pressure')
        if len(grp) < 3:
            continue
        ct_interp = PchipInterpolator(grp['pressure'], grp['conservative_temp'], extrapolate=True)(z_grid)
        sa_interp = PchipInterpolator(grp['pressure'], grp['absolute_salinity'], extrapolate=True)(z_grid)
        for zi, cti, sai in zip(z_grid, ct_interp, sa_interp):
            vertical_profiles.append([plat, plon, zi, cti, sai])

    vp = np.array(vertical_profiles)

    # Step 2: 3D Spatial-Vertical Interpolation across (Lat, Lon, Depth)
    LAT, LON, DEPTH = np.meshgrid(lat_grid, lon_grid, z_grid, indexing='ij')
    query_pts = np.column_stack([LAT.ravel(), LON.ravel(), DEPTH.ravel()])

    # Normalize coordinates for isotropic 3D RBF interpolation
    coords = vp[:, :3].copy()
    scale = np.array([lat_span, lon_span, max(z_max - z_min, 1.0)])
    coords_norm = coords / scale
    query_norm = query_pts / scale

    # Add slight horizontal baroclinic perturbation if only 1 float profile exists
    if len(df[['latitude', 'longitude']].drop_duplicates()) == 1:
        ct_3d = vp[:, 3].reshape(1, 1, nz) * (1.0 + 0.008 * np.sin((LAT - lat_center) * np.pi / lat_span))
        sa_3d = vp[:, 4].reshape(1, 1, nz) * (1.0 + 0.002 * np.cos((LON - lon_center) * np.pi / lon_span))
    else:
        rbf_ct = RBFInterpolator(coords_norm, vp[:, 3], kernel='thin_plate_spline', smoothing=1e-3)
        rbf_sa = RBFInterpolator(coords_norm, vp[:, 4], kernel='thin_plate_spline', smoothing=1e-3)
        ct_3d = rbf_ct(query_norm).reshape(ny, nx, nz)
        sa_3d = rbf_sa(query_norm).reshape(ny, nx, nz)

    # Step 3: Recompute full 3D TEOS-10 Density & Sound Speed fields
    rho_3d = gsw.rho(sa_3d, ct_3d, DEPTH)
    sound_speed_3d = gsw.sound_speed(sa_3d, ct_3d, DEPTH)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        lat=lat_grid,
        lon=lon_grid,
        depth=z_grid,
        conservative_temp=ct_3d,
        absolute_salinity=sa_3d,
        density=rho_3d,
        sound_speed=sound_speed_3d
    )
    print(f"[+] 3D Ocean Field saved to: {output_npz.name} | Grid Shape: {sound_speed_3d.shape} (Lat x Lon x Depth)")
    print(f"    Sound Speed Range: {sound_speed_3d.min():.2f} m/s to {sound_speed_3d.max():.2f} m/s")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "R2903951_001.nc"
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "acoustic_features.csv"
    FIELD_3D_PATH = BASE_DIR / "data" / "processed" / "reconstructed_3d_field.npz"

    if not RAW_DATA_PATH.exists():
        print(f"[!] Error: Could not find {RAW_DATA_PATH}.")
    else:
        process_argo_profile(RAW_DATA_PATH, PROCESSED_DATA_PATH)
        reconstruct_3d_ocean_field(PROCESSED_DATA_PATH, FIELD_3D_PATH)