import xarray as xr

# Abrir el archivo unificado de enero
ds = xr.open_dataset("processed/era5_2024/era5_complete_2024_01.nc")

print("📊 RESUMEN DEL DATASET UNIFICADO:")
print(ds)

print("\n----------------------------------------")
print("🔎 CHEQUEO DE VALORES NULOS (NaNs):")
for var in ds.data_vars:
    nans = ds[var].isnull().sum().values
    print(f" - {var}: {nans} valores nulos encontrados")

print("----------------------------------------")
print("⏰ RANGO DE TIEMPO:")
print(f"Desde: {ds.valid_time.min().values} hasta {ds.valid_time.max().values}")