import cdsapi

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels"

print("📥 Descargando la orografía de superficie (Geopotencial estático) de ERA5...")

request = {
    "product_type": ["reanalysis"],
    "variable": ["geopotential"],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [12.75, -82, -4.5, -66.75]  # [North, West, South, East]
}

output_file = "data/processed/surface_geopotential_era5.nc"

try:
    client.retrieve(dataset, request).download(output_file)
    print(f"✅ ¡Orografía de superficie descargada con éxito en: {output_file}!")
except Exception as e:
    print(f"❌ Error al descargar: {e}")