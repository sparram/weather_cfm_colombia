import cdsapi

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels"

meses = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12"
]

for mes in meses:
    print(f"📥 Descargando ERA5 (Superficie) para el mes 2024-{mes}...")
    
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "total_column_water_vapour",
            "convective_available_potential_energy"
        ],
        "year": ["2024"],
        "month": [mes],
        "day": [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
        ],
        "time": [
            "00:00", "06:00", "12:00", "18:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [12.75, -82, -4.5, -66.75] # [North, West, South, East]
    }
    
    output_file = f"era5_single_2024_{mes}.nc"
    
    try:
        client.retrieve(dataset, request).download(output_file)
        print(f"✅ ¡Completado y guardado: {output_file}!")
    except Exception as e:
        print(f"❌ Error al descargar el mes {mes}: {e}")

print("\n🎉 ¡Descargas de ERA5 (Superficie) finalizadas con éxito!")