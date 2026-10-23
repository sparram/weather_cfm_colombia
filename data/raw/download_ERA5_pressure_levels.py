import cdsapi

client = cdsapi.Client()
dataset = "reanalysis-era5-pressure-levels"

meses = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12"
]

for mes in meses:
    print(f"📥 Descargando ERA5 (Niveles de Presión) para el mes 2024-{mes}...")
    
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity"
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
        "pressure_level": ["250", "500", "850"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [12.75, -82, -4.5, -66.75] # [North, West, South, East]
    }
    
    output_file = f"era5_pressure_2024_{mes}.nc"
    
    try:
        client.retrieve(dataset, request).download(output_file)
        print(f"✅ ¡Completado y guardado: {output_file}!")
    except Exception as e:
        print(f"❌ Error al descargar el mes {mes}: {e}")

print("\n🎉 ¡Descargas de ERA5 (Presión) finalizadas con éxito!")