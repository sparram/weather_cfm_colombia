import xarray as xr
import os

def unificar_era5_mes(mes_str):
    pres_file = f"raw/era5_2024_raw/era5_pressure_2024_{mes_str}.nc"
    single_file = f"raw/era5_2024_raw/era5_single_2024_{mes_str}.nc"
    output_file = f"processed/era5_2024/era5_complete_2024_{mes_str}.nc"
    
    if not os.path.exists(pres_file) or not os.path.exists(single_file):
        print(f"⚠️ Faltan archivos para el mes {mes_str}. Saltando...")
        return
    
    print(f"📦 Unificando ERA5 para el mes 2024-{mes_str}...")
    
    try:
        # Abrir ambos datasets
        ds_pres = xr.open_dataset(pres_file)
        ds_sing = xr.open_dataset(single_file)
        
        # Fusionar explícitamente sin conflictos
        ds_unificado = xr.merge([ds_pres, ds_sing], compat="no_conflicts")
        
        # En ERA5 las coordenadas se llaman 'latitude' y 'longitude'
        if 'latitude' in ds_unificado.coords:
            # Asegurarnos de que las latitudes vayan de mayor a menor (Norte a Sur)
            if ds_unificado.latitude[0] < ds_unificado.latitude[-1]:
                ds_unificado = ds_unificado.sortby('latitude', ascending=False)
            
        # Guardar el resultado limpio
        ds_unificado.to_netcdf(output_file)
        print(f"✅ ¡Guardado con éxito: {output_file} (Dimensiones: {ds_unificado.dims})!")
        
        # Cerrar datasets para liberar memoria
        ds_pres.close()
        ds_sing.close()
        ds_unificado.close()
        
    except Exception as ex:
        print(f"❌ Error procesando el mes {mes_str}: {ex}")

if __name__ == "__main__":
    meses = [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ]
    
    for mes in meses:
        unificar_era5_mes(mes)
        
    print("\n🎉 ¡Proceso de unificación de ERA5 completado!")