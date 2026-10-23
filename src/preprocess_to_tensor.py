import os
import torch
import xarray as xr
import numpy as np
import pandas as pd

def procesar_y_guardar_dataset():
    data_dir = "data/processed/"
    output_dir = "data/tensors/"
    os.makedirs(output_dir, exist_ok=True)
    
    meses = [f"{m:02d}" for m in range(1, 13)]
    
    print("🚀 Iniciando pre-procesamiento mes a mes...")
    
    # Cargar DEM una sola vez
    dem_path = os.path.join(data_dir, "dem_colombia_era5.nc")
    if os.path.exists(dem_path):
        dem_ds = xr.open_dataset(dem_path)
        dem_np = dem_ds['elevation'].values
        dem_ds.close()
    else:
        raise FileNotFoundError(f"❌ No se encontró el DEM en: {dem_path}")

    for mes in meses:
        era5_path = os.path.join(data_dir, f"era5_2024/era5_complete_2024_{mes}.nc")
        imerg_path = os.path.join(data_dir, f"imerg_2024/imerg_complete_2024_{mes}.nc")
        
        if not (os.path.exists(era5_path) and os.path.exists(imerg_path)):
            continue
            
        print(f"📦 Procesando y guardando mes {mes}...")
        era5_ds = xr.open_dataset(era5_path).load()
        imerg_ds = xr.open_dataset(imerg_path).load()
        
        # Normalizar fechas y eliminar duplicados
        era5_ds['valid_time'] = pd.to_datetime([str(t) for t in era5_ds.valid_time.values])
        era5_ds = era5_ds.drop_duplicates(dim='valid_time', keep='first')
        
        imerg_ds['time'] = pd.to_datetime([str(t) for t in imerg_ds.time.values])
        imerg_ds = imerg_ds.drop_duplicates(dim='time', keep='first')
        
        times_imerg = imerg_ds.time.values
        x_list, y_list = [], []
        
        for t in times_imerg:
            era5_slice = era5_ds.sel(valid_time=t, method='nearest')
            
            channels = []
            vars_3d = ['z', 'q', 't', 'u', 'v', 'w']
            levels = [850, 500, 250]
            
            for var in vars_3d:
                for lvl in levels:
                    channels.append(era5_slice[var].sel(pressure_level=lvl).values)
                    
            vars_2d = ['u10', 'v10', 't2m', 'tcwv', 'cape']
            for var in vars_2d:
                channels.append(era5_slice[var].values)
                
            channels.append(dem_np)
            
            # Codificación temporal cíclica (4 canales)
            t_dt = pd.to_datetime(t)
            hora_decimal = t_dt.hour + t_dt.minute / 60.0
            doy = t_dt.dayofyear
            
            h_sin = np.sin(2 * np.pi * hora_decimal / 24.0)
            h_cos = np.cos(2 * np.pi * hora_decimal / 24.0)
            d_sin = np.sin(2 * np.pi * doy / 365.25)
            d_cos = np.cos(2 * np.pi * doy / 365.25)
            
            H, W = dem_np.shape
            channels.append(np.full((H, W), h_sin, dtype=np.float32))
            channels.append(np.full((H, W), h_cos, dtype=np.float32))
            channels.append(np.full((H, W), d_sin, dtype=np.float32))
            channels.append(np.full((H, W), d_cos, dtype=np.float32))
            
            x_np = np.stack(channels, axis=0)
            x_list.append(torch.tensor(x_np, dtype=torch.float32))
            
            imerg_slice = imerg_ds.sel(time=t, method='nearest')
            y_np = imerg_slice['precipitation'].values
            y_np = np.nan_to_num(y_np, nan=0.0)
            if len(y_np.shape) == 2:
                y_np = np.expand_dims(y_np, axis=0)
            y_list.append(torch.tensor(y_np, dtype=torch.float32))
            
        era5_ds.close()
        imerg_ds.close()

        # Guardar archivo .pt individual por mes
        X_tensor = torch.stack(x_list, dim=0)
        Y_tensor = torch.stack(y_list, dim=0)
        
        file_path = os.path.join(output_dir, f"mes_{mes}.pt")
        torch.save({'X': X_tensor, 'Y': Y_tensor}, file_path)
        print(f"   ✅ Guardado: {file_path} ({len(X_tensor)} muestras)")

    print("\n🎉 ¡Pre-procesamiento mes a mes completado sin picos de RAM!")

if __name__ == "__main__":
    procesar_y_guardar_dataset()