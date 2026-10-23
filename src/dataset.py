import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset

class ColombiaWeatherDataset(Dataset):
    def __init__(self, data_dir, meses=["01"]):
        """
        Dataset de PyTorch para conectar ERA5 (inputs) con IMERG (targets) en Colombia.
        
        Args:
            data_dir (str): Ruta a la carpeta que contiene los archivos NetCDF procesados.
            meses (list): Lista de meses a cargar (ej. ["01", "02", ...]).
        """
        self.data_dir = data_dir
        self.meses = meses
        
        self.era5_ds = None
        self.imerg_ds = None
        self.dem_np = None
        self.times = []
        
        self._cargar_datos()
        self._cargar_dem()

    def _cargar_datos(self):
        era5_list = []
        imerg_list = []
        
        for mes in self.meses:
            era5_path = os.path.join(self.data_dir, f"era5_2024/era5_complete_2024_{mes}.nc")
            imerg_path = os.path.join(self.data_dir, f"imerg_2024/imerg_complete_2024_{mes}.nc")

            print(era5_path)
            print(imerg_path)
        
            if os.path.exists(era5_path) and os.path.exists(imerg_path):
                era5_list.append(xr.open_dataset(era5_path))
                imerg_list.append(xr.open_dataset(imerg_path))
            else:
                print(f"⚠️ Advertencia: Faltan archivos (ERA5 o IMERG) para el mes 2024-{mes}. Saltando...")

        if era5_list and imerg_list:
            # Concatenar todos los meses y eliminar duplicados de tiempo inmediatamente
            self.era5_ds = xr.concat(era5_list, dim="valid_time").drop_duplicates(dim='valid_time', keep='first')
            
            # Concatenar IMERG
            self.imerg_ds = xr.concat(imerg_list, dim="time").drop_duplicates(dim='time', keep='first')
            
            # Convertir el calendario cftime de IMERG a fechas estándar de Pandas
            import pandas as pd
            times_str = [str(t) for t in self.imerg_ds.time.values]
            self.imerg_ds['time'] = pd.to_datetime(times_str)
            
            # Usamos las marcas de tiempo de ERA5 como referencia temporal principal
            self.times = self.era5_ds.valid_time.values
            print(f"✅ Dataset cargado con éxito. Total de muestras temporales: {len(self.times)}")
        else:
            raise ValueError("❌ No se pudieron cargar los datasets completos. Revisa las rutas y nombres de archivos.")

    def _cargar_dem(self):
        dem_path = os.path.join(self.data_dir, "dem_colombia_era5.nc")
        if os.path.exists(dem_path):
            dem_ds = xr.open_dataset(dem_path)
            self.dem_np = dem_ds['elevation'].values  # (Lat, Lon)
            dem_ds.close()
            print("✅ DEM de superficie de Colombia cargado y listo como canal estático.")
        else:
            raise FileNotFoundError(f"❌ No se encontró el archivo DEM en: {dem_path}. Ejecuta extract_dem.py primero.")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        t = self.times[idx]
        
        # 1. Extraer corte temporal de ERA5 (Entrada X)
        era5_slice = self.era5_ds.sel(valid_time=t)
        
        channels = []
        vars_3d = ['z', 'q', 't', 'u', 'v', 'w']
        levels = [850, 500, 250]
        
        for var in vars_3d:
            for lvl in levels:
                channels.append(era5_slice[var].sel(pressure_level=lvl).values)
                
        vars_2d = ['u10', 'v10', 't2m', 'tcwv', 'cape']
        for var in vars_2d:
            channels.append(era5_slice[var].values)
            
        # Añadir el DEM estático de Colombia como el canal 24
        channels.append(self.dem_np)
            
        x_np = np.stack(channels, axis=0)
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        
        # 2. Extraer corte temporal de IMERG (Objetivo Y - Alta resolución) con method='nearest'
        imerg_slice = self.imerg_ds.sel(time=t, method='nearest')
        y_np = imerg_slice['precipitation'].values 
        
        y_np = np.nan_to_num(y_np, nan=0.0)
        
        if len(y_np.shape) == 2:
            y_np = np.expand_dims(y_np, axis=0)
            
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        
        return x_tensor, y_tensor


# ==========================================
# BLOQUE DE PRUEBA Y VALIDACIÓN
# ==========================================
if __name__ == "__main__":
    print("🧪 Probando el dataset de PyTorch con orografía integrada...")
    
    data_path = "data/processed/" 
    
    try:
        dataset = ColombiaWeatherDataset(data_dir=data_path, meses=["01"])
        
        print(f"📏 Longitud del dataset (muestras temporales en enero): {len(dataset)}")
        
        if len(dataset) > 0:
            x, y = dataset[0]
            
            print("✅ ¡Primera muestra extraída con éxito!")
            print(f"   - Forma del tensor de entrada (ERA5 + DEM - X): {x.shape} --> Esperado: (24, 70, 62)")
            print(f"   - Forma del tensor objetivo (IMERG - Y): {y.shape} --> Esperado: (1, 153, 173)")
            print(f"   - Rango de valores X: min={x.min():.2f}, max={x.max():.2f}")
            print(f"   - Rango de valores Y (Precipitación): min={y.min():.2f}, max={y.max():.2f}")
            
    except Exception as e:
        print(f"❌ Error durante la prueba del dataset: {e}")