import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset

class ColombiaWeatherDataset(Dataset):
    def __init__(self, data_dir, meses=["01"], normalize=True):
        self.data_dir = data_dir
        self.meses = meses
        self.normalize = normalize
        
        self.era5_ds = None
        self.imerg_ds = None
        self.dem_np = None
        self.times = []
        
        self._cargar_datos()
        self._cargar_dem()
        
        # Rutas para guardar/cargar estadísticas de normalización
        self.stats_path = os.path.join("models_checkpoints", "normalization_stats.pt")
        
        if self.normalize:
            self._pre_calcular_o_cargar_stats()

    def _cargar_datos(self):
        era5_list = []
        imerg_list = []
        
        for mes in self.meses:
            era5_path = os.path.join(self.data_dir, f"era5_2024/era5_complete_2024_{mes}.nc")
            imerg_path = os.path.join(self.data_dir, f"imerg_2024/imerg_complete_2024_{mes}.nc")
        
            if os.path.exists(era5_path) and os.path.exists(imerg_path):
                era5_list.append(xr.open_dataset(era5_path))
                imerg_list.append(xr.open_dataset(imerg_path))

        if era5_list and imerg_list:
            self.era5_ds = xr.concat(era5_list, dim="valid_time").drop_duplicates(dim='valid_time', keep='first')
            self.imerg_ds = xr.concat(imerg_list, dim="time").drop_duplicates(dim='time', keep='first')
            
            import pandas as pd
            times_str = [str(t) for t in self.imerg_ds.time.values]
            self.imerg_ds['time'] = pd.to_datetime(times_str)
            
            self.times = self.era5_ds.valid_time.values
        else:
            raise ValueError("❌ No se pudieron cargar los datasets completos.")

    def _cargar_dem(self):
        dem_path = os.path.join(self.data_dir, "dem_colombia_era5.nc")
        if os.path.exists(dem_path):
            dem_ds = xr.open_dataset(dem_path)
            self.dem_np = dem_ds['elevation'].values
            dem_ds.close()
        else:
            raise FileNotFoundError(f"❌ No se encontró el archivo DEM en: {dem_path}")

    def _pre_calcular_o_cargar_stats(self):
        os.makedirs("models_checkpoints", exist_ok=True)
        
        if os.path.exists(self.stats_path):
            print("📊 Cargando estadísticas de normalización existentes...")
            stats = torch.load(self.stats_path)
            self.x_mean = stats['x_mean']
            self.x_std = stats['x_std']
            self.y_mean = stats['y_mean']
            self.y_std = stats['y_std']
        else:
            print("📊 Calculando medias y desviaciones estándar en todo el dataset...")
            x_list, y_list = [], []
            # Tomamos una muestra representativa para calcular stats rápidamente
            for i in range(len(self)):
                x, y = self._get_raw_item(i)
                x_list.append(x)
                y_list.append(y)
                
            x_all = torch.stack(x_list, dim=0) # (N, 24, H, W)
            y_all = torch.stack(y_list, dim=0) # (N, 1, H, W)
            
            # Media y std por canal (dimensión 0, 2, 3)
            self.x_mean = x_all.mean(dim=[0, 2, 3], keepdim=True) # (1, 24, 1, 1)
            self.x_std = x_all.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-5) # Evitar división por cero
            
            self.y_mean = y_all.mean(dim=[0, 2, 3], keepdim=True) # (1, 1, 1, 1)
            self.y_std = y_all.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-5)
            
            torch.save({
                'x_mean': self.x_mean, 'x_std': self.x_std,
                'y_mean': self.y_mean, 'y_std': self.y_std
            }, self.stats_path)
            print("✅ ¡Estadísticas de normalización guardadas en models_checkpoints/normalization_stats.pt!")

    def _get_raw_item(self, idx):
        t = self.times[idx]
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
            
        channels.append(self.dem_np)
        x_np = np.stack(channels, axis=0)
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        
        imerg_slice = self.imerg_ds.sel(time=t, method='nearest')
        y_np = imerg_slice['precipitation'].values 
        y_np = np.nan_to_num(y_np, nan=0.0)
        
        if len(y_np.shape) == 2:
            y_np = np.expand_dims(y_np, axis=0)
            
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        x_tensor, y_tensor = self._get_raw_item(idx)
        
        if self.normalize:
            x_tensor = (x_tensor - self.x_mean.squeeze(0)) / self.x_std.squeeze(0)
            y_tensor = (y_tensor - self.y_mean.squeeze(0)) / self.y_std.squeeze(0)
            
        return x_tensor, y_tensor