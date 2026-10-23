import os
import torch
import xarray as xr
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ColombiaWeatherDataset(Dataset):
    def __init__(self, data_dir, meses=["01"], normalize=True):
        self.data_dir = data_dir
        self.meses = meses
        self.normalize = normalize
        
        self.samples = []         # Lista global de tuplas: (mes, timestamp_imerg)
        self.month_datasets = {}  # Caché RAM para almacenar meses bajo demanda
        self.dem_np = None
        
        self._indexar_meses()
        
        self.stats_path = os.path.join("models_checkpoints", "normalization_stats.pt")
        if self.normalize:
            self._pre_calcular_o_cargar_stats()

    def _indexar_meses(self):
        print("📅 Indexando tiempos mensuales de IMERG y ERA5 (resolución de 30 min)...")
        for mes in self.meses:
            era5_path = os.path.join(self.data_dir, f"era5_2024/era5_complete_2024_{mes}.nc")
            imerg_path = os.path.join(self.data_dir, f"imerg_2024/imerg_complete_2024_{mes}.nc")
            
            if os.path.exists(era5_path) and os.path.exists(imerg_path):
                ds_imerg = xr.open_dataset(imerg_path)
                times_imerg = pd.to_datetime([str(t) for t in ds_imerg.time.values])
                ds_imerg.close()
                
                for t in times_imerg:
                    self.samples.append((mes, t))
                    
        print(f"✅ Total de muestras indexadas (alta resolución): {len(self.samples)}")

    def _cargar_mes_si_no_existe(self, mes):
        if mes not in self.month_datasets:
            era5_path = os.path.join(self.data_dir, f"era5_2024/era5_complete_2024_{mes}.nc")
            imerg_path = os.path.join(self.data_dir, f"imerg_2024/imerg_complete_2024_{mes}.nc")
            
            # Cargamos este mes específico completo en RAM de forma aislada
            era5_ds = xr.open_dataset(era5_path).load()
            imerg_ds = xr.open_dataset(imerg_path).load()
            
            # Normalizar fechas y eliminar duplicados en la dimensión temporal
            era5_ds['valid_time'] = pd.to_datetime([str(t) for t in era5_ds.valid_time.values])
            era5_ds = era5_ds.drop_duplicates(dim='valid_time', keep='first')
            
            imerg_ds['time'] = pd.to_datetime([str(t) for t in imerg_ds.time.values])
            imerg_ds = imerg_ds.drop_duplicates(dim='time', keep='first')
            
            # Cargar DEM una sola vez
            if self.dem_np is None:
                dem_path = os.path.join(self.data_dir, "dem_colombia_era5.nc")
                if os.path.exists(dem_path):
                    dem_ds = xr.open_dataset(dem_path)
                    self.dem_np = dem_ds['elevation'].values
                    dem_ds.close()
                else:
                    raise FileNotFoundError(f"❌ No se encontró el DEM en: {dem_path}")

            self.month_datasets[mes] = {
                'era5': era5_ds,
                'imerg': imerg_ds
            }

    def _get_raw_item(self, idx):
        mes, t = self.samples[idx]
        self._cargar_mes_si_no_existe(mes)
        
        era5_ds = self.month_datasets[mes]['era5']
        imerg_ds = self.month_datasets[mes]['imerg']
        
        # Sincronización temporal: Buscar el ERA5 más cercano al timestamp de 30-min de IMERG
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
            
        channels.append(self.dem_np)

        # 🕒 Inyección de Codificación Temporal Cíclica (Hora y Día del año)
        t_dt = pd.to_datetime(t)
        hora_decimal = t_dt.hour + t_dt.minute / 60.0
        doy = t_dt.dayofyear
        
        # Codificación trigonométrica (Evita saltos bruscos en la medianoche o fin de año)
        h_sin = np.sin(2 * np.pi * hora_decimal / 24.0)
        h_cos = np.cos(2 * np.pi * hora_decimal / 24.0)
        d_sin = np.sin(2 * np.pi * doy / 365.25)
        d_cos = np.cos(2 * np.pi * doy / 365.25)
        
        H, W = self.dem_np.shape
        channels.append(np.full((H, W), h_sin, dtype=np.float32))
        channels.append(np.full((H, W), h_cos, dtype=np.float32))
        channels.append(np.full((H, W), d_sin, dtype=np.float32))
        channels.append(np.full((H, W), d_cos, dtype=np.float32))
        
        x_np = np.stack(channels, axis=0) # Ahora tendrá 28 canales en total
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        
        # Obtener precipitación de IMERG exacta
        imerg_slice = imerg_ds.sel(time=t, method='nearest')
        y_np = imerg_slice['precipitation'].values 
        y_np = np.nan_to_num(y_np, nan=0.0)
        
        if len(y_np.shape) == 2:
            y_np = np.expand_dims(y_np, axis=0)
            
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_tensor, y_tensor = self._get_raw_item(idx)
        
        if self.normalize:
            x_tensor = (x_tensor - self.x_mean.squeeze(0)) / self.x_std.squeeze(0)
            y_tensor = (y_tensor - self.y_mean.squeeze(0)) / self.y_std.squeeze(0)
            
        return x_tensor, y_tensor

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
            print("📊 Calculando estadísticas con un submuestreo representativo (para evitar picos de RAM)...")
            x_list, y_list = [], []
            # Tomamos una muestra cada 50 elementos para calcular las estadísticas globales rápidamente
            step_sample = max(1, len(self) // 150)
            for i in range(0, len(self), step_sample):
                x, y = self._get_raw_item(i)
                x_list.append(x)
                y_list.append(y)
                
            x_all = torch.stack(x_list, dim=0)
            y_all = torch.stack(y_list, dim=0)
            
            self.x_mean = x_all.mean(dim=[0, 2, 3], keepdim=True)
            self.x_std = x_all.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-5)
            self.y_mean = y_all.mean(dim=[0, 2, 3], keepdim=True)
            self.y_std = y_all.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-5)
            
            torch.save({
                'x_mean': self.x_mean, 'x_std': self.x_std,
                'y_mean': self.y_mean, 'y_std': self.y_std
            }, self.stats_path)
            print("✅ ¡Estadísticas de normalización guardadas en models_checkpoints/normalization_stats.pt!")