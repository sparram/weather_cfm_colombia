import os
import torch
from torch.utils.data import Dataset

class ColombiaWeatherDataset(Dataset):
    def __init__(self, tensors_dir="data/tensors/", meses=["01"], normalize=True):
        self.tensors_dir = tensors_dir
        self.meses = meses
        self.normalize = normalize
        
        self.samples_index = []  
        self.month_cache = {}    
        
        print(f"📦 Indexando metadatos para los meses: {meses}...")
        for mes in meses:
            path = os.path.join(tensors_dir, f"mes_{mes}.pt")
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu")
                n_samples = len(data['X'])
                for local_idx in range(n_samples):
                    self.samples_index.append((mes, local_idx))
                del data
                
        print(f"✅ Total de muestras indexadas (sin saturar RAM): {len(self.samples_index)}")
        
        self.stats_path = os.path.join("models_checkpoints", "normalization_stats.pt")
        if self.normalize:
            self._pre_calcular_o_cargar_stats()

    def _cargar_mes_si_no_existe(self, mes):
        if mes not in self.month_cache:
            if len(self.month_cache) >= 2:
                self.month_cache.clear()
                
            path = os.path.join(self.tensors_dir, f"mes_{mes}.pt")
            data = torch.load(path, map_location="cpu")
            self.month_cache[mes] = {
                'X': data['X'],
                'Y': data['Y']
            }

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        mes, local_idx = self.samples_index[idx]
        self._cargar_mes_si_no_existe(mes)
        
        # x tiene forma (28, H, W)
        x = self.month_cache[mes]['X'][local_idx]
        # y tiene forma (1, H, W)
        y = self.month_cache[mes]['Y'][local_idx]
        
        if self.normalize:
            # self.x_mean tiene forma (28, 1, 1), preservando exactamente (28, H, W)
            x = (x - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std
            
        return x, y

    def _pre_calcular_o_cargar_stats(self):
        os.makedirs("models_checkpoints", exist_ok=True)
        if os.path.exists(self.stats_path):
            stats = torch.load(self.stats_path)
            self.x_mean = stats['x_mean']
            self.x_std = stats['x_std']
            self.y_mean = stats['y_mean']
            self.y_std = stats['y_std']
        else:
            print("📊 Calculando estadísticas globales de forma eficiente...")
            primer_mes = self.meses[0]
            path = os.path.join(self.tensors_dir, f"mes_{primer_mes}.pt")
            data = torch.load(path, map_location="cpu")
            X_sample = data['X'] # (N, 28, H, W)
            Y_sample = data['Y'] # (N, 1, H, W)
            
            # Ajustar formas a (28, 1, 1) y (1, 1, 1) para broadcasting seguro con 3D
            self.x_mean = X_sample.mean(dim=[0, 2, 3])[:, None, None]
            self.x_std = X_sample.std(dim=[0, 2, 3])[:, None, None].clamp(min=1e-5)
            self.y_mean = Y_sample.mean(dim=[0, 2, 3])[:, None, None]
            self.y_std = Y_sample.std(dim=[0, 2, 3])[:, None, None].clamp(min=1e-5)
            
            torch.save({
                'x_mean': self.x_mean, 'x_std': self.x_std,
                'y_mean': self.y_mean, 'y_std': self.y_std
            }, self.stats_path)
            print("✅ Estadísticas guardadas con éxito.")