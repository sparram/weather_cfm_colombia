import os
import torch
from torch.utils.data import Dataset

class ColombiaWeatherDataset(Dataset):
    def __init__(self, tensors_dir="data/tensors/", meses=["01"], normalize=True):
        self.tensors_dir = tensors_dir
        self.meses = meses
        self.normalize = normalize
        
        self.samples_index = []  # (mes, local_idx)
        self.monthly_X = {}
        self.monthly_Y = {}
        
        print(f"📦 Mapeando archivos .pt en memoria para los meses: {meses}...")
        for mes in meses:
            path = os.path.join(tensors_dir, f"mes_{mes}.pt")
            if os.path.exists(path):
                # mmap=True evita cargar todo a la RAM física de golpe
                data = torch.load(path, map_location="cpu", mmap=True)
                self.monthly_X[mes] = data['X']
                self.monthly_Y[mes] = data['Y']
                
                n_samples = len(data['X'])
                for local_idx in range(n_samples):
                    self.samples_index.append((mes, local_idx))
                
        print(f"✅ ¡Mapeo exitoso! Total de muestras indexadas: {len(self.samples_index)}")
        
        self.stats_path = os.path.join("models_checkpoints", "normalization_stats.pt")
        if self.normalize:
            self._pre_calcular_o_cargar_stats()

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        mes, local_idx = self.samples_index[idx]
        
        # PyTorch extrae el dato directamente del archivo mapeado en disco de forma eficiente
        x = self.monthly_X[mes][local_idx]
        y = self.monthly_Y[mes][local_idx]
        
        if self.normalize:
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
            print("📊 Calculando estadísticas globales...")
            primer_mes = self.meses[0]
            X_sample = self.monthly_X[primer_mes]
            Y_sample = self.monthly_Y[primer_mes]
            
            self.x_mean = X_sample.mean(dim=[0, 2, 3])[:, None, None]
            self.x_std = X_sample.std(dim=[0, 2, 3])[:, None, None].clamp(min=1e-5)
            self.y_mean = Y_sample.mean(dim=[0, 2, 3])[:, None, None]
            self.y_std = Y_sample.std(dim=[0, 2, 3])[:, None, None].clamp(min=1e-5)
            
            torch.save({
                'x_mean': self.x_mean, 'x_std': self.x_std,
                'y_mean': self.y_mean, 'y_std': self.y_std
            }, self.stats_path)
            print("✅ Estadísticas guardadas con éxito.")