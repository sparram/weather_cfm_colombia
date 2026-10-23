import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Subset
from src.dataset import ColombiaWeatherDataset
from src.models.unet import ConditionalUNet

def ejecutar_inferencia():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo para inferencia: {device}")
    
    # 1. Cargar el modelo entrenado
    model = ConditionalUNet(in_channels_cond=28, base_channels=32).to(device)
    ckpt_path = "models_checkpoints/best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Cargar estadísticas de normalización guardadas
    stats_path = "models_checkpoints/normalization_stats.pt"
    stats = torch.load(stats_path)
    y_mean = stats['y_mean'].item() # Escalar para desnormalizar
    y_std = stats['y_std'].item()
    
    # 3. Cargar los 12 meses y aplicar el mismo split temporal de validación
    meses_ano = [f"{i:02d}" for i in range(1, 13)]
    dataset = ColombiaWeatherDataset(data_dir="data/processed/", meses=meses_ano, normalize=True)
    
    n_total = len(dataset)
    split_idx = int(n_total * 0.8)
    val_indices = list(range(split_idx, n_total))
    val_dataset = Subset(dataset, val_indices)
    
    # Elegir una muestra del conjunto de validación (ej: la primera de noviembre/diciembre)
    idx_val = 0
    x_cond, y_norm_tensor = val_dataset[idx_val]
    
    # Mapear al índice real del dataset original para obtener el IMERG en milímetros reales
    real_idx = val_indices[idx_val]
    _, y_real_tensor_raw = dataset._get_raw_item(real_idx)
    y_real_np = y_real_tensor_raw.squeeze(0).numpy() # (153, 173) en mm reales
    
    x_cond_batch = x_cond.unsqueeze(0).to(device)
    
    # 4. Inferencia por Flow Matching
    print("🌊 Generando campo normalizado mediante Flow Matching...")
    num_steps = 20
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        xt = torch.randn(1, 1, 153, 173, device=device)
        
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((1,), t_val, device=device)
            v_pred = model(xt, x_cond_batch, t_tensor)
            xt = xt + v_pred * dt
            
        y_pred_norm = xt.squeeze(0).squeeze(0).cpu().numpy()
        
    # 5. Desnormalizar para volver a milímetros reales (x * std + mean)
    y_pred = (y_pred_norm * y_std) + y_mean
    y_pred = np.clip(y_pred, 0.0, None) # Evitar negativos físicos
    
    print(f"📊 ¡Generación lista! Rango real: {y_real_np.min():.1f} a {y_real_np.max():.1f} mm")
    print(f"📊 Rango generado: {y_pred.min():.1f} a {y_pred.max():.1f} mm")
    
    # 6. Gráfico comparativo
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    im0 = axes[0].imshow(y_real_np, cmap='Blues', origin='lower')
    axes[0].set_title("IMERG Real (Milímetros)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(y_pred, cmap='Blues', origin='lower')
    axes[1].set_title("CFM Generado y Desnormalizado (Milímetros)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_img = "results/comparacion_inferencia_normalizada.png"
    plt.savefig(output_img, dpi=300)
    print(f"💾 ¡Gráfico guardado en: {output_img}!")

if __name__ == "__main__":
    ejecutar_inferencia()