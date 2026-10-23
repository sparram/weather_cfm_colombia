import os
# Solución al conflicto de librerías OpenMP en Windows/Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import ColombiaWeatherDataset
from src.models.unet import ConditionalUNet

def ejecutar_inferencia():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo para inferencia: {device}")
    
    # 1. Cargar el modelo entrenado
    print("🧠 Cargando la U-Net y los pesos del último checkpoint...")
    model = ConditionalUNet(in_channels_cond=24, base_channels=32).to(device)
    
    ckpt_path = "models_checkpoints/cfm_unet_epoch_5.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ No se encontró el checkpoint en {ckpt_path}. Entrena el modelo primero.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print("✅ ¡Pesos cargados con éxito!")
    
    # 2. Cargar una muestra de prueba del dataset
    print("📦 Cargando datos de validación para prueba...")
    dataset = ColombiaWeatherDataset(data_dir="data/processed/", meses=["01"])
    
    # Tomamos la primera muestra del dataset (puedes cambiar el índice si quieres ver otra)
    idx_muestra = 0
    x_cond, y_real = dataset[idx_muestra]
    
    # Añadir dimensión de lote (Batch = 1)
    x_cond_batch = x_cond.unsqueeze(0).to(device) # (1, 24, 70, 62)
    y_real_np = y_real.squeeze(0).cpu().numpy()  # (153, 173)
    
    # 3. Proceso de Generación por Flow Matching (Integración de Euler)
    print("🌊 Generando campo de precipitación mediante Flow Matching (Integración de Euler)...")
    
    # Número de pasos de integración (más pasos = mayor precisión, 10 o 20 es ideal para empezar)
    num_steps = 20
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        # A. Inicializar con ruido gaussiano x_0 a t = 0
        xt = torch.randn(1, 1, 153, 173, device=device)
        
        # B. Bucle de integración temporal de t = 0 a t = 1
        for step in range(num_steps):
            t_val = step * dt
            # Crear un tensor con el tiempo t para el batch
            t_tensor = torch.full((1,), t_val, device=device)
            
            # Predecir la velocidad v_t con la U-Net
            v_pred = model(xt, x_cond_batch, t_tensor)
            
            # Actualizar el estado xt usando el paso de Euler: xt_new = xt + v * dt
            xt = xt + v_pred * dt
            
        # El resultado final en t = 1 es nuestra precipitación generada
        y_pred = xt.squeeze(0).squeeze(0).cpu().numpy() # (153, 173)
        
    # Aplicar un clip a 0 (la precipitación no puede ser negativa)
    y_pred = np.clip(y_pred, 0.0, None)
    
    print("📊 ¡Generación completada! Creando gráfico comparativo...")
    
    # 4. Visualización comparativa
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # IMERG Real
    im0 = axes[0].imshow(y_real_np, cmap='Blues', origin='lower')
    axes[0].set_title("IMERG Real (Objetivo de Alta Resolución)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Precipitación Generada por el Modelo
    im1 = axes[1].imshow(y_pred, cmap='Blues', origin='lower')
    axes[1].set_title("CFM Generado (Downscaling Sintético)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_img = "results/comparacion_inferencia.png"
    plt.savefig(output_img, dpi=300)
    print(f"💾 ¡Gráfico guardado con éxito en: {output_img}!")
    #plt.show()

if __name__ == "__main__":
    ejecutar_inferencia()