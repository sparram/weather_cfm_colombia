import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import ColombiaWeatherDataset
from src.models.unet import ConditionalUNet

def entrenar_modelo():
    # 1. Configuración de dispositivos e hiperparámetros
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")
    
    batch_size = 4
    epochs = 50
    lr = 1e-4
    
    # 2. Cargar Dataset completo con los 12 meses de 2024
    print("📦 Cargando dataset completo para el año 2024 con normalización...")
    meses_ano = [f"{i:02d}" for i in range(1, 13)]
    dataset = ColombiaWeatherDataset(data_dir="data/processed/", meses=meses_ano, normalize=True)
    
    # 3. División temporal por bloques (Validación estricta sin fuga de datos)
    # En lugar de random_split, separamos por bloques continuos (ej: cada 5 muestras, 1 para validación)
    # O porcentaje cronológico: el 80% inicial para entrenar, el 20% final para validar.
    n_total = len(dataset)
    indices = list(range(n_total))
    
    # Ejemplo: Validación basada en bloques temporales (ej. el último 20% del tiempo de cada mes o del año)
    split_idx = int(n_total * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"📊 Total de muestras: {n_total} | Entrenamiento (Bloque Temporal 1): {len(train_indices)} | Validación (Bloque Temporal 2): {len(val_indices)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # 4. Inicializar el modelo U-Net
    print("🧠 Inicializando ConditionalUNet...")
    model = ConditionalUNet(in_channels_cond=28, base_channels=32).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    os.makedirs("models_checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = "models_checkpoints/best_model.pt"
    
    print(f"🚀 Iniciando entrenamiento por {epochs} épocas con partición temporal...\n")
    
    for epoch in range(epochs):
        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs} [Train]", unit="batch")
        for x_cond, x1 in train_pbar:
            x_cond = x_cond.to(device)
            x1 = x1.to(device)
            bs = x1.shape[0]
            
            x0 = torch.randn_like(x1)
            t = torch.rand(bs, device=device)
            t_expanded = t[:, None, None, None]
            xt = (1.0 - t_expanded) * x0 + t_expanded * x1
            v_target = x1 - x0
            
            optimizer.zero_grad()
            v_pred = model(xt, x_cond, t)
            loss = torch.nn.functional.mse_loss(v_pred, v_target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- FASE DE VALIDACIÓN ---
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Época {epoch+1}/{epochs} [Val]  ", unit="batch")
        with torch.no_grad():
            for x_cond_val, x1_val in val_pbar:
                x_cond_val = x_cond_val.to(device)
                x1_val = x1_val.to(device)
                bs_val = x1_val.shape[0]
                
                x0_val = torch.randn_like(x1_val)
                t_val = torch.rand(bs_val, device=device)
                t_val_expanded = t_val[:, None, None, None]
                xt_val = (1.0 - t_val_expanded) * x0_val + t_val_expanded * x1_val
                v_target_val = x1_val - x0_val
                
                v_pred_val = model(xt_val, x_cond_val, t_val)
                loss_val = torch.nn.functional.mse_loss(v_pred_val, v_target_val)
                
                val_loss += loss_val.item()
                val_pbar.set_postfix(val_loss=f"{loss_val.item():.4f}")
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"🎯 Resultados Época {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Guardar el mejor modelo según validación temporal
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"⭐ ¡Nuevo mejor modelo guardado en: {best_model_path} (Val Loss: {best_val_loss:.4f})!\n")
        else:
            print(f"💾 Checkpoint regular guardado (Mejor Val Loss actual: {best_val_loss:.4f})\n")
            torch.save(model.state_dict(), f"models_checkpoints/cfm_unet_epoch_{epoch+1}.pt")

    print("\n🎉 ¡Entrenamiento con validación temporal completado con éxito!")

if __name__ == "__main__":
    entrenar_modelo()