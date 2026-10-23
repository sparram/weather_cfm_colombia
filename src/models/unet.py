import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """Genera embeddings sinusoidales para el tiempo t del Flow Matching."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DoubleConv(nn.Module):
    """Bloque convolucional doble estándar: (Conv -> GN -> SiLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


class ConditionalUNet(nn.Module):
    """
    U-Net adaptada para Conditional Flow Matching (CFM) en downscaling meteorológico.
    Entradas:
        - x_t: Estado de precipitación interpolado/ruidoso (B, 1, H, W)
        - cond: Variables de ERA5 + DEM (B, 24, H_low, W_low) -> se interpolarán a (B, 24, H, W)
        - t: Tiempo del flujo (B,) en el rango [0, 1]
    Salida:
        - Campo vectorial de velocidad predicho v_t (B, 1, H, W)
    """
    def __init__(self, in_channels_cond=24, base_channels=64):
        super().__init__()
        
        # Canal de entrada: 1 (estado x_t) + 24 (condicionamiento ERA5/DEM interpolado) = 25 canales
        total_in_channels = 1 + in_channels_cond
        
        # Embedding temporal
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder (Contracción)
        self.inc = DoubleConv(total_in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))
        
        # Cuello de botella (Bottleneck) con inyección de tiempo
        self.bot1 = DoubleConv(base_channels * 4, base_channels * 4)
        self.time_proj = nn.Linear(time_dim, base_channels * 4)
        self.bot2 = DoubleConv(base_channels * 4, base_channels * 4)
        
        # Decoder (Expansión con Skip Connections)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_channels * 2, base_channels)
        
        # Capa de salida para predecir el campo de velocidad v_t (1 canal de precipitación)
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x_t, cond, t):
        # 1. Asegurar que las condiciones (ERA5+DEM) tengan la misma resolución espacial que x_t (alta resolución)
        if cond.shape[-2:] != x_t.shape[-2:]:
            cond = F.interpolate(cond, size=x_t.shape[-2:], mode='bilinear', align_corners=False)
            
        # 2. Concatenar estado x_t y condiciones físicas en la dimensión de canales
        x = torch.cat([x_t, cond], dim=1) # (B, 25, H, W)
        
        # 3. Calcular embedding temporal
        t_emb = self.time_mlp(t) # (B, time_dim)
        
        # 4. Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # 5. Bottleneck + Inyección de tiempo
        b = self.bot1(x3)
        # Sumar el embedding temporal proyectado al cuello de botella
        b = b + self.time_proj(t_emb)[:, :, None, None]
        b = self.bot2(b)
        
        # 6. Decoder con Skip Connections (ajustando tamaños si hay impares por las dimensiones de la malla)
        x = self.up1(b)
        # Manejar posibles diferencias de tamaño por redondeo en maxpool/upsample
        if x.shape != x2.shape:
            x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        if x.shape != x1.shape:
            x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)
        
        # 7. Salida final (velocidad del flujo)
        out = self.outc(x)
        return out


# ==========================================
# BLOQUE DE PRUEBA Y VALIDACIÓN DEL MODELO
# ==========================================
if __name__ == "__main__":
    print("🧪 Probando la arquitectura de la U-Net para CFM...")
    
    # Creamos un tensor simulado de lote (Batch size = 2)
    # x_t de alta resolución (IMERG): (2, 1, 153, 173)
    x_t_dummy = torch.randn(2, 1, 153, 173)
    # cond de baja resolución (ERA5 + DEM): (2, 24, 70, 62)
    cond_dummy = torch.randn(2, 24, 70, 62)
    # Tiempo t aleatorio entre 0 y 1: (2,)
    t_dummy = torch.rand(2)
    
    model = ConditionalUNet(in_channels_cond=24, base_channels=32)
    
    with torch.no_grad():
        v_pred = model(x_t_dummy, cond_dummy, t_dummy)
        
    print(f"✅ ¡Modelo U-Net inicializado y probado con éxito!")
    print(f"   - Forma de la salida (Campo de velocidad v_t): {v_pred.shape} --> Esperado: (2, 1, 153, 173)")