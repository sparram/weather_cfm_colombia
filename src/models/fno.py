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


class SpectralConv2d(nn.Module):
    """Capa de convolución espectral en el dominio de Fourier para FNO."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Modos de Fourier a retener (bajas frecuencias)
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Transformada rápida de Fourier 2D real
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Retorno al espacio físico
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class ConditionalFNO2d(nn.Module):
    """
    Fourier Neural Operator adaptado para Flow Matching (FNO-CFM) en downscaling meteorológico.
    Entradas:
        - x_t: Estado de precipitación (B, 1, H, W)
        - cond: Variables de ERA5 + DEM interpoladas (B, 24, H, W) -> Total canales entrada = 29
        - t: Tiempo del flujo (B,) en el rango [0, 1]
    Salida:
        - Campo vectorial de velocidad predicho v_t (B, 1, H, W)
    """
    def __init__(self, in_channels_cond=28, modes=12, width=64):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        
        # Canal de entrada: 1 (x_t) + 28 (condición) = 29 canales
        total_in_channels = 1 + in_channels_cond
        
        # Embedding temporal
        time_dim = width * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(width),
            nn.Linear(width, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, width)
        )
        
        # Capas del FNO
        self.fc0 = nn.Conv2d(total_in_channels, self.width, 1) # Capa de elevación

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        # Proyección de tiempo hacia los canales intermedios
        self.time_proj = nn.Linear(width, width)

        # Capas de proyección final (decodificador)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1) # Salida: 1 canal (velocidad v_t)

    def forward(self, x_t, cond, t):
        # 🛡️ Escudo defensivo para dimensiones
        if cond.ndim == 3:
            cond = cond.unsqueeze(0)

        # 1. Asegurar resolución espacial coincidente
        if cond.shape[-2:] != x_t.shape[-2:]:
            cond = F.interpolate(cond, size=x_t.shape[-2:], mode='bilinear', align_corners=False)
            
        # 2. Concatenar estado x_t y condiciones físicas
        x = torch.cat([x_t, cond], dim=1) # (B, 29, H, W)
        
        # 3. Embedding temporal
        t_emb = self.time_mlp(t) # (B, width)
        t_cond = self.time_proj(t_emb)[:, :, None, None] # (B, width, 1, 1)

        # 4. Procesamiento FNO
        x = self.fc0(x)
        
        # Bloque FNO 1 + Conexión Residual + Inyección Temporal
        x1 = self.conv0(x) + self.w0(x) + t_cond
        x1 = F.gelu(x1)

        # Bloque FNO 2
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)

        # Bloque FNO 3
        x3 = self.conv2(x2) + self.w2(x2)
        x = F.gelu(x3)

        # 5. Proyección final al espacio físico de salida (B, 1, H, W)
        x = x.permute(0, 2, 3, 1) # (B, H, W, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # (B, H, W, 1)
        x = x.permute(0, 3, 1, 2) # (B, 1, H, W)
        
        return x