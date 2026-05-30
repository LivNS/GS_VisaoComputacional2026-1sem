"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definição das arquiteturas CNN para classificação de detritos orbitais.

Integrantes: 
Debora da Silva Amaral - RM 550412 
Eduardo Pielich - RM 99767 
Gabriel Machado - RM 99880
Livia Namba Seraphim - RM 97819 
Vitor Hugo Rodrigues - RM 97758

Global Solution — Applied Computer Vision — 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn

# Constantes 

NUM_CLASSES  = 4
IMG_CHANNELS = 3
IMG_SIZE     = 128   # px

CLASS_NAMES = [
    'critical_fragment',
    'orbital_structure',
    'satellite_operational',
    'small_debris'
]


# CNN 1 — Baseline

class CNN1_Baseline(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(CNN1_Baseline, self).__init__()

        self.features = nn.Sequential(
            # Bloco 1 — features de baixo nível: bordas, gradientes, texturas simples
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          

            # Bloco 2 — features de nível médio: formas, padrões locais
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                   
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),                    
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─── CNN 2 — Regularizada ─────────────────────────────────────────────────────

class CNN2_Regularized(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(CNN2_Regularized, self).__init__()

        self.features = nn.Sequential(
            # ── Bloco 1 ──────────────────────────────────────────────────────
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             
            nn.Dropout2d(p=0.25),

            # ── Bloco 2 ──────────────────────────────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             
            nn.Dropout2d(p=0.25),

            # ── Bloco 3 ──────────────────────────────────────────────────────
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                  
            nn.Linear(128 * 16 * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),                    
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# Utilitários 
def count_parameters(model: nn.Module) -> int:
    """Conta o total de parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, model_name: str) -> str:
    """Retorna um resumo legível da arquitetura."""
    lines = [
        f"{'='*50}",
        f"  {model_name}",
        f"{'='*50}",
        str(model),
        f"{'─'*50}",
        f"  Parâmetros treináveis: {count_parameters(model):,}",
        f"  Entrada esperada: ({IMG_CHANNELS}, {IMG_SIZE}, {IMG_SIZE})",
        f"  Saída: ({NUM_CLASSES},) — logits por classe",
        f"{'='*50}",
    ]
    return '\n'.join(lines)


def build_models(device: torch.device = None) -> tuple:
    """
    Instancia e retorna ambos os modelos prontos para treinamento.

    Returns:
        (CNN1_Baseline, CNN2_Regularized) movidos para o device especificado.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1 = CNN1_Baseline(num_classes=NUM_CLASSES).to(device)
    model2 = CNN2_Regularized(num_classes=NUM_CLASSES).to(device)
    return model1, model2


# Execução direta: exibe resumo das arquiteturas

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    m1, m2 = build_models(device)

    print(get_model_summary(m1, 'CNN 1 — Baseline'))
    print()
    print(get_model_summary(m2, 'CNN 2 — Regularizada'))
    print()

    # Teste de forward pass
    dummy = torch.randn(4, IMG_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
    out1  = m1(dummy)
    out2  = m2(dummy)
    print(f'Forward pass OK!')
    print(f'  CNN1 saída: {out1.shape}   (esperado: torch.Size([4, 4]))')
    print(f'  CNN2 saída: {out2.shape}   (esperado: torch.Size([4, 4]))')

    # Tabela comparativa
    print('\n  Comparação de parâmetros:')
    print(f'  {"Modelo":<25} {"Parâmetros":>12}')
    print(f'  {"─"*38}')
    print(f'  {"CNN 1 — Baseline":<25} {count_parameters(m1):>12,}')
    print(f'  {"CNN 2 — Regularizada":<25} {count_parameters(m2):>12,}')
    ratio = count_parameters(m2) / count_parameters(m1)
    print(f'  CNN 2 tem {ratio:.1f}× mais parâmetros que CNN 1.')
