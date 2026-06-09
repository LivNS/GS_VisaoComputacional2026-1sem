# 🛰️ Classificação de Detritos Orbitais — Applied Computer Vision
## Global Solution 2026 · FIAP · Indústria Espacial

### Integrantes
- Debora da Silva Amaral — RM 550412
- Eduardo Pielich — RM 99767
- Gabriel Machado — RM 99880
- Livia Namba Seraphim — RM 97819
- Vitor Hugo Rodrigues — RM 97758

---

## Sobre o projeto

Sistema de visão computacional para classificação automática de objetos em
órbita terrestre em 4 categorias, usando redes neurais convolucionais
treinadas do zero com PyTorch.

O lixo espacial é um problema crescente: existem mais de 36.000 objetos
rastreados em órbita e estimativa de 1 milhão de fragmentos não rastreáveis.
A classificação automática por imagem acelera a triagem de objetos detectados
por sensores ópticos orbitais.

### Classes (ordem alfabética = ordem do ImageFolder)

| Índice | Classe | Risco |
|--------|--------|-------|
| 0 | `critical_fragment` — Fragmento Crítico (>10 cm) | Alto |
| 1 | `orbital_structure` — Estrutura Orbital inativa | Moderado |
| 2 | `satellite_operational` — Satélite Operacional | Baixo |
| 3 | `small_debris` — Pequeno Detrito (<10 cm) | Médio |

---

## Estrutura do projeto

```
space_debris_acv/
├── notebook_treinamento.ipynb      Notebook principal (gera dataset, treina, avalia)
├── requirements.txt                Dependências
├── README.md                       Este arquivo
│
├── models/
│   ├── architectures.py            Definição das 2 CNNs
│   ├── cnn1_baseline_best.pth       Pesos treinados CNN 1
│   ├── cnn2_regularized_best.pth    Pesos treinados CNN 2
│   └── PESOS.txt                   Documentação dos pesos
│
├── dataset_builder/
│   └── build_dataset.py            Gerador do dataset sintético
│
├── demo/
│   ├── app.py                      Interface Gradio
│   └── api_fastapi.py              API FastAPI
│
└── docs/
    └── dataset_samples/            Amostras do dataset (8 por classe)
```

---

## Instalação

```bash
cd space_debris_acv
pip install -r requirements.txt
```

---

## Como executar

### 1. Notebook de treinamento (gera tudo)

```bash
jupyter notebook notebook_treinamento.ipynb
```

Faça Kernel → Restart & Run All. Ele gera o dataset, treina as duas CNNs,
gera gráficos e salva os pesos `.pth`.

Os pesos já treinados estão incluídos em `models/`, então você pode pular
direto para a demonstração se quiser.

### 2. Demonstração — escolha uma das formas

**Gradio (interface visual):**
```bash
python demo/app.py
# Acesse http://localhost:7860
```

**FastAPI (API REST):**
```bash
uvicorn demo.api_fastapi:app --reload --port 8000
# Acesse http://localhost:8000/docs
```

---

## Arquiteturas

### CNN 1 — Baseline
```
Conv(3→32) → ReLU → MaxPool
Conv(32→64) → ReLU → MaxPool
Flatten → Linear(65536→128) → ReLU → Linear(128→4)
```
Sem regularização. ~1.1M parâmetros.

### CNN 2 — Regularizada
```
[Conv(3→32)+BN+ReLU]×2 → MaxPool → Dropout(0.25)
[Conv(32→64)+BN+ReLU]×2 → MaxPool → Dropout(0.25)
Conv(64→128)+BN+ReLU → MaxPool
Flatten → Linear(32768→256)+BN+ReLU+Dropout(0.5) → Linear(256→4)
```
BatchNormalization + Dropout. ~2.4M parâmetros.

---

## Dataset

2.000 imagens sintéticas (500 por classe), 128×128 px, geradas com Pillow
simulando a aparência de objetos capturados por câmeras de rastreamento orbital.
Divisão 70% treino / 15% validação / 15% teste, com embaralhamento estratificado
e augmentation (flip, rotação, color jitter) aplicado apenas no treino.

---

## Link Youtube

https://youtu.be/ZUAvVeypENA

---

*Global Solution 2026 — FIAP · Applied Computer Vision*
