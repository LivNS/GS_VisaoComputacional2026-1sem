import sys
import os
import io
import base64
import time
from pathlib import Path
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.architectures import (
    CNN1_Baseline, CNN2_Regularized,
    CLASS_NAMES, IMG_SIZE, NUM_CLASSES
)

# Configuração 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_META = {
    'satellite_operational': {
        'label':       'Satélite Operacional',
        'description': 'Satélite em plena operação com controle ativo.',
        'risk':        'Baixo',
        'risk_code':   1,
    },
    'critical_fragment': {
        'label':       'Fragmento Crítico',
        'description': 'Fragmento >10 cm, alto risco de colisão.',
        'risk':        'Alto',
        'risk_code':   4,
    },
    'small_debris': {
        'label':       'Pequeno Detrito',
        'description': 'Partícula <10 cm, rastreamento difícil.',
        'risk':        'Médio',
        'risk_code':   3,
    },
    'orbital_structure': {
        'label':       'Estrutura Orbital',
        'description': 'Estrutura inativa (estágio de foguete, módulo).',
        'risk':        'Moderado',
        'risk_code':   2,
    },
}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.05, 0.05, 0.07], std=[0.12, 0.12, 0.14])
])


# Schemas Pydantic

class ClassProbability(BaseModel):
    class_name:  str
    label:       str
    probability: float
    risk:        str

class PredictionResponse(BaseModel):
    success:          bool
    model_used:       str
    predicted_class:  str
    label:            str
    confidence:       float
    risk_level:       str
    risk_code:        int
    description:      str
    all_probabilities: list[ClassProbability]
    inference_ms:     float

class HealthResponse(BaseModel):
    status:     str
    device:     str
    models:     dict
    classes:    list[str]
    img_size:   int

class Base64Request(BaseModel):
    image_base64: str
    model:        Optional[str] = 'cnn2'


# Carregamento dos modelos 

def load_model(ModelClass, weights_path: str):
    model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)
    path  = Path(weights_path)
    loaded = False
    if path.exists():
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            loaded = True
        except Exception as e:
            print(f'⚠️  Erro ao carregar {path}: {e}')
    else:
        print(f'⚠️  Pesos não encontrados: {path} — usando pesos aleatórios.')
    model.eval()
    return model, loaded

print('Carregando modelos...')
MODEL1, MODEL1_LOADED = load_model(CNN1_Baseline,    'models/cnn1_baseline_best.pth')
MODEL2, MODEL2_LOADED = load_model(CNN2_Regularized, 'models/cnn2_regularized_best.pth')
print(f'CNN1: {"✅ pesos reais" if MODEL1_LOADED else "⚠️ pesos aleatórios"}')
print(f'CNN2: {"✅ pesos reais" if MODEL2_LOADED else "⚠️ pesos aleatórios"}')


# FastAPI app

app = FastAPI(
    title='🛰️ Classificador de Detritos Orbitais',
    description=(
        'API para classificação automática de objetos em órbita terrestre.\n\n'
        '**Global Solution 2025 · FIAP · Applied Computer Vision**\n\n'
        'Classifica imagens em 4 categorias:\n'
        '- `satellite_operational` — Satélite operacional\n'
        '- `critical_fragment` — Fragmento crítico (>10 cm)\n'
        '- `small_debris` — Pequeno detrito (<10 cm)\n'
        '- `orbital_structure` — Estrutura orbital inativa\n\n'
        'Dois modelos disponíveis: **cnn1** (baseline) e **cnn2** (regularizado, recomendado).'
    ),
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# Função de inferência 

@torch.no_grad()
def run_inference(pil_img: Image.Image, model_key: str = 'cnn2') -> dict:
    model      = MODEL2 if model_key == 'cnn2' else MODEL1
    model_name = 'CNN 2 — Regularizada' if model_key == 'cnn2' else 'CNN 1 — Baseline'

    t0     = time.perf_counter()
    tensor = TRANSFORM(pil_img.convert('RGB')).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    ms     = (time.perf_counter() - t0) * 1000

    idx  = int(probs.argmax())
    cls  = CLASS_NAMES[idx]
    meta = CLASS_META[cls]

    return {
        'success':          True,
        'model_used':       model_name,
        'predicted_class':  cls,
        'label':            meta['label'],
        'confidence':       float(probs[idx]),
        'risk_level':       meta['risk'],
        'risk_code':        meta['risk_code'],
        'description':      meta['description'],
        'all_probabilities': [
            ClassProbability(
                class_name=CLASS_NAMES[i],
                label=CLASS_META[CLASS_NAMES[i]]['label'],
                probability=float(probs[i]),
                risk=CLASS_META[CLASS_NAMES[i]]['risk'],
            )
            for i in range(NUM_CLASSES)
        ],
        'inference_ms': round(ms, 2),
    }


# Endpoints 
@app.get('/', summary='Informações da API')
def root():
    return {
        'name':        'Classificador de Detritos Orbitais',
        'version':     '1.0.0',
        'description': 'API de visão computacional para classificação orbital',
        'endpoints': {
            'docs':            '/docs',
            'health':          '/health',
            'classes':         '/classes',
            'predict_upload':  'POST /predict?model=cnn2',
            'predict_base64':  'POST /predict/base64',
            'sample_image':    'GET /sample/{class_name}',
        },
        'models': {
            'cnn1': 'CNN 1 — Baseline (referência)',
            'cnn2': 'CNN 2 — Regularizada (recomendado)',
        },
    }


@app.get('/health', response_model=HealthResponse, summary='Status dos modelos')
def health():
    return HealthResponse(
        status='ok',
        device=str(DEVICE),
        models={
            'cnn1': {'loaded': MODEL1_LOADED, 'name': 'CNN 1 — Baseline'},
            'cnn2': {'loaded': MODEL2_LOADED, 'name': 'CNN 2 — Regularizada'},
        },
        classes=CLASS_NAMES,
        img_size=IMG_SIZE,
    )


@app.get('/classes', summary='Lista de classes suportadas')
def get_classes():
    return {
        'total': NUM_CLASSES,
        'classes': [
            {
                'index':       i,
                'class_name':  CLASS_NAMES[i],
                **CLASS_META[CLASS_NAMES[i]],
            }
            for i in range(NUM_CLASSES)
        ]
    }


@app.post('/predict', response_model=PredictionResponse,
          summary='Classificar imagem (upload de arquivo)')
async def predict_upload(
    file:  UploadFile = File(..., description='Imagem JPG, PNG ou WEBP'),
    model: str        = 'cnn2',
):
    """
    Classifica uma imagem enviada como arquivo multipart.

    - **file**: imagem nos formatos JPG, PNG ou WEBP
    - **model**: `cnn2` (padrão, recomendado) ou `cnn1` (baseline)

    Retorna a classe predita, confiança, nível de risco e probabilidades para todas as classes.
    """
    if model not in ('cnn1', 'cnn2'):
        raise HTTPException(status_code=400, detail="model deve ser 'cnn1' ou 'cnn2'")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Arquivo deve ser uma imagem (image/*)')

    try:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Não foi possível ler a imagem enviada')

    try:
        result = run_inference(img, model_key=model)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Erro na inferência: {str(e)}')


@app.post('/predict/base64', response_model=PredictionResponse,
          summary='Classificar imagem (base64 JSON)')
def predict_base64(body: Base64Request):
    """
    Classifica uma imagem enviada como string base64 em JSON.

    Útil para integração com aplicações front-end ou outros serviços.

    Exemplo de request body:
    ```json
    {
      "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
      "model": "cnn2"
    }
    ```
    """
    model = body.model or 'cnn2'
    if model not in ('cnn1', 'cnn2'):
        raise HTTPException(status_code=400, detail="model deve ser 'cnn1' ou 'cnn2'")

    try:
        b64 = body.image_base64
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        img_bytes = base64.b64decode(b64)
        img       = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='base64 inválido ou não é uma imagem')

    try:
        result = run_inference(img, model_key=model)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Erro na inferência: {str(e)}')


@app.get('/sample/{class_name}',
         summary='Gerar imagem sintética de amostra',
         response_class=Response)
def get_sample(class_name: str, size: int = 256):
    """
    Gera e retorna uma imagem sintética da classe especificada.

    Útil para testar a API sem ter uma imagem própria.

    - **class_name**: `satellite_operational` | `critical_fragment` | `small_debris` | `orbital_structure`
    - **size**: tamanho da imagem em pixels (padrão: 256, máx: 512)
    """
    if class_name not in CLASS_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Classe '{class_name}' não existe. Classes válidas: {CLASS_NAMES}"
        )
    size = min(max(64, size), 512)

    try:
        from dataset_builder.build_dataset import SyntheticGenerator
        gen = SyntheticGenerator(size=size)
        method_map = {
            'satellite_operational': gen.satellite,
            'critical_fragment':     gen.fragment,
            'small_debris':          gen.small_debris,
            'orbital_structure':     gen.structure,
        }
        img = method_map[class_name]()
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=92)
        return Response(content=buf.getvalue(), media_type='image/jpeg')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Erro ao gerar imagem: {str(e)}')


# Execução 
if __name__ == '__main__':
    import uvicorn
    print('\n' + '='*55)
    print('  🛰️  API de Classificação de Detritos Orbitais')
    print('='*55)
    print(f'  Dispositivo : {DEVICE}')
    print(f'  CNN 1       : {"✅" if MODEL1_LOADED else "⚠️  pesos aleatórios"}')
    print(f'  CNN 2       : {"✅" if MODEL2_LOADED else "⚠️  pesos aleatórios"}')
    print('='*55)
    print('  Documentação: http://localhost:8000/docs')
    print('  ReDoc       : http://localhost:8000/redoc')
    print('  Health      : http://localhost:8000/health')
    print('='*55 + '\n')
    uvicorn.run('demo.api_fastapi:app', host='0.0.0.0', port=8000, reload=True)
