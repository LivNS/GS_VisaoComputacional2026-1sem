import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from models.architectures import (
    CNN1_Baseline, CNN2_Regularized, CLASS_NAMES, IMG_SIZE, NUM_CLASSES
)

# Constantes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Metadados indexados por nome de classe (funciona em qualquer ordem)
CLASS_LABELS = {
    'satellite_operational': 'Satélite Operacional',
    'critical_fragment':     'Fragmento Crítico',
    'small_debris':          'Pequeno Detrito',
    'orbital_structure':     'Estrutura Orbital',
}
CLASS_DESCRIPTIONS = {
    'satellite_operational': 'Satélite em plena operação. Baixo risco de colisão.',
    'critical_fragment':     'Fragmento grande (>10 cm), risco elevado. Rastreamento prioritário.',
    'small_debris':          'Partícula pequena (<10 cm). Difícil rastreamento, risco acumulativo.',
    'orbital_structure':     'Estrutura inativa (estágio de foguete, módulo). Órbita controlada.',
}
RISK_LEVEL = {
    'satellite_operational': 'Baixo',
    'critical_fragment':     'Alto',
    'small_debris':          'Médio',
    'orbital_structure':     'Moderado',
}
# Rótulos curtos seguindo a ORDEM REAL de CLASS_NAMES
SHORT_BY_NAME = {
    'satellite_operational': 'Sat. Op.',
    'critical_fragment':     'Frag. Crít.',
    'small_debris':          'Peq. Det.',
    'orbital_structure':     'Est. Orb.',
}
COLOR_BY_NAME = {
    'satellite_operational': '#2196F3',
    'critical_fragment':     '#FF5722',
    'small_debris':          '#9C27B0',
    'orbital_structure':     '#4CAF50',
}

MEAN = torch.tensor([0.05, 0.05, 0.07]).view(3, 1, 1)
STD  = torch.tensor([0.12, 0.12, 0.14]).view(3, 1, 1)

# ─── Carregamento dos modelos ─────────────────────────────────────────────────

def load_models():
    m1 = CNN1_Baseline(num_classes=NUM_CLASSES).to(DEVICE)
    m2 = CNN2_Regularized(num_classes=NUM_CLASSES).to(DEVICE)

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    path1 = os.path.join(model_dir, 'cnn1_baseline_best.pth')
    path2 = os.path.join(model_dir, 'cnn2_regularized_best.pth')

    if os.path.exists(path1):
        m1.load_state_dict(torch.load(path1, map_location=DEVICE))
        print("CNN 1 carregada de arquivo.")
    else:
        print("Pesos CNN 1 nao encontrados. Usando pesos aleatorios.")

    if os.path.exists(path2):
        m2.load_state_dict(torch.load(path2, map_location=DEVICE))
        print("CNN 2 carregada de arquivo.")
    else:
        print("Pesos CNN 2 nao encontrados. Usando pesos aleatorios.")

    m1.eval()
    m2.eval()
    return m1, m2

MODEL1, MODEL2 = load_models()

# ─── Pré-processamento ────────────────────────────────────────────────────────

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    t = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
    t = (t - MEAN) / STD
    return t.unsqueeze(0).to(DEVICE)

# ─── Gerador de amostras sintéticas ──────────────────────────────────────────

def generate_sample(class_choice: str) -> Image.Image:
    size = IMG_SIZE
    img = Image.new('RGB', (size, size), color=(random.randint(0, 8), random.randint(0, 8), 10))
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(8, 20)):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        b = random.randint(180, 255)
        draw.point((x, y), fill=(b, b, b))

    cx, cy = size // 2, size // 2

    if class_choice == 'satellite_operational':
        m = (180, 180, 200); p = (20, 40, 120)
        draw.rectangle([cx-22, cy-14, cx+22, cy+14], fill=m, outline=(220,220,220))
        draw.rectangle([cx-52, cy-8, cx-22, cy+8], fill=p, outline=(80,80,180))
        draw.rectangle([cx+22, cy-8, cx+52, cy+8], fill=p, outline=(80,80,180))
        draw.line([cx, cy-14, cx, cy-24], fill=(220,220,220), width=1)
        draw.ellipse([cx-3, cy-28, cx+3, cy-22], fill=(200,200,220))
    elif class_choice == 'critical_fragment':
        pts = [(cx + int(25*np.cos(2*np.pi*i/7 + r*0.4)*random.uniform(0.5,1.0)),
                cy + int(25*np.sin(2*np.pi*i/7 + r*0.4)*random.uniform(0.5,1.0)))
               for i, r in enumerate(np.random.rand(7))]
        draw.polygon(pts, fill=(200,200,210), outline=(240,240,245))
        draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255,255,240))
    elif class_choice == 'small_debris':
        b = random.randint(160, 220); s = random.randint(3, 5)
        draw.ellipse([cx-s, cy-s, cx+s, cy+s], fill=(b,b,b))
        for hr in [s+2, s+4, s+6]:
            a = max(30, b - hr*18)
            draw.ellipse([cx-hr, cy-hr, cx+hr, cy+hr], fill=None, outline=(a,a,a))
        angle = random.uniform(0, 2*np.pi)
        for t in range(12):
            tx = int(cx - t*np.cos(angle)*1.2); ty = int(cy - t*np.sin(angle)*1.2)
            a = max(10, b - t*15)
            draw.ellipse([tx-1, ty-1, tx+1, ty+1], fill=(a,a,a))
    else:  # orbital_structure
        m = (160, 155, 150)
        a = random.uniform(0, np.pi)
        ln, wd = 50, 10
        corners = [
            (cx + (-ln)*np.cos(a) - (-wd)*np.sin(a), cy + (-ln)*np.sin(a) + (-wd)*np.cos(a)),
            (cx + ln*np.cos(a) - (-wd)*np.sin(a),  cy + ln*np.sin(a) + (-wd)*np.cos(a)),
            (cx + ln*np.cos(a) - wd*np.sin(a),     cy + ln*np.sin(a) + wd*np.cos(a)),
            (cx + (-ln)*np.cos(a) - wd*np.sin(a),  cy + (-ln)*np.sin(a) + wd*np.cos(a)),
        ]
        draw.polygon([(int(x), int(y)) for x,y in corners], fill=m, outline=(200,195,185))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img.resize((256, 256), Image.NEAREST)

# ─── Predição ─────────────────────────────────────────────────────────────────

def classify_image(pil_img, model_choice: str):
    import matplotlib.pyplot as plt

    if pil_img is None:
        return None, "Por favor, envie ou gere uma imagem.", None

    try:
        # Garantir que e uma imagem PIL valida em RGB
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(np.array(pil_img))
        pil_img = pil_img.convert('RGB')

        model = MODEL2 if 'CNN 2' in model_choice else MODEL1

        with torch.no_grad():
            tensor = preprocess(pil_img)
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Garantir array 1D com NUM_CLASSES elementos
        probs = np.atleast_1d(probs).flatten()

        pred_idx   = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        result_text = (
            f"## {CLASS_LABELS[pred_class]}\n\n"
            f"**Confiança:** {confidence:.1%}\n\n"
            f"**Risco orbital:** {RISK_LEVEL[pred_class]}\n\n"
            f"**Descrição:** {CLASS_DESCRIPTIONS[pred_class]}"
        )

        # Grafico de probabilidades — barras na ordem REAL de CLASS_NAMES
        short_names = [SHORT_BY_NAME[c] for c in CLASS_NAMES]
        bar_colors  = [COLOR_BY_NAME[c] for c in CLASS_NAMES]

        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.barh(short_names, probs * 100, color=bar_colors, edgecolor='white', linewidth=0.5)
        bars[pred_idx].set_edgecolor('#FFD700')
        bars[pred_idx].set_linewidth(2.5)
        ax.set_xlabel('Probabilidade (%)')
        ax.set_title('Distribuição de Confiança', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 110)
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1%}', va='center', fontsize=9)
        fig.tight_layout()

        return pil_img, result_text, fig

    except Exception as e:
        # Nunca quebra a interface — mostra o erro como texto
        import traceback
        err = f"## Erro ao classificar\n\n```\n{str(e)}\n```\n\nDetalhes:\n```\n{traceback.format_exc()}\n```"
        return pil_img, err, None

def generate_and_classify(class_choice: str, model_choice: str):
    img = generate_sample(class_choice)
    return classify_image(img, model_choice)

# ─── Interface Gradio ─────────────────────────────────────────────────────────

with gr.Blocks(title="Classificador de Detritos Orbitais", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style="text-align:center; padding:1rem 0;">
        <h1>🛰️ Classificador de Detritos Orbitais</h1>
        <p style="color:#666;">Visão Computacional para monitoramento de lixo espacial — Global Solution · Indústria Espacial</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Entrada")
            model_choice = gr.Dropdown(
                choices=['CNN 2 — Regularizada (Recomendado)', 'CNN 1 — Baseline'],
                value='CNN 2 — Regularizada (Recomendado)',
                label="Modelo CNN"
            )
            with gr.Tab("Upload de Imagem"):
                upload_img = gr.Image(type='pil', label="Imagem espacial", height=256)
                classify_btn = gr.Button("Classificar Imagem", variant='primary')
            with gr.Tab("Gerar Amostra Sintética"):
                gr.Markdown("*Gera uma imagem sintética para demonstração.*")
                class_selector = gr.Dropdown(choices=list(CLASS_NAMES),
                                             value=CLASS_NAMES[0], label="Classe para gerar")
                gen_btn = gr.Button("Gerar e Classificar", variant='secondary')
                gen_preview = gr.Image(label="Imagem gerada", height=200)

        with gr.Column(scale=1):
            gr.Markdown("### Resultado")
            result_text = gr.Markdown(value="*Aguardando classificação...*")
            prob_chart   = gr.Plot(label="Distribuição de Probabilidades")

    classify_btn.click(classify_image, inputs=[upload_img, model_choice],
                       outputs=[upload_img, result_text, prob_chart])
    gen_btn.click(generate_and_classify, inputs=[class_selector, model_choice],
                  outputs=[gen_preview, result_text, prob_chart])

if __name__ == '__main__':
    demo.launch(
        server_name='127.0.0.1',
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )