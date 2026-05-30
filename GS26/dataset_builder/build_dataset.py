import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

DATASET_DIR = Path('dataset')
SEED        = 42
IMG_SIZE    = 128
N_PER_CLASS = 500

CLASSES = ['critical_fragment', 'orbital_structure', 'satellite_operational', 'small_debris']

random.seed(SEED)
np.random.seed(SEED)


class SyntheticGenerator:
    """Gera imagens sinteticas REALISTAS com dificuldade controlada."""

    def __init__(self, size: int = 128):
        self.size = size

    def _bg(self):
        base = random.randint(0, 30)
        img = Image.new('RGB', (self.size, self.size),
                        (base, base, base + random.randint(0, 8)))
        d = ImageDraw.Draw(img)
        for _ in range(random.randint(3, 35)):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            b = random.randint(120, 255)
            d.point((x, y), fill=(b, b, b))
        return img, d

    def _add_sensor_noise(self, img):
        arr = np.array(img).astype(np.float32)
        noise_level = random.uniform(8, 28)
        noise = np.random.normal(0, noise_level, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _post_process(self, img):
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.8)))
        img = self._add_sensor_noise(img)
        if random.random() > 0.5:
            arr = np.array(img).astype(np.float32)
            factor = random.uniform(0.6, 1.4)
            arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        return img

    def satellite(self):
        img, d = self._bg()
        cx, cy = self.size//2 + random.randint(-15, 15), self.size//2 + random.randint(-15, 15)
        bright = random.randint(90, 210)
        m = (bright, bright, min(255, bright + random.randint(0, 30)))
        p = random.choice([(20,40,120),(10,30,100),(40,40,60),(60,60,80)])
        bw, bh = random.randint(12, 26), random.randint(10, 20)
        d.rectangle([cx-bw, cy-bh, cx+bw, cy+bh], fill=m, outline=(180,180,180))
        n_panels = random.choice([0, 1, 2, 2])
        pw, ph = random.randint(15, 32), random.randint(6, 12)
        if n_panels >= 1:
            d.rectangle([cx-bw-pw, cy-ph, cx-bw, cy+ph], fill=p, outline=(80,80,140))
        if n_panels >= 2:
            d.rectangle([cx+bw, cy-ph, cx+bw+pw, cy+ph], fill=p, outline=(80,80,140))
        if random.random() > 0.4:
            d.line([cx, cy-bh, cx, cy-bh-random.randint(6,14)], fill=(200,200,200), width=1)
        return self._post_process(img)

    def fragment(self):
        img, d = self._bg()
        cx, cy = self.size//2+random.randint(-18,18), self.size//2+random.randint(-18,18)
        bright = random.randint(110, 230)
        m = (bright, bright, bright)
        n = random.randint(4, 9)
        sz = random.randint(6, 26)
        pts = [(cx+sz*random.uniform(0.4,1.0)*np.cos(2*np.pi*i/n+random.uniform(-0.4,0.4)),
                cy+sz*random.uniform(0.4,1.0)*np.sin(2*np.pi*i/n+random.uniform(-0.4,0.4)))
               for i in range(n)]
        d.polygon(pts, fill=m, outline=(200,200,200))
        if random.random() > 0.5:
            gx, gy = cx+random.randint(-5,5), cy+random.randint(-5,5)
            d.ellipse([gx-2,gy-2,gx+2,gy+2],fill=(255,255,240))
        return self._post_process(img)

    def small_debris(self):
        img, d = self._bg()
        cx = self.size//2+random.randint(-22,22); cy = self.size//2+random.randint(-22,22)
        b = random.randint(120, 235)
        sz = random.randint(2, 9)
        d.ellipse([cx-sz,cy-sz,cx+sz,cy+sz],fill=(b,b,b))
        for hr in [sz+2,sz+4,sz+6]:
            a = max(20,b-hr*18); d.ellipse([cx-hr,cy-hr,cx+hr,cy+hr],fill=None,outline=(a,a,a))
        if random.random() > 0.35:
            angle = random.uniform(0,2*np.pi)
            for t in range(random.randint(4,22)):
                tx=int(cx-t*np.cos(angle)*1.2); ty=int(cy-t*np.sin(angle)*1.2)
                a=max(10,b-t*12)
                if 0 <= tx < self.size and 0 <= ty < self.size:
                    d.ellipse([tx-1,ty-1,tx+1,ty+1],fill=(a,a,a))
        return self._post_process(img)

    def structure(self):
        img, d = self._bg()
        cx,cy = self.size//2+random.randint(-12,12), self.size//2+random.randint(-12,12)
        bright = random.randint(90, 200)
        m = (bright, bright-random.randint(0,15), bright-random.randint(0,20))
        accent = random.choice([(200,80,60),(60,80,200),(80,160,80),(120,120,120)])
        a = np.radians(random.uniform(0,180))
        ln, wd = random.randint(22,52), random.randint(7,18)
        corners = []
        for sx,sy in [(-ln,-wd),(ln,-wd),(ln,wd),(-ln,wd)]:
            corners.append((cx+sx*np.cos(a)-sy*np.sin(a), cy+sx*np.sin(a)+sy*np.cos(a)))
        d.polygon([(int(x),int(y)) for x,y in corners], fill=m, outline=(180,175,170))
        if random.random() > 0.4:
            for frac in [0.35, 0.65]:
                fx = cx+(ln*2*frac-ln)*np.cos(a); fy = cy+(ln*2*frac-ln)*np.sin(a)
                dx=wd*np.sin(a); dy=wd*np.cos(a)
                d.line([(fx-dx,fy+dy),(fx+dx,fy-dy)],fill=accent,width=2)
        return self._post_process(img)

    def generate_for(self, class_name):
        return {
            'satellite_operational': self.satellite,
            'critical_fragment':     self.fragment,
            'small_debris':          self.small_debris,
            'orbital_structure':     self.structure,
        }[class_name]()


def build(n_per_class=N_PER_CLASS):
    gen = SyntheticGenerator(IMG_SIZE)
    DATASET_DIR.mkdir(exist_ok=True)
    print(f'Gerando {n_per_class * len(CLASSES)} imagens ({n_per_class}/classe)...')
    for cls in CLASSES:
        cls_dir = DATASET_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for f in cls_dir.glob('*.jpg'):
            f.unlink()
        for i in range(n_per_class):
            img = gen.generate_for(cls).resize((IMG_SIZE, IMG_SIZE))
            img.save(cls_dir / f'{i:04d}.jpg', quality=85)
        print(f'  {cls}: {n_per_class} imagens')
    print(f'Dataset gerado em {DATASET_DIR}/')


if __name__ == '__main__':
    build()
