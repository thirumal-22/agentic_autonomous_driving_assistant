#!/usr/bin/env python3
"""Generate a 2-page PDF with architecture diagram and talking points."""
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DOCS = os.path.join(ROOT, 'docs')
OUT = os.path.join(DOCS, 'Presentation.pdf')

WIDTH, HEIGHT = 1200, 900
BG = (13, 14, 21)
ACCENT = (56, 189, 248)  # cyan
TEXT = (232, 238, 244)

# Load files
with open(os.path.join(DOCS, 'ARCHITECTURE.md'), 'r') as f:
    arch = f.read()
with open(os.path.join(DOCS, 'INTERVIEW_TALKING_POINTS.md'), 'r') as f:
    talk = f.read()

# Simple helper to draw wrapped text
def draw_wrapped(draw, text, x, y, font, max_width, fill):
    lines = []
    for paragraph in text.split('\n'):
        wrapped = textwrap.wrap(paragraph, width=80)
        if not wrapped:
            lines.append('')
        else:
            lines.extend(wrapped)
    def text_size(txt):
        bbox = draw.textbbox((0,0), txt, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]

    oy = y
    for line in lines:
        draw.text((x, oy), line, font=font, fill=fill)
        oy += text_size(line)[1] + 6
    return oy

# Fonts (system may not have fancy fonts; use default)
try:
    title_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 40)
    heading_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 28)
    body_font = ImageFont.truetype('DejaVuSans.ttf', 18)
except Exception:
    title_font = ImageFont.load_default()
    heading_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

# Slide 1: Title + architecture boxes
img1 = Image.new('RGB', (WIDTH, HEIGHT), BG)
d1 = ImageDraw.Draw(img1)

# Title
d1.text((WIDTH//2 - 360, 40), 'Agentic Autonomous Driving Assistant', font=title_font, fill=TEXT)
d1.text((WIDTH//2 - 240, 100), 'Architecture Overview', font=heading_font, fill=ACCENT)

# Draw simple box diagram
box_w, box_h = 220, 80
cx = WIDTH//2
start_y = 180
boxes = ['Perception', 'Environment', 'Decision', 'Reasoning', 'Safety']
positions = []
for i, name in enumerate(boxes):
    x = cx - (box_w//2)
    y = start_y + i*(box_h+30)
    positions.append((x,y))
    d1.rectangle([x, y, x+box_w, y+box_h], outline=ACCENT, width=3, fill=(20,24,32))
    bbox = d1.textbbox((0,0), name, font=body_font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    d1.text((x + (box_w-w)//2, y + (box_h-h)//2), name, font=body_font, fill=TEXT)

# Arrows between boxes
for i in range(len(positions)-1):
    x1 = positions[i][0] + box_w//2
    y1 = positions[i][1] + box_h
    x2 = positions[i+1][0] + box_w//2
    y2 = positions[i+1][1]
    d1.line([(x1,y1+5),(x2,y2-5)], fill=ACCENT, width=3)
    # arrow head
    d1.polygon([(x2-8, y2-7),(x2+8,y2-7),(x2,y2+5)], fill=ACCENT)

# Footer note
d1.text((60, HEIGHT-80), 'Generated from project docs — use for interviews', font=body_font, fill=(150,150,150))

# Slide 2: Talking points (first ~12 lines)
img2 = Image.new('RGB', (WIDTH, HEIGHT), BG)
d2 = ImageDraw.Draw(img2)

d2.text((60, 40), 'Interview Talking Points', font=title_font, fill=TEXT)

# Extract key bullets from talk
lines = []
for line in talk.split('\n'):
    line = line.strip()
    if line.startswith('##') and 'Elevator' in line:
        continue
    if line.startswith('##') or line.startswith('###'):
        lines.append(line.replace('#','').strip())
    elif line.startswith('-') or line.startswith('Q:') or line.startswith('A:') or line.startswith('1.'):
        lines.append(line)

# Take first ~12 bullets
bullets = lines[:12]
oy = 120
for b in bullets:
    wrapped = textwrap.wrap(b, width=90)
    for sub in wrapped:
        text = u'• ' + sub if not sub.startswith('Q:') else sub
        d2.text((80, oy), text, font=body_font, fill=TEXT)
        bbox = d2.textbbox((0,0), text, font=body_font)
        oy += (bbox[3] - bbox[1]) + 8
    oy += 4

# Save as multi-page PDF
imgs = [img1.convert('RGB'), img2.convert('RGB')]
imgs[0].save(OUT, save_all=True, append_images=imgs[1:])
print('Presentation saved to', OUT)
