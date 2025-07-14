import os
from PIL import Image, ImageDraw, ImageFont

# ==== CONFIGURATION ====
BASE_DIR  = "generation_output_CFG/result_norm"
STEPS     = ["skip_steps_2","skip_steps_3","skip_steps_4"]
BORDER_W  = 3                # px white border on skip_1.00
TEXT_PAD  = 10               # px space between image bottom & text top

# Point to any .ttf on your machine + size you like
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE = 64
FONT      = ImageFont.truetype(FONT_PATH, FONT_SIZE)
# =======================

def val_from_fname(fname):
    """Extract float from 'skip_0.60.png' → 0.60"""
    return float(os.path.splitext(fname)[0].split("_")[1])

# Pre-make a tiny canvas just for text measurements
_dummy_img  = Image.new("RGB", (1, 1))
_dummy_draw = ImageDraw.Draw(_dummy_img)

# 1) Compute full text line height via font metrics
ascent, descent = FONT.getmetrics()
text_h = ascent + descent   # total pixel height of any line

for category in os.listdir(BASE_DIR):
    out_root = os.path.join(BASE_DIR, category, "output")
    if not os.path.isdir(out_root):
        continue

    for out_name in os.listdir(out_root):
        folder = os.path.join(out_root, out_name)
        if not os.path.isdir(folder):
            continue

        # skip if missing any of the three subfolders
        if not all(os.path.isdir(os.path.join(folder, s)) for s in STEPS):
            continue

        # --- 2) Load & sort all images into 3 rows ---
        grid = []
        for step in STEPS:
            d = os.path.join(folder, step)
            pngs = [f for f in os.listdir(d) if f.endswith(".png")]
            pngs.sort(key=val_from_fname)
            imgs = [Image.open(os.path.join(d, f)) for f in pngs]
            grid.append((pngs, imgs))

        # assume uniform cell size
        cell_w, cell_h = grid[0][1][0].size
        cols = len(grid[0][1])

        # --- 3) Compute a single row height (image + pad + text) ---
        row_h = cell_h

        # --- 4) Make a canvas big enough for 3 rows of that height ---
        canvas_w = cols * cell_w
        canvas_h = 3 * row_h + TEXT_PAD + text_h
        canvas   = Image.new("RGB", (canvas_w, canvas_h), "black")
        draw     = ImageDraw.Draw(canvas)

        # --- 5) Paste, border, and label each cell ---
        for row_idx, (fnames, imgs) in enumerate(grid):
            y0 = row_idx * row_h
            for col_idx, img in enumerate(imgs):
                x0 = col_idx * cell_w
                canvas.paste(img, (x0, y0))

                # if value==1.00, draw white border
                v = val_from_fname(fnames[col_idx])
                if abs(v - 1.0) < 1e-6:
                    draw.rectangle(
                        [x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                        outline="white", width=BORDER_W
                    )

                if row_idx == len(grid) - 1:
                    # measure this label via textbbox and center it
                    label = f"{v:.2f}"
                    bbox  = _dummy_draw.textbbox((0, 0), label, font=FONT)
                    text_w = bbox[2] - bbox[0]

                    tx = x0 + (cell_w - text_w)//2
                    ty = y0 + cell_h + TEXT_PAD
                    draw.text((tx, ty), label, fill="white", font=FONT)

        # --- 6) Save the composite right next to its source folder ---
        out_png = os.path.join(folder, f"grid_{category}_{out_name}.png")
        canvas.save(out_png)
        print("Saved:", out_png)