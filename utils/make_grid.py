import sys, re
from pathlib import Path
from collections import OrderedDict
from PIL import Image, ImageDraw

IMG_SUFFIX = ".png"

PROMPT_RE        = re.compile(r"output_(\d+)")                   # → “N”
SKIP_STEPS_RE    = re.compile(r"skip_steps_(\d+)")               # → “M”
COEF_RE          = re.compile(r"skip_([-+]?\d*\.?\d+|\d+)\.png") # → “coef” as str or float

# ---------- helpers -----------------------------------------------------------
def natural_sort_key(s):
    " human/natural sort for strings that contain numbers "
    return [int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", str(s))]

def load_and_resize(path, size):
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img

def add_border(img, border=4, outline="red"):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(border):
        draw.rectangle((i, i, w - 1 - i, h - 1 - i), outline=outline)
    return img

def paste_text(canvas, text, xy, font=None, fill="white", anchor="mm"):
    from PIL import ImageFont
    font = ImageFont.load_default(size=40)
    # PIL shipped w/o fonts – draw simple mono-space text via default bitmap font
    draw = ImageDraw.Draw(canvas)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

# ---------- core grid builder --------------------------------------------------
def build_grid(folder: Path):
    """
    folder = …/prompt_name/output_N
    Builds its grid and stores it as grid.png
    """
    prompt_root = folder.parent          # …/prompt_name
    orig_path   = prompt_root / "output" / "orig.png"

    if not orig_path.exists():
        print("❌  Missing", orig_path)
        return

    # ---- collect all rows (M) and all coef values ----------------------------
    rows_dict           = OrderedDict()   # M  -> {coef : img_path}
    all_coef_set        = set()
    smal_img_path       = folder / "smal.png"

    for skip_dir in sorted(folder.glob("skip_steps_*"), key=natural_sort_key):
        m_match = SKIP_STEPS_RE.fullmatch(skip_dir.name)
        if not m_match:
            continue
        M = int(m_match.group(1))
        rows_dict[M] = {}
        for coef_file in skip_dir.glob("skip_*" + IMG_SUFFIX):
            c_match = COEF_RE.fullmatch(coef_file.name)
            if c_match:
                coef = float(c_match.group(1))
                rows_dict[M][coef] = coef_file
                all_coef_set.add(coef)

    # add smal.png as coef == 2
    coef_smal = 2.0
    all_coef_set.add(coef_smal)
    for m in rows_dict:
        rows_dict[m][coef_smal] = smal_img_path

    if not rows_dict:
        print("⚠️  No skip_steps_* dirs in", folder)
        return

    # ---- fixed ordering ------------------------------------------------------
    Ms     = sorted(rows_dict.keys())                       # ascending M rows
    coefs  = sorted(all_coef_set)                           # ascending columns
    n_rows = len(Ms)
    n_cols = 1 + len(coefs)                                 # +1 for orig column

    # ---- load first image to know the size -----------------------------------
    #   any image inside will do:
    first_img_path = next(iter(rows_dict[Ms[0]].values()))
    im_w, im_h     = Image.open(first_img_path).size

    # prepare orig.png (with red border)
    orig_img   = add_border(load_and_resize(orig_path, (im_w, im_h)), outline="red")

    # ---- blank canvases ------------------------------------------------------
    bg_color   = (0, 0, 0)                     # black background – no white space
    grid_w     = im_w * n_cols
    grid_h     = im_h * n_rows
    canvas     = Image.new("RGB", (grid_w, grid_h), bg_color)

    # ---- paste all images ----------------------------------------------------
    for row_idx, M in enumerate(Ms):
        y0 = row_idx * im_h
        # orig → first column
        canvas.paste(orig_img, (0, y0))
        # the rest
        for col_idx, coef in enumerate(coefs):
            x0 = (1 + col_idx) * im_w
            img_path = rows_dict[M].get(coef)
            if not img_path or not img_path.exists():
                continue
            img = load_and_resize(img_path, (im_w, im_h))
            if coef == 2:
                img = add_border(img, outline="white")
            canvas.paste(img, (x0, y0))

    # ---- annotations (coef under columns, M to the left) ---------------------
    # draw coef labels
    for col_idx, coef in enumerate(coefs):
        x_center = (1 + col_idx) * im_w + im_w // 2
        
    # extend canvas downwards for coef text if necessary
    text_height = 64
    final_h     = grid_h + text_height
    final_canvas = Image.new("RGB", (grid_w, final_h), bg_color)
    final_canvas.paste(canvas, (0, 0))

    # coef labels
    for col_idx, coef in enumerate(coefs):
        x_center = (1 + col_idx) * im_w + im_w // 2
        paste_text(final_canvas, str(coef), (x_center, grid_h + text_height//2), font=40)

    # # M labels
    # for row_idx, M in enumerate(Ms):
    #     y_center = row_idx * im_h + im_h // 2
    #     paste_text(final_canvas, str(M), (im_w//2, y_center), font=40)  # left half of orig

    # ---- write file ----------------------------------------------------------
    out_path = folder / "grid.png"
    final_canvas.save(out_path)
    print("✅  Saved", out_path)

# ---------- discovery ---------------------------------------------------------
def walk_generation_tree(root: Path):
    if (root / "output").exists() and any(root.glob("output_*")):
        # called directly on a prompt dir
        prompt_dirs = [root]
    else:
        prompt_dirs = [p for p in root.glob("*") if (p / "output").exists()]

    for prompt_dir in prompt_dirs:
        for out_dir in sorted(prompt_dir.glob("output_*"), key=natural_sort_key):
            build_grid(out_dir)

# ---------- main --------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_grid.py /path/to/generation")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print("Given path doesn’t exist:", root)
        sys.exit(1)

    walk_generation_tree(root)