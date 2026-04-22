#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageColor, ImageDraw, ImageFont

if hasattr(Image, "Resampling"):
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
else:
    RESAMPLE_BICUBIC = Image.BICUBIC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a paper-style NIAH accuracy heatmap from a niah.py JSON result file."
    )
    parser.add_argument("--input", required=True, help="Path to niah JSON output")
    parser.add_argument("--output", required=True, help="Path to output PNG")
    parser.add_argument("--title", default="", help="Optional title override, e.g. q3_0_head")
    parser.add_argument("--score-label", default="Score", help="Prefix for the top score text")
    return parser.parse_args()


def load_summary(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", [])
    if not summary:
        raise ValueError(f"no summary rows found in {path}")
    return data, summary


def format_ctx_label(ctx_len: int) -> str:
    if ctx_len % 1024 == 0:
        return f"{ctx_len // 1024}k"
    return f"{ctx_len / 1024:.0f}k"


def format_depth_label(depth: float) -> str:
    return str(int(round(depth * 100)))


def find_font_path(name: str):
    try:
        result = subprocess.run(
            ["fc-match", name, "-f", "%{file}\n"],
            check=True,
            capture_output=True,
            text=True,
        )
        path = result.stdout.strip()
        if path:
            return path
    except Exception:
        return None
    return None


SERIF_FONT_PATH = find_font_path("DejaVu Serif")
SANS_FONT_PATH = find_font_path("DejaVu Sans")


def load_font(size: int, serif: bool = False):
    path = SERIF_FONT_PATH if serif else SANS_FONT_PATH
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(c0, c1, t: float):
    return tuple(int(round(lerp(c0[i], c1[i], t))) for i in range(3))


def score_to_color(score: float):
    score = max(0.0, min(1.0, score))
    red = ImageColor.getrgb("#e65a4f")
    yellow = ImageColor.getrgb("#ffd27a")
    green = ImageColor.getrgb("#6abd6e")
    if score <= 0.5:
        return lerp_color(red, yellow, score / 0.5 if score > 0 else 0.0)
    return lerp_color(yellow, green, (score - 0.5) / 0.5)


def measure_text(draw, text, font):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(font, "getsize"):
        return font.getsize(text)

    mask = font.getmask(text)
    return mask.size


def draw_centered_text(draw, box, text, font, fill):
    left, top, right, bottom = box
    w, h = measure_text(draw, text, font)
    x = left + (right - left - w) / 2
    y = top + (bottom - top - h) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_rotated_text(base_image, position, text, font, fill, angle):
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    dummy_draw = ImageDraw.Draw(dummy)
    w, h = measure_text(dummy_draw, text, font)
    text_image = Image.new("RGBA", (w + 8, h + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4, 4), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True, resample=RESAMPLE_BICUBIC)
    base_image.alpha_composite(rotated, dest=position)


def render_heatmap(data, summary, out_path: Path, title_override: str, score_label: str):
    ctx_lens = sorted({int(row["ctx_len"]) for row in summary})
    depths = sorted({float(row["requested_depth"]) for row in summary})
    score_map = {
        (int(row["ctx_len"]), float(row["requested_depth"])): float(row["accuracy"])
        for row in summary
    }

    overall_score = sum(float(row["accuracy"]) for row in summary) / len(summary)

    title_font = load_font(42, serif=True)
    score_font = load_font(32, serif=True)
    axis_font = load_font(18, serif=False)
    tick_font = load_font(16, serif=False)
    colorbar_font = load_font(16, serif=False)

    cell_w = 78
    cell_h = 52
    grid_w = len(ctx_lens) * cell_w
    grid_h = len(depths) * cell_h

    left_margin = 112
    top_margin = 118
    right_margin = 126
    bottom_margin = 118

    width = left_margin + grid_w + right_margin
    height = top_margin + grid_h + bottom_margin

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_color = (25, 25, 25)
    label_color = (45, 45, 45)
    grid_line = (55, 55, 55)

    title_text = title_override.strip()
    if title_text:
        draw_centered_text(draw, (0, 18, width, 62), title_text, title_font, title_color)
        score_box = (0, 62, width, 106)
    else:
        score_box = (0, 24, width, 68)

    draw_centered_text(
        draw,
        score_box,
        f"{score_label}: {overall_score:.3f}",
        score_font,
        title_color,
    )

    grid_x0 = left_margin
    grid_y0 = top_margin
    grid_x1 = grid_x0 + grid_w
    grid_y1 = grid_y0 + grid_h

    # Cells
    for row_idx, depth in enumerate(depths):
        for col_idx, ctx_len in enumerate(ctx_lens):
            x0 = grid_x0 + col_idx * cell_w
            y0 = grid_y0 + row_idx * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            score = score_map[(ctx_len, depth)]
            draw.rectangle((x0, y0, x1, y1), fill=score_to_color(score), outline=grid_line)

    draw.rectangle((grid_x0, grid_y0, grid_x1, grid_y1), outline=grid_line, width=2)

    # Axis labels
    for row_idx, depth in enumerate(depths):
        y0 = grid_y0 + row_idx * cell_h
        y1 = y0 + cell_h
        draw_centered_text(draw, (28, y0, left_margin - 16, y1), format_depth_label(depth), tick_font, label_color)

    for col_idx, ctx_len in enumerate(ctx_lens):
        x0 = grid_x0 + col_idx * cell_w
        x1 = x0 + cell_w
        label = format_ctx_label(ctx_len)
        dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        dummy_draw = ImageDraw.Draw(dummy)
        w, h = measure_text(dummy_draw, label, tick_font)
        text_image = Image.new("RGBA", (w + 10, h + 10), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_image)
        text_draw.text((5, 5), label, font=tick_font, fill=label_color)
        rotated = text_image.rotate(35, expand=True, resample=RESAMPLE_BICUBIC)
        tx = int(x0 + cell_w * 0.15)
        ty = int(grid_y1 + 12)
        image.alpha_composite(rotated, dest=(tx, ty))

    draw_rotated_text(image, (14, grid_y0 + grid_h // 3), "Depth Percent", axis_font, label_color, 90)
    draw_centered_text(draw, (grid_x0, grid_y1 + 68, grid_x1, height - 16), "Token Limit", axis_font, label_color)

    # Color bar
    bar_x0 = grid_x1 + 38
    bar_x1 = bar_x0 + 22
    bar_y0 = grid_y0
    bar_y1 = grid_y1
    for i in range(bar_y1 - bar_y0):
        score = 1.0 - i / max(1, (bar_y1 - bar_y0 - 1))
        draw.line((bar_x0, bar_y0 + i, bar_x1, bar_y0 + i), fill=score_to_color(score))
    draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y1), outline=grid_line)

    ticks = [(1.0, "1.00"), (0.75, "0.75"), (0.50, "0.50"), (0.25, "0.25"), (0.0, "0.00")]
    for value, label in ticks:
        y = bar_y0 + int(round((1.0 - value) * (bar_y1 - bar_y0)))
        draw.line((bar_x1 + 2, y, bar_x1 + 8, y), fill=label_color)
        draw.text((bar_x1 + 12, y - 9), label, font=colorbar_font, fill=label_color)

    draw_rotated_text(image, (bar_x0 + 30, bar_y0 + grid_h // 3), "Score", axis_font, label_color, 90)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    data, summary = load_summary(input_path)
    render_heatmap(data, summary, output_path, args.title, args.score_label)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
