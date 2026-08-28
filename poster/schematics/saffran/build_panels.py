"""Build poster Saffran panels from intro_front slide exports."""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("poster/schematics/saffran")


def trim_white(im: Image.Image, pad: int = 12, thresh: int = 245) -> Image.Image:
    arr = np.asarray(im.convert("RGB"))
    mask = (arr < thresh).any(axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return im
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(arr.shape[0], int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(arr.shape[1], int(xs.max()) + pad)
    return im.crop((x0, y0, x1, y1))


def main() -> None:
    # Already-clean spectrogram panel.
    sil = Image.open(OUT / "image3.png").convert("RGB")
    sil.save(OUT / "no_silences.png")

    slide = Image.open(OUT / "slide7.png").convert("RGB")
    # Continuous stream line + transitional-probability arcs (skip bullet text).
    stream = trim_white(slide.crop((120, 318, 1480, 358)), pad=6)
    arcs = trim_white(slide.crop((80, 605, 1520, 840)), pad=8)

    # Stack stream over arcs with a thin white gap.
    gap = 14
    w = max(stream.width, arcs.width)
    h = stream.height + gap + arcs.height
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(stream, ((w - stream.width) // 2, 0))
    canvas.paste(arcs, ((w - arcs.width) // 2, stream.height + gap))
    canvas = trim_white(canvas, pad=10)
    # Slide text sometimes leaves a stray "words" under the stream; blank it.
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [int(canvas.width * 0.08), int(canvas.height * 0.10),
         int(canvas.width * 0.22), int(canvas.height * 0.26)],
        fill=(255, 255, 255),
    )
    canvas.save(OUT / "saffran_stream.png")
    print("wrote", OUT / "no_silences.png", sil.size)
    print("wrote", OUT / "saffran_stream.png", canvas.size)


if __name__ == "__main__":
    main()
