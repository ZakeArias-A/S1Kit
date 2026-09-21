"""Render six paired image fixtures. No model output is used to create them."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
INK = "#172b4d"
MUTED = "#53657d"
BLUE = "#3265ce"
GREEN = "#13855e"
RED = "#c33e40"


def canvas(title, subtitle):
    image = Image.new("RGB", (640, 360), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 16, 624, 344), radius=16, fill="white")
    label(draw, (38, 34), title, 27)
    label(draw, (38, 73), subtitle, 16, MUTED)
    return image, draw


def label(draw, xy, value, size=20, color=INK):
    draw.text(xy, value, font=ImageFont.load_default(size=size), fill=color)


def invoice(paid):
    image, draw = canvas("INVOICE 1042", "Northwind Supplies  /  21 September 2026")
    for y, name, value in [(120, "Packing materials", "$120.00"), (164, "Shipping", "$48.00"),
                            (222, "TOTAL", "$168.00"), (263, "Payment received", "$168.00" if paid else "$0.00")]:
        label(draw, (40, y), name)
        label(draw, (475, y), value)
    draw.line((40, 205, 599, 205), fill="#dae1ed", width=2)
    draw.rounded_rectangle((38, 298, 601, 334), radius=6, fill="#edf2fa")
    label(draw, (49, 305), "BALANCE DUE", 18)
    label(draw, (475, 305), "$0.00" if paid else "$168.00", 18)
    return image


def inventory(values):
    image, draw = canvas("WAREHOUSE / STOCK", "Available units by part. Dashed line: reorder below 25.")
    left, top, bottom, scale = 88, 118, 288, 1.9
    for units in (0, 25, 50, 75):
        y = bottom - units * scale
        draw.line((left, y, 595, y), fill="#e2e7f0")
        label(draw, (45, y - 8), str(units), 15, MUTED)
    for x, name, value in zip((145, 315, 485), ("Part A", "Part B", "Part C"), values, strict=True):
        y = bottom - value * scale
        draw.rectangle((x - 33, y, x + 33, bottom), fill=BLUE)
        label(draw, (x - 12, y - 26), str(value), 19)
        label(draw, (x - 30, 303), name, 18)
    y = bottom - 25 * scale
    for x in range(left, 594, 12):
        draw.line((x, y, x + 6, y), fill=RED, width=2)
    return image


def services(outage):
    image, draw = canvas("SERVICE HEALTH", "Live status  /  14:30 UTC")
    for y, name, status in [(123, "API", "DOWN" if outage else "HEALTHY"),
                             (191, "Database", "HEALTHY"), (259, "Payments", "HEALTHY")]:
        draw.rounded_rectangle((36, y - 6, 604, y + 43), radius=8, fill="#f3f6fb")
        label(draw, (53, y + 6), name, 22)
        color = RED if status == "DOWN" else GREEN
        draw.ellipse((421, y + 12, 436, y + 27), fill=color)
        label(draw, (449, y + 8), status, 19, color)
    return image


def main():
    specs = [
        ("invoice", "Determine payment status from the attached invoice.",
         {"type": "noul", "instructions": "Has this invoice been paid in full? A zero balance due means paid in full."},
         [(invoice(True), "yes"), (invoice(False), "no")]),
        ("inventory", "Choose a replenishment action using the attached inventory chart.",
         {"type": "choice", "instructions": "Which part has stock below the reorder threshold of 25 units?",
          "criteria": {"a": "Reorder Part A", "b": "Reorder Part B", "c": "Reorder Part C"}},
         [(inventory([12, 48, 70]), "a"), (inventory([65, 15, 48]), "b")]),
        ("services", "Assess the attached service status dashboard.",
         {"type": "score", "instructions": "Rate the current incident severity using the displayed service statuses.",
          "criteria": ["All services healthy", "At least one degraded service, but none down", "At least one service down"]},
         [(services(False), "0"), (services(True), "2")]),
    ]
    cases = []
    preview = Image.new("RGB", (1280, 1080), "white")
    for row, (group, state, question, variants) in enumerate(specs):
        for column, (image, expected) in enumerate(variants):
            name = f"{group}-{column + 1}"
            image.save(ROOT / f"{name}.png")
            preview.paste(image, (column * 640, row * 360))
            cases.append({"id": name, "group": group, "images": [f"{name}.png"], "expected": expected,
                          "request": {"state": state, "questions": {"decision": question}}})
    preview.save(ROOT / "preview.png")
    (ROOT / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
