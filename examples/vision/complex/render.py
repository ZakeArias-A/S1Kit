"""Create browser and policy fixtures for a multi-image returns decision.

item.jpg is an unchanged user-provided photograph and is not generated here.
"""

import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
INK = "#172b4d"
MUTED = "#53657d"


def text(draw, xy, content, size=24, color=INK):
    draw.text(xy, content, font=ImageFont.load_default(size=size), fill=color)


def browser(stock, color="GREEN", delivered="2026-09-09"):
    image = Image.new("RGB", (1040, 680), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1040, 47), fill="#dae1eb")
    for x, indicator_color in [(22, "#ea655c"), (44, "#e9ba4d"), (66, "#5abd83")]:
        draw.ellipse((x, 18, x + 11, 29), fill=indicator_color)
    draw.rounded_rectangle((114, 8, 1006, 39), radius=7, fill="white")
    text(draw, (134, 14), "https://support.example/orders/ORD-1042", 18, MUTED)
    draw.rectangle((0, 48, 202, 680), fill="#1c2f50")
    text(draw, (22, 81), "NORTHWIND", 23, "white")
    for y, name in [(154, "Overview"), (207, "Orders"), (260, "Inventory"), (313, "Returns")]:
        text(draw, (24, y), name, 22, "#d3ddef")
    text(draw, (237, 77), "Return request / ORD-1042", 32)
    text(draw, (239, 124), "Paid  |  Customer verified  |  Requested 2026-09-21", 21, MUTED)
    draw.rounded_rectangle((230, 173, 1010, 461), radius=12, fill="white")
    text(draw, (252, 193), "ORDERED ITEM", 19, MUTED)
    text(draw, (252, 232), "Desk creature figurine", 30)
    sku = "FIG-GRN-01" if color == "GREEN" else "FIG-YLW-01"
    text(draw, (252, 280), f"Ordered color: {color}", 28)
    for y, line in [(329, f"SKU: {sku}    Quantity: 1    Total: $29.00"),
                     (372, f"Delivered: {delivered}"), (414, "Request reason: item does not match ordered color")]:
        text(draw, (252, y), line, 22)
    draw.rounded_rectangle((230, 482, 1010, 634), radius=12, fill="white")
    text(draw, (252, 501), "LIVE REPLACEMENT STOCK", 19, MUTED)
    text(draw, (252, 538), f"{sku} ({color}): {stock} units", 28)
    text(draw, (252, 585), "FIG-BLU-01 (BLUE): 18 units    /    Other variant", 21, MUTED)
    text(draw, (238, 650), "Simulation fixture. No real customer or order data.", 16, MUTED)
    return image


def policy():
    image = Image.new("RGB", (820, 1080), "white")
    draw = ImageDraw.Draw(image)
    text(draw, (54, 48), "NORTHWIND SUPPLIES", 20, MUTED)
    text(draw, (54, 91), "Color mismatch returns", 35)
    text(draw, (54, 144), "Policy v3  /  Effective 2026-09-01", 23)
    draw.line((54, 192, 766, 192), fill="#bcc9df", width=2)
    paragraphs = [
        ("1. Evidence", "Use the ordered color on the order page and the visible body color in the customer's photo. If the customer photo is missing or color cannot be established, request a clear photo before deciding."),
        ("2. Eligibility", "For a verified, paid order, a confirmed color mismatch is eligible when the request is at most 14 calendar days after delivery. Outside this window, send to manual review."),
        ("3. Remedy", "For an eligible mismatch, replace with the ordered SKU if its stock is greater than zero. If that SKU has zero stock, refund. Stock of another color cannot be substituted."),
        ("4. Match", "If the visible body color matches the ordered color, reject a color-mismatch claim. Strings, packaging, and shadows do not by themselves establish product damage."),
    ]
    y = 219
    for heading, paragraph in paragraphs:
        text(draw, (54, y), heading, 27)
        y += 43
        for line in textwrap.wrap(paragraph, width=58):
            text(draw, (54, y), line, 23)
            y += 30
        y += 26
    draw.rectangle((42, 955, 778, 1041), fill="#eff2f6")
    text(draw, (55, 970), "ARCHIVED v2 - superseded, not current policy", 22, MUTED)
    text(draw, (55, 1006), "Previously: all color claims went to manual review.", 21, MUTED)
    return image


def main():
    for index, stock in enumerate((0, 8), 1):
        browser(stock).save(ROOT / f"order-{index}.png")
    browser(8, color="YELLOW").save(ROOT / "order-3.png")
    browser(8, delivered="2026-09-01").save(ROOT / "order-4.png")
    policy().save(ROOT / "policy.png")
    request = {
        "state": "Review this return case using the attached order page, policy document and customer photo, if present. Apply the current policy to the evidence.",
        "questions": {"decision": {
            "type": "choice", "instructions": "What is the next action for this return request?",
            "criteria": {
                "refund": "Approve a refund for the color mismatch",
                "replace": "Send a replacement of the ordered SKU",
                "request_photo": "Request a clear customer photo before deciding",
                "manual_review": "Send the case for manual review",
                "reject": "Reject the color-mismatch claim",
            },
        }},
    }
    cases = [
        {"id": "return-no-stock", "group": "returns", "images": ["order-1.png", "policy.png", "item.jpg"], "expected": "refund", "request": request},
        {"id": "return-in-stock", "group": "returns", "images": ["order-2.png", "policy.png", "item.jpg"], "expected": "replace", "request": request},
        {"id": "return-color-matches", "group": "returns", "images": ["order-3.png", "policy.png", "item.jpg"], "expected": "reject", "request": request},
        {"id": "return-outside-window", "group": "returns", "images": ["order-4.png", "policy.png", "item.jpg"], "expected": "manual_review", "request": request},
        {"id": "return-no-photo", "group": "returns", "images": ["order-1.png", "policy.png"], "expected": "request_photo", "request": request},
    ]
    (ROOT / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
