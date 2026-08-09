"""
Combine globe_template.html + web_data.json into the final, self-contained
globe.html artifact. Run export_web_data.py first to (re)generate the data.
"""
from __future__ import annotations

import os

PLACEHOLDER = "/*__PLANET_DATA__*/"

if __name__ == "__main__":
    with open("web_data.json", "r", encoding="utf-8") as f:
        data = f.read()
    with open("globe_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    if PLACEHOLDER not in template:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in globe_template.html")

    with open("globe.html", "w", encoding="utf-8") as f:
        f.write(template.replace(PLACEHOLDER, data))

    size_mb = os.path.getsize("globe.html") / 1024 / 1024
    print(f"wrote globe.html ({size_mb:.2f} MB)")
