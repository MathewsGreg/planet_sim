"""
Combine port_template.html + port_data.json into the final, self-contained
port viz. Run export_web_data.py first to (re)generate the data.

Writes two copies:
  - port.html                     working copy, gitignored (regenerable)
  - docs/container_port/index.html  tracked -- served by GitHub Pages at
                                     https://<user>.github.io/planet_sim/container_port/
"""
from __future__ import annotations

import os

PLACEHOLDER = "/*__PORT_DATA__*/"

if __name__ == "__main__":
    with open("port_data.json", "r", encoding="utf-8") as f:
        data = f.read()
    with open("port_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    if PLACEHOLDER not in template:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in port_template.html")

    combined = template.replace(PLACEHOLDER, data)

    with open("port.html", "w", encoding="utf-8") as f:
        f.write(combined)

    out_dir = os.path.join("..", "docs", "container_port")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(combined)

    size_mb = os.path.getsize("port.html") / 1024 / 1024
    print(f"wrote port.html and {out_path} ({size_mb:.2f} MB each)")
