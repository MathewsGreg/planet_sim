"""
Combine globe_template.html + web_data.json into the final, self-contained
globe.html artifact. Run export_web_data.py first to (re)generate the data.

Writes two copies:
  - globe.html       working copy, gitignored (regenerable, like web_data.json)
  - docs/index.html  tracked -- this is what GitHub Pages serves at
                      https://<user>.github.io/planet_sim/, so a re-run
                      followed by `git add docs/index.html && git commit`
                      is the whole "publish an update" workflow.
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

    combined = template.replace(PLACEHOLDER, data)

    with open("globe.html", "w", encoding="utf-8") as f:
        f.write(combined)

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(combined)

    size_mb = os.path.getsize("globe.html") / 1024 / 1024
    print(f"wrote globe.html and docs/index.html ({size_mb:.2f} MB each)")
