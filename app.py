"""Generate a lightweight MineralForge field report."""

from __future__ import annotations

from pathlib import Path

from mineralforge.edge_simulator import run_demo_assessment


def render_report(zone: str = "Stope 3", stress_multiplier: float = 2.8) -> str:
    assessment = run_demo_assessment(zone=zone, stress_multiplier=stress_multiplier)
    drivers = "\n".join(
        f"<li><strong>{driver['feature']}</strong>: {driver['value']:.3f}</li>"
        for driver in assessment["drivers"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MineralForge Field Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #17201a;
      background: #f3f6f1;
    }}
    main {{
      max-width: 920px;
      margin: 0 auto;
      padding: 40px 20px;
    }}
    header {{
      border-bottom: 3px solid #2d6a4f;
      padding-bottom: 18px;
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: 0;
    }}
    .risk {{
      display: inline-block;
      padding: 8px 12px;
      background: #b42318;
      color: white;
      font-weight: 700;
    }}
    section {{
      margin: 24px 0;
      padding: 20px;
      background: white;
      border-left: 4px solid #2d6a4f;
    }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>MineralForge Field Report</h1>
      <p>Zone: {assessment['zone']}</p>
      <p class="risk">Risk: {assessment['risk_level']} ({assessment['probability']:.1%})</p>
    </header>
    <section>
      <h2>Decision Drivers</h2>
      <ul>{drivers}</ul>
    </section>
    <section>
      <h2>TARP Action</h2>
      <p>{assessment['tarp']['action']}</p>
      <p>{assessment['tarp']['notification']}</p>
      <p>{assessment['tarp']['access_control']}</p>
    </section>
  </main>
</body>
</html>"""


def main() -> None:
    output = Path("mineralforge_report.html")
    output.write_text(render_report(), encoding="utf-8")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
