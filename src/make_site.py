"""Build the static GitHub Pages site (site/index.html) from metrics.json.

Copies figures and the compiled PDF into site/ so everything is self-contained
and served at https://maddoxk.github.io/quant-statarb-research/ with relative paths.
"""
import os
import json
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
os.makedirs(os.path.join(SITE, "figures"), exist_ok=True)

M = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
c = M["cointegration"]
b = M["backtest"]
p = M["params"]
s = M["sensitivity"]
y, x = M["tickers"]
real = M["data_source"] == "real"

# Copy assets with relative paths
for fn in ["prices.png", "spread_zscore.png", "equity_curve.png", "sensitivity_heatmap.png"]:
    shutil.copy(os.path.join(ROOT, "figures", fn), os.path.join(SITE, "figures", fn))
shutil.copy(os.path.join(ROOT, "paper.pdf"), os.path.join(SITE, "paper.pdf"))
# .nojekyll so GitHub Pages serves files as-is
open(os.path.join(SITE, ".nojekyll"), "w").write("")

data_badge = ("REAL market data (yfinance)" if real
              else "SYNTHETIC cointegrated data (fetch unavailable)")
badge_color = "#1a7f37" if real else "#9a6700"


def sens_rows():
    out = []
    for e, row in zip(s["entries"], s["sharpe_grid"]):
        cells = "".join(
            f'<td style="background:{heat(v)}">{v:.2f}</td>' for v in row
        )
        out.append(f"<tr><th>{e:g}</th>{cells}</tr>")
    return "\n".join(out)


def heat(v):
    # green for high Sharpe, red for low; clamp around [-0.3, 1.0]
    lo, hi = -0.3, 1.0
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    r = int(220 + (26 - 220) * t)
    g = int(80 + (160 - 80) * t)
    bl = int(80 + (60 - 80) * t)
    return f"rgb({r},{g},{bl})"


cost_header = "".join(f"<th>{cc:g} bps</th>" for cc in s["costs"])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cointegration Statistical Arbitrage &mdash; Maddox Krape</title>
<style>
  :root {{ --fg:#1f2328; --muted:#656d76; --bg:#ffffff; --card:#f6f8fa;
           --border:#d0d7de; --accent:#0969da; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",
          Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg);
          line-height:1.6; }}
  .wrap {{ max-width:920px; margin:0 auto; padding:2rem 1.25rem 4rem; }}
  header {{ border-bottom:1px solid var(--border); padding-bottom:1.5rem;
            margin-bottom:2rem; }}
  h1 {{ font-size:1.9rem; margin:0 0 .4rem; line-height:1.25; }}
  h2 {{ font-size:1.35rem; margin-top:2.5rem; border-bottom:1px solid var(--border);
        padding-bottom:.3rem; }}
  .sub {{ color:var(--muted); font-size:1.05rem; }}
  .badge {{ display:inline-block; padding:.2rem .6rem; border-radius:2rem;
            color:#fff; font-size:.8rem; font-weight:600; background:{badge_color}; }}
  .btns {{ margin:1.2rem 0; display:flex; gap:.6rem; flex-wrap:wrap; }}
  .btn {{ display:inline-block; padding:.55rem 1rem; border-radius:.5rem;
          text-decoration:none; font-weight:600; font-size:.95rem; }}
  .btn-primary {{ background:var(--accent); color:#fff; }}
  .btn-ghost {{ background:var(--card); color:var(--fg); border:1px solid var(--border); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
            gap:.8rem; margin:1.5rem 0; }}
  .metric {{ background:var(--card); border:1px solid var(--border); border-radius:.6rem;
             padding:1rem; text-align:center; }}
  .metric .v {{ font-size:1.6rem; font-weight:700; }}
  .metric .l {{ color:var(--muted); font-size:.82rem; text-transform:uppercase;
                letter-spacing:.03em; }}
  table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:.92rem; }}
  th,td {{ border:1px solid var(--border); padding:.45rem .6rem; text-align:center; }}
  thead th {{ background:var(--card); }}
  td:first-child, th:first-child {{ text-align:left; }}
  figure {{ margin:1.5rem 0; }}
  img {{ max-width:100%; border:1px solid var(--border); border-radius:.5rem; }}
  figcaption {{ color:var(--muted); font-size:.88rem; margin-top:.4rem; }}
  code {{ background:var(--card); padding:.1rem .35rem; border-radius:.3rem;
          font-size:.88em; }}
  footer {{ margin-top:3rem; padding-top:1.5rem; border-top:1px solid var(--border);
            color:var(--muted); font-size:.9rem; }}
  a {{ color:var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>A Cointegration-Based Statistical-Arbitrage Strategy</h1>
  <div class="sub">Engle&ndash;Granger &amp; Johansen cointegration, OLS hedge
  ratios, and a z-score pairs-trading backtest on the {y}/{x} ETF pair.</div>
  <div style="margin-top:.8rem"><span class="badge">{data_badge}</span></div>
  <div class="btns">
    <a class="btn btn-primary" href="./paper.pdf">Download full paper (PDF)</a>
    <a class="btn btn-ghost" href="https://github.com/maddoxk/quant-statarb-research">View source on GitHub</a>
  </div>
</header>

<p>This research builds a market-neutral pairs-trading strategy on
<strong>{y}</strong> (iShares MSCI Australia) and <strong>{x}</strong>
(iShares MSCI Canada), two commodity-linked country ETFs. We test for a long-run
equilibrium relationship, construct a mean-reverting spread, and trade its
deviations with a rolling z-score rule &mdash; reporting honest, reproducible
metrics on <strong>{M['n_obs']:,}</strong> daily observations
({M['start']} &rarr; {M['end']}).</p>

<h2>Headline results</h2>
<div class="cards">
  <div class="metric"><div class="v">{b['sharpe']:.2f}</div><div class="l">Sharpe (ann.)</div></div>
  <div class="metric"><div class="v">{b['total_return']*100:.1f}%</div><div class="l">Total return</div></div>
  <div class="metric"><div class="v">{b['ann_return']*100:.2f}%</div><div class="l">Ann. return</div></div>
  <div class="metric"><div class="v">{b['max_drawdown']*100:.1f}%</div><div class="l">Max drawdown</div></div>
  <div class="metric"><div class="v">{b['n_trades']}</div><div class="l">Trades</div></div>
  <div class="metric"><div class="v">{b['hit_rate']*100:.1f}%</div><div class="l">Hit rate</div></div>
</div>
<p class="sub">Net of {p['cost_bps']:g} bps/side transaction costs; entry
&plusmn;{p['entry']:g}&sigma;, exit &plusmn;{p['exit']:g}&sigma;, rolling window
{p['window']} days.</p>

<h2>1. Cointegration evidence</h2>
<p>Both the Engle&ndash;Granger two-step test and the Johansen trace test agree
that the pair is cointegrated.</p>
<table>
  <thead><tr><th>Test</th><th>Statistic</th><th>p-value / 95% crit.</th></tr></thead>
  <tbody>
    <tr><td>Engle&ndash;Granger cointegration</td><td>{c['engle_granger_stat']:.3f}</td><td>p = {c['engle_granger_pvalue']:.4f}</td></tr>
    <tr><td>ADF on spread residual</td><td>{c['adf_spread_stat']:.3f}</td><td>p = {c['adf_spread_pvalue']:.4f}</td></tr>
    <tr><td>Johansen trace (r=0)</td><td>{c['johansen_trace_stat']:.3f}</td><td>crit&#8324;&#8325; = {c['johansen_trace_crit_95']:.3f}</td></tr>
    <tr><td>OLS hedge ratio &beta;</td><td colspan="2">{c['hedge_ratio_beta']:.4f}</td></tr>
  </tbody>
</table>
<figure>
  <img src="./figures/prices.png" alt="Normalised prices">
  <figcaption>Figure 1. Normalised {y} and {x} prices indexed to 1.0 at sample start.</figcaption>
</figure>

<h2>2. Spread &amp; z-score signal</h2>
<p>The traded spread is <code>s&#8348; = {y} &minus; ({c['ols_alpha']:.3f} +
{c['hedge_ratio_beta']:.3f}&middot;{x})</code>. We enter when the rolling z-score
breaches &plusmn;{p['entry']:g} and exit when it reverts inside
&plusmn;{p['exit']:g}.</p>
<figure>
  <img src="./figures/spread_zscore.png" alt="Spread and z-score">
  <figcaption>Figure 2. The OLS spread (top) and its rolling z-score with entry/exit bands (bottom).</figcaption>
</figure>

<h2>3. Backtest equity curve</h2>
<figure>
  <img src="./figures/equity_curve.png" alt="Equity curve">
  <figcaption>Figure 3. Cumulative equity of the dollar-neutral strategy, net of costs.</figcaption>
</figure>

<h2>4. Sensitivity analysis</h2>
<p>Annualised Sharpe ratio across entry thresholds (rows) and per-side
transaction costs (columns). The edge peaks at moderate thresholds and decays
monotonically as costs rise.</p>
<table>
  <thead><tr><th>Entry z*</th>{cost_header}</tr></thead>
  <tbody>
{sens_rows()}
  </tbody>
</table>
<figure>
  <img src="./figures/sensitivity_heatmap.png" alt="Sensitivity heatmap">
  <figcaption>Figure 4. Sharpe-ratio sensitivity heatmap.</figcaption>
</figure>

<h2>Reproduce it</h2>
<p>Everything runs offline from a bundled CSV:</p>
<pre style="background:var(--card);padding:1rem;border-radius:.5rem;overflow:auto"><code>pip install -r requirements.txt
python src/run_analysis.py      # cointegration + backtest + figures
python -m pytest tests/ -q      # 15 unit tests
~/.local/bin/tectonic -X compile paper.tex --outdir .</code></pre>

<footer>
  &copy; 2026 Maddox Krape &middot; MIT License &middot;
  <a href="https://github.com/maddoxk/quant-statarb-research">github.com/maddoxk/quant-statarb-research</a>
  <br>Data source: {data_badge}. For research/educational use only &mdash; not
  investment advice.
</footer>
</div>
</body>
</html>
"""

open(os.path.join(SITE, "index.html"), "w").write(html)
print("wrote site/index.html and copied figures + paper.pdf")
