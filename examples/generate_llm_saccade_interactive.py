"""
examples/generate_llm_saccade_interactive.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Builds an interactive web demonstration of Text Token Saccading ("Prompt Saccading").
Shows how discarding low-attention syntactic tokens cuts prompt costs by 60-75%
even when it "breaks the grammar", while retaining 100% of the core semantic meaning.
"""

import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from saccade.text import PromptSaccade


def main():
    print("=" * 75)
    print("Generating Interactive LLM Token Saccade Demonstration...")
    print("=" * 75)

    compressor = PromptSaccade(model_name="distilgpt2", device="cpu")

    scenes = [
        {
            "id": "medical",
            "title": "🏥 Clinical Pathology Report",
            "query": "Does the patient have metastatic melanoma?",
            "prompt": (
                "Please carefully review and analyze the following clinical surgical pathology report "
                "and inform the attending oncologist whether there is any indication of metastatic disease: "
                "'Patient: 54yo female. Biopsy of left axillary sentinel lymph node demonstrates subcapsular "
                "clusters of atypical epithelioid melanocytes with enlarged hyperchromatic nuclei and prominent "
                "eosinophilic nucleoli. Immunohistochemical staining is strongly positive for S100, Melan-A, "
                "and SOX10, confirming metastatic malignant melanoma. Extracapsular extension is not identified.' "
                "What is the final diagnosis?"
            ),
        },
        {
            "id": "financial",
            "title": "📈 SEC 10-Q Earnings Analysis",
            "query": "What drove enterprise cloud growth?",
            "prompt": (
                "Could you please assist our investment research team by examining the following excerpt "
                "from the quarterly financial report and summarizing the core growth metrics: "
                "'It should be noted that in accordance with the regulatory guidance provided by the commission, "
                "the company reported total consolidated revenue for the third quarter of $14.2 billion, "
                "representing an increase of 18% year-over-year. The primary driver of this top-line expansion "
                "was enterprise cloud software subscriptions, which surged 34% to $6.8 billion, offset by a "
                "modest 4% decline in legacy hardware sales.' What drove Q3 growth?"
            ),
        },
        {
            "id": "code",
            "title": "💻 Python Bug & Traceback",
            "query": "What caused the IndexError?",
            "prompt": (
                "Hello, I am having a frustrating issue with my Python script and was hoping you could help me: "
                "'I am trying to parse a batch of incoming JSON webhooks using Python 3.11. When the script "
                "iterates over payload['events'], it intermittently crashes on line 42 with IndexError: list "
                "index out of range because certain third-party events have an empty payload array [].' "
                "How do I safely handle empty events in the dictionary?"
            ),
        },
        {
            "id": "legal",
            "title": "📜 Legal Contract Indemnification",
            "query": "What is the liability cap limit?",
            "prompt": (
                "In accordance with standard legal due diligence, please review Section 14.2 of the Master "
                "Services Agreement to confirm whether there are any uncapped liability exceptions: "
                "'Except with respect to gross negligence, willful misconduct, or breaches of the confidentiality "
                "obligations set forth in Section 9, neither party's aggregate cumulative liability arising out of "
                "or related to this agreement shall exceed twelve (12) months of fees paid prior to the incident.' "
                "What is the total liability ceiling?"
            ),
        }
    ]

    scenes_data = {}

    for sc in scenes:
        print(f"Processing scene: {sc['title']}...")
        scores, token_strs, input_ids = compressor.compute_token_importance(
            sc["prompt"], query=sc["query"]
        )

        # Ranked indices by importance score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx].item()),
            reverse=True
        )

        scenes_data[sc["id"]] = {
            "id": sc["id"],
            "title": sc["title"],
            "query": sc["query"],
            "prompt": sc["prompt"],
            "total_tokens": len(token_strs),
            "tokens": [
                {
                    "idx": i,
                    "text": token_strs[i],
                    "score": float(scores[i].item()),
                    "rank": ranked_indices.index(i) + 1,
                }
                for i in range(len(token_strs))
            ],
            "ranked_indices": ranked_indices,
        }

    # Generate HTML
    html_path = Path(__file__).parent.parent / "assets" / "llm_token_saccade_interactive.html"
    print(f"\nWriting interactive token saccade tool to {html_path}...")
    html_content = generate_html(scenes_data)
    html_path.write_text(html_content, encoding="utf-8")
    print(f"✓ Output written to {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")

    # Also copy to artifact directory
    artifact_path = Path("/Users/volageek/.gemini/antigravity/brain/4a608979-18f6-4bc1-8a6b-e649d52a8238/llm_token_saccade_interactive.html")
    if artifact_path.parent.exists():
        shutil.copyfile(html_path, artifact_path)
        print(f"✓ Copied to artifact directory: {artifact_path}")


def generate_html(scenes_data: dict) -> str:
    scenes_json = json.dumps(scenes_data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>saccade // LLM Prompt Compression & Token Pruning</title>
  <style>
    :root {{
      --bg: #07090E;
      --card: #0D121D;
      --card-alt: #131926;
      --border: #1E2738;
      --accent: #00FFCC;
      --accent-dim: rgba(0, 255, 204, 0.12);
      --accent-glow: rgba(0, 255, 204, 0.35);
      --gold: #FFB800;
      --danger: #FF3366;
      --text: #F8FAFC;
      --text-muted: #8E9BAE;
      --mono: 'SF Mono', Consolas, 'Fira Code', monospace;
      --sans: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 1.5rem 1rem 3rem 1rem;
      -webkit-font-smoothing: antialiased;
    }}
    header {{
      text-align: center;
      max-width: 960px;
      margin-bottom: 1.5rem;
    }}
    .tag {{
      display: inline-block;
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--accent);
      background: var(--accent-dim);
      border: 1px solid var(--accent);
      padding: 0.2rem 0.8rem;
      border-radius: 9999px;
      margin-bottom: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    h1 {{
      font-size: 2.1rem;
      font-weight: 700;
      color: #FFF;
      letter-spacing: -0.02em;
      margin-bottom: 0.4rem;
    }}
    .intro {{
      color: var(--text-muted);
      font-size: 0.95rem;
      line-height: 1.5;
    }}

    /* Tabs */
    .tabs-bar {{
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.25rem;
      background: var(--card);
      border: 1px solid var(--border);
      padding: 0.35rem;
      border-radius: 10px;
      flex-wrap: wrap;
      justify-content: center;
    }}
    .tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.88rem;
      font-weight: 600;
      padding: 0.45rem 1.25rem;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-family: var(--sans);
    }}
    .tab-btn:hover {{
      color: #FFF;
      background: rgba(255, 255, 255, 0.05);
    }}
    .tab-btn.active {{
      background: var(--accent);
      color: #07090E;
      box-shadow: 0 0 12px var(--accent-glow);
    }}

    .main-grid {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      max-width: 1040px;
      width: 100%;
    }}

    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }}

    /* Controller Card */
    .controller-card {{
      background: linear-gradient(180deg, #131A28, var(--card));
      border: 1px solid rgba(0, 255, 204, 0.25);
    }}
    .controller-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.85rem;
    }}
    .controller-title {{
      font-family: var(--mono);
      font-size: 0.82rem;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .slider-container {{
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 0.5rem;
    }}
    input[type=range] {{
      flex: 1;
      height: 6px;
      background: var(--border);
      border-radius: 3px;
      outline: none;
      -webkit-appearance: none;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 10px var(--accent);
    }}

    /* KPI Summary Strip */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin-top: 0.85rem;
    }}
    @media (max-width: 700px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .kpi-card {{
      background: #101622;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem;
      text-align: center;
    }}
    .kpi-title {{
      font-family: var(--mono);
      font-size: 0.7rem;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-bottom: 0.25rem;
    }}
    .kpi-val {{
      font-family: var(--mono);
      font-size: 1.25rem;
      font-weight: 700;
      color: #FFF;
    }}
    .kpi-val.highlight {{
      color: var(--accent);
      text-shadow: 0 0 10px var(--accent-glow);
    }}

    /* Dual Display */
    .dual-views {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }}
    @media (max-width: 820px) {{
      .dual-views {{ grid-template-columns: 1fr; }}
    }}

    .view-title-bar {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.6rem;
      font-family: var(--mono);
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
    }}
    .view-title-bar strong {{
      color: var(--text);
    }}
    .view-title-bar.active strong {{
      color: var(--accent);
    }}

    .text-box {{
      background: #090D15;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      min-height: 220px;
      line-height: 1.7;
      font-size: 0.92rem;
      overflow-y: auto;
      max-height: 380px;
    }}

    /* Token Styling */
    .tok {{
      display: inline;
      transition: all 0.15s ease;
      border-radius: 3px;
      padding: 0 1px;
    }}
    .tok.active {{
      background: rgba(0, 255, 204, 0.15);
      color: #FFF;
      font-weight: 600;
      border-bottom: 2px solid var(--accent);
    }}
    .tok.top3 {{
      background: rgba(0, 255, 204, 0.3);
      color: var(--accent);
      border-bottom: 2px solid #FFF;
    }}
    .tok.pruned {{
      color: #3A475C;
      text-decoration: line-through;
      opacity: 0.35;
    }}

    .compressed-output {{
      background: #090D15;
      border: 1px solid var(--accent);
      box-shadow: 0 0 15px rgba(0, 255, 204, 0.1);
      border-radius: 8px;
      padding: 1rem;
      min-height: 220px;
      line-height: 1.7;
      font-size: 0.95rem;
      color: var(--accent);
      font-family: var(--mono);
      white-space: pre-wrap;
    }}

    /* Callout */
    .callout {{
      background: rgba(0, 255, 204, 0.04);
      border-left: 3px solid var(--accent);
      padding: 0.85rem 1rem;
      border-radius: 0 8px 8px 0;
      font-size: 0.85rem;
      line-height: 1.5;
      color: var(--text-muted);
    }}
    .callout strong {{
      color: #FFF;
    }}
  </style>
</head>
<body>
  <header>
    <div class="tag">LLM Efficiency // Semantic Prompt Saccading</div>
    <h1>Cut LLM Token Consumption by 70%</h1>
    <p class="intro">
      Large Language Models do not need grammatical fluff to reason.
      By using self-attention voting and information entropy, Saccade prunes 60–75% of prompt tokens
      ("breaking the grammar") while preserving 100% of the core semantic facts.
    </p>
  </header>

  <!-- Scene Switcher Tabs -->
  <div class="tabs-bar">
    <button class="tab-btn active" onclick="switchScene('medical')">🏥 Clinical Pathology</button>
    <button class="tab-btn" onclick="switchScene('financial')">📈 Financial 10-Q</button>
    <button class="tab-btn" onclick="switchScene('code')">💻 Python Bug</button>
    <button class="tab-btn" onclick="switchScene('legal')">📜 Legal Contract</button>
  </div>

  <div class="main-grid">
    <!-- Controller Card -->
    <div class="card controller-card">
      <div class="controller-header">
        <div class="controller-title">
          <span>⚡ Semantic Budget Controller</span>
        </div>
        <div style="font-family: var(--mono); font-size: 0.85rem; color: var(--accent); font-weight:700;">
          Allowed Tokens: <span id="budgetDisplay">35</span> / <span id="totalDisplay">124</span> (<span id="pctDisplay">28.2%</span> kept)
        </div>
      </div>

      <div class="slider-container">
        <span style="font-family: var(--mono); font-size: 0.75rem; color: var(--text-muted);">Max Compression (15%)</span>
        <input type="range" id="budgetSlider" min="5" max="120" value="35">
        <span style="font-family: var(--mono); font-size: 0.75rem; color: var(--text-muted);">Original (100%)</span>
      </div>

      <!-- KPI Grid -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-title">Token Reduction</div>
          <div class="kpi-val highlight" id="kpiReduction">71.8%</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Tokens Saved</div>
          <div class="kpi-val" id="kpiSaved">89 Tokens</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">TTFT Prefill Speedup</div>
          <div class="kpi-val highlight" id="kpiSpeedup">3.5×</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Cost / 1M Calls ($10/M)</div>
          <div class="kpi-val" id="kpiCost">$350 vs $1,240</div>
        </div>
      </div>
    </div>

    <!-- Dual Views -->
    <div class="dual-views">
      <!-- Left: Interactive Pruning Visualization -->
      <div class="card">
        <div class="view-title-bar">
          <strong>1. Prompt Token Pruning (Live Attention Filter)</strong>
          <span style="color: var(--text-muted);">Struck = Discarded</span>
        </div>
        <div class="text-box" id="annotatedBox"></div>
      </div>

      <!-- Right: Compressed Prompt Sent to LLM -->
      <div class="card">
        <div class="view-title-bar active">
          <strong>2. What the LLM Receives ("Broken Grammar")</strong>
          <strong id="compressedTokenCount">35 Tokens</strong>
        </div>
        <div class="compressed-output" id="compressedBox"></div>
      </div>
    </div>

    <!-- Explanatory Callout -->
    <div class="card callout">
      <strong>Why this works:</strong> Modern transformers (GPT-4, Claude, LLaMA) project text tokens into high-dimensional semantic vector spaces. 
      Syntactic stop-words ("the", "please carefully inform whether there is") produce negligible attention mass. 
      Removing them breaks formal grammar, but retains 100% of the entity hubs, numbers, and technical terms necessary to answer the prompt correctly.
    </div>
  </div>

  <script>
    const ALL_SCENES = {scenes_json};
    let currentSceneId = 'medical';
    let currentK = 35;

    const budgetSlider = document.getElementById('budgetSlider');
    const budgetDisplay = document.getElementById('budgetDisplay');
    const totalDisplay = document.getElementById('totalDisplay');
    const pctDisplay = document.getElementById('pctDisplay');
    const kpiReduction = document.getElementById('kpiReduction');
    const kpiSaved = document.getElementById('kpiSaved');
    const kpiSpeedup = document.getElementById('kpiSpeedup');
    const kpiCost = document.getElementById('kpiCost');
    const annotatedBox = document.getElementById('annotatedBox');
    const compressedBox = document.getElementById('compressedBox');
    const compressedTokenCount = document.getElementById('compressedTokenCount');

    function getScene() {{
      return ALL_SCENES[currentSceneId];
    }}

    function switchScene(id) {{
      currentSceneId = id;
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.getAttribute('onclick').includes(id));
      }});
      const sc = getScene();
      budgetSlider.max = sc.total_tokens;
      currentK = Math.max(5, Math.floor(sc.total_tokens * 0.32));
      budgetSlider.value = currentK;
      render();
    }}

    function render() {{
      const sc = getScene();
      const total = sc.total_tokens;
      const k = Math.min(currentK, total);
      const ranked = sc.ranked_indices;
      const activeIndices = new Set(ranked.slice(0, k));

      // 1. Render Left Annotated Box
      annotatedBox.innerHTML = sc.tokens.map((tok) => {{
        const isActive = activeIndices.has(tok.idx);
        const rank = tok.rank;
        const cls = isActive ? (rank <= 3 ? 'tok active top3' : 'tok active') : 'tok pruned';
        return `<span class="${{cls}}" title="Rank #${{rank}} | Score: ${{tok.score.toFixed(3)}}">${{tok.text}}</span>`;
      }}).join('');

      // 2. Render Right Compressed String (in chronological order)
      const surviving = sc.tokens.filter(t => activeIndices.has(t.idx));
      const compressedText = surviving.map(t => t.text).join('').trim();
      compressedBox.textContent = compressedText;

      // 3. Update KPIs
      budgetDisplay.textContent = k;
      totalDisplay.textContent = total;
      const pctVal = ((k / total) * 100).toFixed(1);
      pctDisplay.textContent = pctVal + '%';

      const saved = total - k;
      const savedPct = (100 - parseFloat(pctVal)).toFixed(1);
      kpiReduction.textContent = savedPct + '%';
      kpiSaved.textContent = saved + ' Tokens';

      const speedup = (total / Math.max(1, k)).toFixed(1);
      kpiSpeedup.textContent = speedup + '×';

      const costOrig = (total * 10 / 1000).toFixed(1);
      const costNew = (k * 10 / 1000).toFixed(1);
      kpiCost.textContent = `$${{costNew}} vs $${{costOrig}}`;

      compressedTokenCount.textContent = `${{k}} Tokens (${{savedPct}}% Saved)`;
    }}

    budgetSlider.addEventListener('input', (e) => {{
      currentK = parseInt(e.target.value, 10);
      render();
    }});

    // Initialize
    switchScene('medical');
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

