"""
Generate self-contained HTML for zembed-1 configuration explorer.

Usage:
    python ml/training/embedding/generate_config_explorer.py
"""

import asyncio
import json
import hashlib
import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.ai import AIEmbedding, decode_embedding
from ml.s3_utils import s3_list_objects, s3_read_key
from ml.utils import async_iterate_with_prefetching, async_zip, unwrap, wrap_sem
from ml.training.embedding.evaluate import QueryEvals

# ── Eval checkpoints ─────────────────────────────────────────────────────────

CHECKPOINTS: list[str] = [
    "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450",
    "aimodel-embed/modal/voyage-4.zeroentropy.dev",
    "aimodel-embed/modal/zeroentropy--voyage-4-nano-model-endpoint.modal.run",
    "aimodel-embed/openai/text-embedding-3-large",
    "aimodel-embed/openai/text-embedding-3-small",
]

# ── Configuration registry ───────────────────────────────────────────────────

CONFIGURATIONS: list[dict[str, object]] = [
    # zembed-1: cloud
    {"id": "zembed-1-f32-full-cloud", "name": "zembed-1", "provider": "ZeroEntropy", "quantization": "float32", "matryoshka": "full", "dims": 2048, "bits_per_dim": 32, "infra": "ZE Serverless", "price_per_m_tokens": 0.06, "latency_ms": 45, "security": "cloud", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.0, "color": "#4f46e5", "group": "zembed-1"},
    {"id": "zembed-1-f32-half-cloud", "name": "zembed-1 (½ dim)", "provider": "ZeroEntropy", "quantization": "float32", "matryoshka": "½", "dims": 1024, "bits_per_dim": 32, "infra": "ZE Serverless", "price_per_m_tokens": 0.06, "latency_ms": 45, "security": "cloud", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.015, "color": "#6366f1", "group": "zembed-1"},
    {"id": "zembed-1-f32-quarter-cloud", "name": "zembed-1 (¼ dim)", "provider": "ZeroEntropy", "quantization": "float32", "matryoshka": "¼", "dims": 512, "bits_per_dim": 32, "infra": "ZE Serverless", "price_per_m_tokens": 0.06, "latency_ms": 45, "security": "cloud", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.04, "color": "#818cf8", "group": "zembed-1"},
    {"id": "zembed-1-f32-eighth-cloud", "name": "zembed-1 (⅛ dim)", "provider": "ZeroEntropy", "quantization": "float32", "matryoshka": "⅛", "dims": 256, "bits_per_dim": 32, "infra": "ZE Serverless", "price_per_m_tokens": 0.06, "latency_ms": 45, "security": "cloud", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.08, "color": "#c7d2fe", "group": "zembed-1"},
    {"id": "zembed-1-int8-full-cloud", "name": "zembed-1 (int8)", "provider": "ZeroEntropy", "quantization": "int8", "matryoshka": "full", "dims": 2048, "bits_per_dim": 8, "infra": "ZE Serverless", "price_per_m_tokens": 0.06, "latency_ms": 45, "security": "cloud", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.005, "color": "#7c3aed", "group": "zembed-1"},
    {"id": "zembed-1-int8-half-cloud", "name": "zembed-1 (½ dim, int8)", "provider": "ZeroEntropy", "quantization": "int8", "matryoshka": "½", "dims": 1024, "bits_per_dim": 8, "infra": "ZE Serverless", "price_per_m_tokens": 0.06, "latency_ms": 45, "security": "cloud", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.02, "color": "#8b5cf6", "group": "zembed-1"},
    {"id": "zembed-1-bin-full-cloud", "name": "zembed-1 (binary)", "provider": "ZeroEntropy", "quantization": "binary", "matryoshka": "full", "dims": 2048, "bits_per_dim": 1, "infra": "ZE Serverless", "price_per_m_tokens": 0.06, "latency_ms": 45, "security": "cloud", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.025, "color": "#a78bfa", "group": "zembed-1"},
    {"id": "zembed-1-bin-half-cloud", "name": "zembed-1 (½ dim, binary)", "provider": "ZeroEntropy", "quantization": "binary", "matryoshka": "½", "dims": 1024, "bits_per_dim": 1, "infra": "ZE Serverless", "price_per_m_tokens": 0.06, "latency_ms": 45, "security": "cloud", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.035, "color": "#a5b4fc", "group": "zembed-1"},
    # zembed-1: VPC
    {"id": "zembed-1-f32-full-vpc", "name": "zembed-1 (VPC)", "provider": "ZeroEntropy", "quantization": "float32", "matryoshka": "full", "dims": 2048, "bits_per_dim": 32, "infra": "AWS/Azure VPC", "price_per_m_tokens": 0.08, "latency_ms": 35, "security": "vpc", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.0, "color": "#4f46e5", "group": "zembed-1"},
    {"id": "zembed-1-f32-half-vpc", "name": "zembed-1 (½ dim, VPC)", "provider": "ZeroEntropy", "quantization": "float32", "matryoshka": "½", "dims": 1024, "bits_per_dim": 32, "infra": "AWS/Azure VPC", "price_per_m_tokens": 0.08, "latency_ms": 35, "security": "vpc", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.015, "color": "#6366f1", "group": "zembed-1"},
    {"id": "zembed-1-bin-half-vpc", "name": "zembed-1 (½ dim, binary, VPC)", "provider": "ZeroEntropy", "quantization": "binary", "matryoshka": "½", "dims": 1024, "bits_per_dim": 1, "infra": "AWS/Azure VPC", "price_per_m_tokens": 0.08, "latency_ms": 35, "security": "vpc", "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.035, "color": "#a5b4fc", "group": "zembed-1"},
    # Competitors
    {"id": "voyage-4-large", "name": "Voyage 4 Large", "provider": "Voyage", "quantization": "float32", "matryoshka": "full", "dims": 2048, "bits_per_dim": 32, "infra": "Voyage Cloud", "price_per_m_tokens": 0.12, "latency_ms": 60, "security": "cloud", "checkpoint": "aimodel-embed/modal/voyage-4.zeroentropy.dev", "accuracy_penalty": 0.0, "color": "#f59e0b", "group": "Competitors"},
    {"id": "voyage-4", "name": "Voyage 4", "provider": "Voyage", "quantization": "float32", "matryoshka": "full", "dims": 1024, "bits_per_dim": 32, "infra": "Voyage Cloud", "price_per_m_tokens": 0.06, "latency_ms": 40, "security": "cloud", "checkpoint": "aimodel-embed/modal/voyage-4.zeroentropy.dev", "accuracy_penalty": 0.01, "color": "#d97706", "group": "Competitors"},
    {"id": "voyage-4-lite", "name": "Voyage 4 Lite", "provider": "Voyage", "quantization": "float32", "matryoshka": "full", "dims": 512, "bits_per_dim": 32, "infra": "Voyage Cloud", "price_per_m_tokens": 0.02, "latency_ms": 25, "security": "cloud", "checkpoint": "aimodel-embed/modal/zeroentropy--voyage-4-nano-model-endpoint.modal.run", "accuracy_penalty": 0.0, "color": "#fbbf24", "group": "Competitors"},
    {"id": "openai-3-large", "name": "text-embedding-3-large", "provider": "OpenAI", "quantization": "float32", "matryoshka": "full", "dims": 3072, "bits_per_dim": 32, "infra": "OpenAI Cloud", "price_per_m_tokens": 0.13, "latency_ms": 50, "security": "cloud", "checkpoint": "aimodel-embed/openai/text-embedding-3-large", "accuracy_penalty": 0.0, "color": "#10a37f", "group": "Competitors"},
    {"id": "openai-3-small", "name": "text-embedding-3-small", "provider": "OpenAI", "quantization": "float32", "matryoshka": "full", "dims": 1536, "bits_per_dim": 32, "infra": "OpenAI Cloud", "price_per_m_tokens": 0.02, "latency_ms": 35, "security": "cloud", "checkpoint": "aimodel-embed/openai/text-embedding-3-small", "accuracy_penalty": 0.0, "color": "#6ee7b7", "group": "Competitors"},
    {"id": "cohere-embed-4", "name": "Cohere Embed 4", "provider": "Cohere", "quantization": "float32", "matryoshka": "full", "dims": 1024, "bits_per_dim": 32, "infra": "Cohere Cloud", "price_per_m_tokens": 0.12, "latency_ms": 55, "security": "cloud", "accuracy_override": 0.72, "color": "#d946ef", "group": "Competitors"},
]

K_MAX = 100
DOWNLOAD_SEM = asyncio.Semaphore(64)
OUTPUT_PATH = Path("config_explorer.html")
CACHE_DIR = Path(".cache_evals")
RECALL_K_FOR_ACCURACY = 10


# ── S3 / eval helpers ────────────────────────────────────────────────────────

async def load_evals(checkpoint: str) -> dict[str, list[QueryEvals]]:
    prefix = f"checkpoints/{checkpoint}/"
    eval_paths: list[str] = []
    async for path in s3_list_objects(prefix):
        if path.endswith("/evals.jsonl"):
            eval_paths.append(path)
    result: dict[str, list[QueryEvals]] = {}
    async for data, path in async_iterate_with_prefetching(
        [lambda p=path: async_zip(wrap_sem(s3_read_key(p), DOWNLOAD_SEM), p) for path in eval_paths],
        max_concurrent=32,
    ):
        split_id = path.rsplit("/", 2)[-2]
        result[split_id] = [
            QueryEvals.model_validate_json(line)
            for line in unwrap(data).decode().strip().split("\n")
            if line.strip()
        ]
    return result


def recalls_at_all_k(similarities: AIEmbedding, qrels: dict[int, float], k_max: int) -> NDArray[np.float32]:
    total_relevancy = sum(qrels.values())
    assert total_relevancy > 0
    sorted_indices = np.argsort(-similarities, kind="stable")[:k_max]
    recalls = np.zeros(k_max, dtype=np.float32)
    hits = 0.0
    for k_idx, idx in enumerate(sorted_indices):
        hits += qrels.get(int(idx), 0.0)
        recalls[k_idx] = hits / total_relevancy
    if len(sorted_indices) < k_max:
        recalls[len(sorted_indices):] = recalls[len(sorted_indices) - 1]
    return recalls


def cache_key(checkpoints: list[str]) -> Path:
    h = hashlib.sha256(",".join(checkpoints).encode()).hexdigest()[:16]
    return CACHE_DIR / f"evals_{h}.pkl"


# ── HTML sections ────────────────────────────────────────────────────────────

def html_head() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>zembed-1 Configuration Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #f5f5f7; padding: 40px 20px; color: #1a1a1a;
    -webkit-font-smoothing: antialiased;
}
.page { max-width: 1100px; margin: 0 auto; }
.card {
    background: white; border-radius: 16px; border: 1px solid #e5e5e5;
    box-shadow: 0 4px 24px rgba(0,0,0,0.05); padding: 32px 40px; margin-bottom: 28px;
}
.card h2 { font-size: 20px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 6px; }
.card .desc { font-size: 14px; color: #888; margin-bottom: 24px; line-height: 1.5; }
.input-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px; margin-bottom: 20px;
}
.input-group label {
    display: block; font-size: 11px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
}
.input-group input {
    width: 100%; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px;
    font-size: 14px; font-family: inherit; background: #fafafa; transition: border-color 0.15s;
}
.input-group input:focus { outline: none; border-color: #4f46e5; background: white; }
.input-hint { font-size: 11px; color: #aaa; margin-top: 3px; }
.divider { height: 1px; background: #eee; margin: 20px 0; }
.filter-section { margin-bottom: 16px; }
.filter-section-label {
    font-size: 11px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
}
.filter-row { display: flex; flex-wrap: wrap; gap: 8px; }
.filter-chip {
    padding: 5px 14px; border: 1px solid #ddd; border-radius: 20px;
    font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.15s;
    background: white; user-select: none;
}
.filter-chip:hover { background: #f0f0f0; }
.filter-chip.active { background: #4f46e5; color: white; border-color: #4f46e5; }
.ternary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 8px; }
.ternary-box h3 { font-size: 14px; font-weight: 600; color: #555; margin-bottom: 4px; text-align: center; }
.ternary-box .sub { font-size: 12px; color: #aaa; text-align: center; margin-bottom: 8px; }
.ternary-box {
    display: flex;
    flex-direction: column;
}
.ternary-box > div:last-child {
    margin-top: auto;
}
.scatter-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 8px; }
.scatter-box {
    position: relative; height: 360px; background: #fafafa;
    border: 1px solid #eee; border-radius: 12px; padding: 16px;
}
.filter-chip.active.dominated {
    background: #e0e0e0;
    color: #999;
    border-color: #ccc;
    text-decoration: line-through;
}
.filter-chip .dom-tag {
    font-size: 9px;
    font-weight: 600;
    color: #aaa;
    margin-left: 4px;
    text-decoration: none;
    display: inline-block;
}
.scatter-box h3 { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #555; }
.scatter-box canvas { max-height: 300px; }
.result-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.result-table thead th {
    text-align: left; padding: 10px 12px; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px; color: #999;
    border-bottom: 2px solid #eee; cursor: pointer; user-select: none; white-space: nowrap;
}
.result-table thead th:hover { color: #4f46e5; }
.result-table thead th.num { text-align: right; }
.result-table tbody td { padding: 11px 12px; border-bottom: 1px solid #f0f0f0; }
.result-table tbody td.num {
    text-align: right; font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', 'Monaco', monospace; font-size: 12px;
}
.result-table tbody tr { transition: background 0.1s; }
.result-table tbody tr:hover { background: #f8f8ff; }
.result-table tbody tr.is-ze { background: #f5f3ff; }
.result-table tbody tr.is-ze:hover { background: #ede9fe; }
.model-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 8px; vertical-align: middle;
}
.tag {
    display: inline-block; font-size: 10px; font-weight: 600;
    padding: 2px 7px; border-radius: 4px; margin-left: 4px;
}
.tag-green { background: #ecfdf5; color: #059669; }
.tag-red { background: #fef2f2; color: #dc2626; }
.tag-blue { background: #eff6ff; color: #3b82f6; }
.section-label td {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #bbb; padding: 14px 12px 6px; border-bottom: 1px solid #f0f0f0;
}
.footnote { font-size: 11px; color: #aaa; line-height: 1.6; margin-top: 20px; }
@media (max-width: 800px) {
    .ternary-grid, .scatter-grid { grid-template-columns: 1fr; }
    .input-grid { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<div class="page">"""


def html_usage_inputs() -> str:
    return """
<div class="card">
    <h2>Your Workload</h2>
    <p class="desc">Enter your expected usage to see accurate cost comparisons across all configurations.</p>
    <div class="input-grid">
        <div class="input-group">
            <label>Corpus Size (docs)</label>
            <input type="text" id="corpusSize" value="10,000,000" oninput="recalc()">
        </div>
        <div class="input-group">
            <label>Avg Tokens / Document</label>
            <input type="text" id="avgDocTokens" value="512" oninput="recalc()">
        </div>
        <div class="input-group">
            <label>Queries / Month</label>
            <input type="text" id="queriesMonth" value="1,000,000" oninput="recalc()">
        </div>
        <div class="input-group">
            <label>Avg Tokens / Query</label>
            <input type="text" id="avgQueryTokens" value="32" oninput="recalc()">
        </div>
        <div class="input-group">
            <label>Vector DB $/GB/mo</label>
            <input type="text" id="vectorDbCost" value="0.15" oninput="recalc()">
            <div class="input-hint">Pinecone ≈ 0.33 · Qdrant ≈ 0.15 · Self-host ≈ 0.05</div>
        </div>
    </div>
</div>"""


def html_ternary() -> str:
    return """
<div class="card">
    <h2>Priority Tradeoff Map</h2>
    <p class="desc">Each point represents a weighting of cost, latency, and accuracy priorities. Color shows the optimal configuration. Toggle configurations to include in the comparison.</p>
    <div class="ternary-grid">
        <div class="ternary-box">
            <h3>Public Cloud</h3>
            <div class="sub">ZE Serverless + all competitors</div>
            <div class="filter-row" id="ternaryTogglesCloud" style="margin-bottom:12px;justify-content:center;"></div>
            <div id="ternaryCloud" style="width:100%;height:500px;"></div>
        </div>
        <div class="ternary-box">
            <h3>VPC / Private</h3>
            <div class="sub">AWS/Azure PrivateLink deployments</div>
            <div class="filter-row" id="ternaryTogglesVpc" style="margin-bottom:12px;justify-content:center;"></div>
            <div id="ternaryVpc" style="width:100%;height:500px;"></div>
        </div>
    </div>
</div>"""


def html_scatterplots() -> str:
    return """
<div class="card">
    <h2>Scatter Comparison</h2>
    <p class="desc">Filter configurations to compare on cost vs accuracy and latency vs accuracy.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-bottom:20px;">
        <div class="filter-section">
            <div class="filter-section-label">Provider</div>
            <div class="filter-row" id="filterProvider"></div>
        </div>
        <div class="filter-section">
            <div class="filter-section-label">Quantization</div>
            <div class="filter-row" id="filterQuant"></div>
        </div>
        <div class="filter-section">
            <div class="filter-section-label">Matryoshka</div>
            <div class="filter-row" id="filterMatryoshka"></div>
        </div>
        <div class="filter-section">
            <div class="filter-section-label">Security</div>
            <div class="filter-row" id="filterSecurity"></div>
        </div>
    </div>
    <div class="divider"></div>
    <div class="scatter-grid">
        <div class="scatter-box">
            <h3>Monthly Cost vs Recall@10</h3>
            <canvas id="scatterCostAcc"></canvas>
        </div>
        <div class="scatter-box">
            <h3>Latency vs Recall@10</h3>
            <canvas id="scatterLatAcc"></canvas>
        </div>
    </div>
</div>"""


def html_table() -> str:
    return """
<div class="card">
    <h2>Configuration Comparison</h2>
    <p class="desc">Click column headers to sort. Respects filters above. All costs are monthly.</p>
    <div style="overflow-x:auto;">
        <table class="result-table" id="resultTable">
            <thead><tr>
                <th data-col="name">Model</th>
                <th data-col="infra">Infra</th>
                <th data-col="dims" class="num">Dims</th>
                <th data-col="quantization">Quant</th>
                <th data-col="storage" class="num">Storage</th>
                <th data-col="embedCost" class="num">Embed†</th>
                <th data-col="queryCost" class="num">Query/mo</th>
                <th data-col="storageCost" class="num">Storage/mo</th>
                <th data-col="totalCost" class="num">Total/mo</th>
                <th data-col="accuracy" class="num">Recall@10</th>
                <th data-col="latency" class="num">Latency</th>
            </tr></thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>
    <div class="footnote">
        † One-time embedding cost amortized over 12 months. Storage = docs × dims × bits ÷ 8.
    </div>
</div>"""

def html_script(configs_json: str, recall_curves_json: str) -> str:
    return """
<script>
const ALL_CONFIGS = """ + configs_json + """;
const RECALL_CURVES = """ + recall_curves_json + """;
const RECALL_K = """ + str(RECALL_K_FOR_ACCURACY) + """;

const filters = { provider: new Set(), quant: new Set(), matryoshka: new Set(), security: new Set() };
let sortCol = 'totalCost';
let sortAsc = true;
let scatterChart1 = null;
let scatterChart2 = null;

function parseNum(id) {
    return parseFloat(document.getElementById(id).value.replace(/,/g, '')) || 0;
}
function fmtDollars(n) {
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (n >= 1000) return '$' + n.toLocaleString('en-US', { maximumFractionDigits: 0 });
    if (n >= 1) return '$' + n.toFixed(2);
    return '$' + n.toFixed(3);
}
function fmtDollarsMo(n) {
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M/mo';
    if (n >= 1000) return '$' + n.toLocaleString('en-US', { maximumFractionDigits: 0 }) + '/mo';
    if (n >= 1) return '$' + n.toFixed(0) + '/mo';
    return '$' + n.toFixed(2) + '/mo';
}
function fmtStorage(bytes) {
    const gb = bytes / (1024 ** 3);
    if (gb >= 1000) return (gb / 1000).toFixed(1) + ' TB';
    if (gb >= 1) return gb.toFixed(1) + ' GB';
    return (gb * 1024).toFixed(0) + ' MB';
}
function getAccuracy(c) {
    if (c.accuracy_override !== undefined) return c.accuracy_override;
    if (c.checkpoint && RECALL_CURVES[c.checkpoint])
        return Math.max(0, RECALL_CURVES[c.checkpoint][RECALL_K - 1] - (c.accuracy_penalty || 0));
    return 0;
}
function computeRow(c) {
    const corpusSize = parseNum('corpusSize');
    const avgDocTokens = parseNum('avgDocTokens');
    const queriesMonth = parseNum('queriesMonth');
    const avgQueryTokens = parseNum('avgQueryTokens');
    const vectorDbCost = parseNum('vectorDbCost');
    const totalDocTokens = corpusSize * avgDocTokens;
    const monthlyQueryTokens = queriesMonth * avgQueryTokens;
    const embedCost = (totalDocTokens / 1e6) * c.price_per_m_tokens;
    const queryCost = (monthlyQueryTokens / 1e6) * c.price_per_m_tokens;
    const storageBytes = corpusSize * c.dims * (c.bits_per_dim / 8);
    const storageCost = (storageBytes / (1024 ** 3)) * vectorDbCost;
    const totalCost = (embedCost / 12) + queryCost + storageCost;
    return { ...c, embedCost, queryCost, storageBytes, storageCost, totalCost, accuracy: getAccuracy(c) };
}
function getAllRows() { return ALL_CONFIGS.map(computeRow); }
function getFilteredRows() {
    return getAllRows().filter(r =>
        filters.provider.has(r.provider) &&
        filters.quant.has(r.quantization) &&
        filters.matryoshka.has(r.matryoshka) &&
        filters.security.has(r.security)
    );
}

// ── Filters ──
function initFilters() {
    const unique = (key) => [...new Set(ALL_CONFIGS.map(c => c[key]))];
    initChips('filterProvider', unique('provider'), 'provider');
    initChips('filterQuant', unique('quantization'), 'quant');
    initChips('filterMatryoshka', unique('matryoshka'), 'matryoshka');
    initChips('filterSecurity', unique('security'), 'security');
}
function initChips(containerId, values, filterKey) {
    const el = document.getElementById(containerId);
    values.forEach(v => {
        filters[filterKey].add(v);
        const chip = document.createElement('span');
        chip.className = 'filter-chip active';
        chip.textContent = v;
        chip.onclick = () => {
            if (filters[filterKey].has(v)) { filters[filterKey].delete(v); chip.classList.remove('active'); }
            else { filters[filterKey].add(v); chip.classList.add('active'); }
            recalc();
        };
        el.appendChild(chip);
    });
}

// ── Ternary ──
// ── Ternary ──
const TERNARY_RES = 80;
const ternaryState = {};
const ternaryToggles = { cloud: new Set(), vpc: new Set() };

function initTernaryToggles() {
    const allRows = getAllRows();
    initTernaryToggleSet('ternaryTogglesCloud', 'cloud', allRows.filter(r => r.security === 'cloud'));
    initTernaryToggleSet('ternaryTogglesVpc', 'vpc', allRows.filter(r => r.security === 'vpc'));
}

function initTernaryToggleSet(containerId, secKey, rows) {
    const el = document.getElementById(containerId);
    const seen = new Set();
    for (const r of rows) {
        if (seen.has(r.id)) continue;
        seen.add(r.id);
        ternaryToggles[secKey].add(r.id);
        const chip = document.createElement('span');
        chip.className = 'filter-chip active';
        chip.style.borderLeft = '4px solid ' + r.color;
        chip.textContent = r.name;
        chip.dataset.configId = r.id;
        chip.onclick = () => {
            if (ternaryToggles[secKey].has(r.id)) {
                ternaryToggles[secKey].delete(r.id);
                chip.classList.remove('active');
            } else {
                ternaryToggles[secKey].add(r.id);
                chip.classList.add('active');
            }
            recalcTernary();
        };
        el.appendChild(chip);
    }
}

function recalcTernary() {
    const all = getAllRows();
    const cloudRows = all.filter(r => r.security === 'cloud' && ternaryToggles.cloud.has(r.id));
    const vpcRows = all.filter(r => r.security === 'vpc' && ternaryToggles.vpc.has(r.id));
    renderTernary('ternaryCloud', cloudRows);
    renderTernary('ternaryVpc', vpcRows);
}

function renderTernary(divId, rows) {
    const secKey = divId === 'ternaryCloud' ? 'cloud' : 'vpc';
    const toggleContainer = document.getElementById(
        divId === 'ternaryCloud' ? 'ternaryTogglesCloud' : 'ternaryTogglesVpc'
    );

    if (rows.length === 0) {
        Plotly.purge(divId);
        document.getElementById(divId).innerHTML =
            '<p style="text-align:center;color:#aaa;padding:80px 20px;">No configurations selected.</p>';
        for (const chip of toggleContainer.children) {
            chip.classList.remove('dominated');
            const dt = chip.querySelector('.dom-tag');
            if (dt) dt.remove();
        }
        return;
    }


    const minCost = Math.min(...rows.map(r => r.totalCost));
    const maxCost = Math.max(...rows.map(r => r.totalCost));
    const minLat = Math.min(...rows.map(r => r.latency_ms));
    const maxLat = Math.max(...rows.map(r => r.latency_ms));
    const minAcc = Math.min(...rows.map(r => r.accuracy));
    const maxAcc = Math.max(...rows.map(r => r.accuracy));

    const normVal = (v, lo, hi, inv) => {
        const n = hi > lo ? (v - lo) / (hi - lo) : 0.5;
        return inv ? 1 - n : n;
    };
    const normed = rows.map(r => ({
        nCost: normVal(r.totalCost, minCost, maxCost, true),
        nLat: normVal(r.latency_ms, minLat, maxLat, true),
        nAcc: normVal(r.accuracy, minAcc, maxAcc, false),
    }));

    function bestIdx(wC, wL, wA) {
        let best = 0, bestS = -Infinity;
        for (let i = 0; i < normed.length; i++) {
            const s = wC * normed[i].nCost + wL * normed[i].nLat + wA * normed[i].nAcc;
            if (s > bestS) { bestS = s; best = i; }
        }
        return best;
    }

    // Find which configs actually win somewhere
    const winners = new Set();
    const ta = [], tb = [], tc = [], colors = [], hover = [];
    for (let i = 0; i <= TERNARY_RES; i++) {
        for (let j = 0; j <= TERNARY_RES - i; j++) {
            const k = TERNARY_RES - i - j;
            const wC = i / TERNARY_RES, wL = j / TERNARY_RES, wA = k / TERNARY_RES;
            const idx = bestIdx(wC, wL, wA);
            winners.add(idx);
            ta.push(wC); tb.push(wL); tc.push(wA);
            colors.push(rows[idx].color);
            hover.push(
                '<b>' + rows[idx].name + '</b>' +
                '<br>' +
                '<br>Cost: ' + fmtDollarsMo(rows[idx].totalCost) +
                '<br>Latency: ' + rows[idx].latency_ms + ' ms' +
                '<br>Recall@10: ' + (rows[idx].accuracy * 100).toFixed(1) + '%' +
                '<br>' +
                '<br><span style="color:#999">Priority weights:</span>' +
                '<br><span style="color:#999">Cost ' + (wC * 100).toFixed(0) + '% · Latency ' + (wL * 100).toFixed(0) + '% · Accuracy ' + (wA * 100).toFixed(0) + '%</span>'
            );
        }
    }

    ternaryState[divId] = { len: ta.length };

    const dotsTrace = {
        type: 'scatterternary', mode: 'markers',
        a: ta, b: tb, c: tc,
        marker: { size: 6, color: colors, line: { width: 0 }, opacity: 0.85 },
        text: hover, hoverinfo: 'text',
        hoverlabel: { bgcolor: 'white', bordercolor: '#ddd', font: { size: 12, family: 'Inter, sans-serif', color: '#1a1a1a' }, align: 'left' },
        showlegend: false,
    };

    // Only plot centroid dots for configs that actually win somewhere
    const cfgA = [], cfgB = [], cfgC = [], cfgColors = [], cfgHover = [];
    for (const i of winners) {
        const sum = normed[i].nCost + normed[i].nLat + normed[i].nAcc || 1;
        cfgA.push(normed[i].nCost / sum);
        cfgB.push(normed[i].nLat / sum);
        cfgC.push(normed[i].nAcc / sum);
        cfgColors.push(rows[i].color);
        cfgHover.push(
            '<b>' + rows[i].name + '</b>' +
            '<br>' +
            '<br>Cost: ' + fmtDollarsMo(rows[i].totalCost) +
            '<br>Latency: ' + rows[i].latency_ms + ' ms' +
            '<br>Recall@10: ' + (rows[i].accuracy * 100).toFixed(1) + '%'
        );
    }
    const cfgDotsTrace = {
        type: 'scatterternary', mode: 'markers',
        a: cfgA, b: cfgB, c: cfgC,
        marker: { size: 12, color: cfgColors, line: { color: 'white', width: 3 }, opacity: 1 },
        text: cfgHover, hoverinfo: 'text',
        hoverlabel: { bgcolor: 'white', bordercolor: '#ddd', font: { size: 12, family: 'Inter, sans-serif', color: '#1a1a1a' }, align: 'left' },
        showlegend: false,
    };

    const medianTraces = [
        { a: [1, 0], b: [0, 0.5], c: [0, 0.5] },
        { a: [0, 0.5], b: [1, 0], c: [0, 0.5] },
        { a: [0, 0.5], b: [0, 0.5], c: [1, 0] },
    ].map(m => ({
        type: 'scatterternary', mode: 'lines',
        a: m.a, b: m.b, c: m.c,
        line: { color: 'rgba(0,0,0,0.45)', width: 2.5, dash: 'dot' },
        hoverinfo: 'skip', showlegend: false,
    }));

    const cornerBest = {
        type: 'scatterternary', mode: 'text',
        a: [1, 0.05, 0.05],
        b: [0, 0.975, 0],
        c: [0, 0, 0.975],
        text: [
            'Best: ' + fmtDollarsMo(minCost) + '<br><br>',
            'Best: ' + minLat + 'ms    ',
            '     Best: ' + (maxAcc * 100).toFixed(1) + '%',
        ],
        textfont: { size: 11, color: '#059669', family: 'Inter, sans-serif' },
        textposition: ['top center', 'bottom left', 'bottom right'],
        hoverinfo: 'skip', showlegend: false, cliponaxis: false,
    };

    const midWorst = {
        type: 'scatterternary', mode: 'text',
        a: [0, 0.5, 0.5],
        b: [0.5, 0, 0.5],
        c: [0.5, 0.5, 0],
        text: [
            '<br>Worst: ' + fmtDollarsMo(maxCost),
            '    Worst: ' + maxLat + 'ms',
            'Worst: ' + (minAcc * 100).toFixed(1) + '%    ',
        ],
        textfont: { size: 10, color: '#dc2626', family: 'Inter, sans-serif' },
        textposition: ['bottom center', 'middle right', 'middle left'],
        hoverinfo: 'skip', showlegend: false, cliponaxis: false,
    };

    // Legend only for winners
    const legendTraces = [...winners].map(i => ({
        type: 'scatterternary', mode: 'markers',
        a: [null], b: [null], c: [null],
        marker: { size: 10, color: rows[i].color },
        name: rows[i].name, showlegend: true, hoverinfo: 'skip',
    }));

    Plotly.react(divId,
        [dotsTrace, ...medianTraces, cornerBest, midWorst, ...legendTraces],
        {
            ternary: {
                sum: 1,
                aaxis: { title: { text: '<b>Cost Priority</b>', font: { size: 12, color: '#444' } }, min: 0, linewidth: 2, gridcolor: 'rgba(0,0,0,0.06)', showticklabels: false },
                baxis: { title: { text: '<b>Latency Priority</b>', font: { size: 12, color: '#444' } }, min: 0, linewidth: 2, gridcolor: 'rgba(0,0,0,0.06)', showticklabels: false },
                caxis: { title: { text: '<b>Accuracy Priority</b>', font: { size: 12, color: '#444' } }, min: 0, linewidth: 2, gridcolor: 'rgba(0,0,0,0.06)', showticklabels: false },
                bgcolor: '#fafafa',
            },
            font: { family: 'Inter, -apple-system, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 80, b: 80, l: 80, r: 80 },
            legend: {
                orientation: 'h',
                x: 0.5,
                y: -0.05,
                xanchor: 'center',
                yanchor: 'top',
                font: { size: 11 },
                itemsizing: 'constant',
                valign: 'middle',
                itemwidth: 30,
            },
            showlegend: true,
        },
        { displayModeBar: false, responsive: true }
    );
    // Update toggle chips: mark dominated configs
    const winnerIds = new Set([...winners].map(i => rows[i].id));
    for (const chip of toggleContainer.children) {
        const cid = chip.dataset.configId;
        if (!cid) continue;
        const isActive = ternaryToggles[secKey].has(cid);
        const isWinner = winnerIds.has(cid);
        chip.classList.toggle('dominated', isActive && !isWinner);
        // Add/remove "(dominated)" tag
        let domTag = chip.querySelector('.dom-tag');
        if (isActive && !isWinner) {
            if (!domTag) {
                domTag = document.createElement('span');
                domTag.className = 'dom-tag';
                domTag.textContent = 'dominated';
                chip.appendChild(domTag);
            }
        } else if (domTag) {
            domTag.remove();
        }
    }

    const el = document.getElementById(divId);
    el.on('plotly_hover', function(data) {
        if (data.points[0].curveNumber !== 0) return;
        const idx = data.points[0].pointNumber;
        const len = ternaryState[divId].len;
        const sizes = new Array(len).fill(6);
        sizes[idx] = 14;
        const opacities = new Array(len).fill(0.85);
        opacities[idx] = 1;
        Plotly.restyle(divId, { 'marker.size': [sizes], 'marker.opacity': [opacities] }, [0]);
    });
    el.on('plotly_unhover', function() {
        const len = ternaryState[divId].len;
        Plotly.restyle(divId, {
            'marker.size': [new Array(len).fill(6)],
            'marker.opacity': [new Array(len).fill(0.85)],
        }, [0]);
    });
}
// ── Scatter ──
function renderScatter() {
    const rows = getFilteredRows();
    const pts1 = rows.map(r => ({ x: r.totalCost, y: r.accuracy, label: r.name, color: r.color }));
    const pts2 = rows.map(r => ({ x: r.latency_ms, y: r.accuracy, label: r.name, color: r.color }));

    const mkDs = (pts) => [{ data: pts.map(p => ({ x: p.x, y: p.y })), backgroundColor: pts.map(p => p.color), borderColor: pts.map(p => p.color), pointRadius: 8, pointHoverRadius: 11 }];
    const mkTt = (pts) => ({
        itemSort: (a, b) => b.parsed.y - a.parsed.y,
        callbacks: {
            label: (item) => {
                const p = pts[item.dataIndex];
                return p.label + ': (' + (p.x > 100 ? fmtDollars(p.x) : p.x) + ', ' + (p.y * 100).toFixed(1) + '%)';
            }
        }
    });
    const mkOpts = (pts, xLabel) => ({
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 400, easing: 'easeInOutCubic' },
        plugins: { legend: { display: false }, tooltip: mkTt(pts) },
        scales: {
            x: { title: { display: true, text: xLabel, font: { size: 12 } } },
            y: { title: { display: true, text: 'Recall@10', font: { size: 12 } }, min: 0, max: 1 },
        },
    });

    if (scatterChart1) {
        scatterChart1.data.datasets = mkDs(pts1);
        scatterChart1.options = mkOpts(pts1, 'Monthly Cost ($)');
        scatterChart1.update();
    } else {
        scatterChart1 = new Chart(document.getElementById('scatterCostAcc'), {
            type: 'scatter', data: { datasets: mkDs(pts1) }, options: mkOpts(pts1, 'Monthly Cost ($)'),
        });
    }
    if (scatterChart2) {
        scatterChart2.data.datasets = mkDs(pts2);
        scatterChart2.options = mkOpts(pts2, 'Latency (ms)');
        scatterChart2.update();
    } else {
        scatterChart2 = new Chart(document.getElementById('scatterLatAcc'), {
            type: 'scatter', data: { datasets: mkDs(pts2) }, options: mkOpts(pts2, 'Latency (ms)'),
        });
    }
}

// ── Table ──
function renderTable() {
    const rows = getFilteredRows();
    const sortFn = (a, b) => {
        let va = a[sortCol], vb = b[sortCol];
        if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
        return sortAsc ? (va < vb ? -1 : va > vb ? 1 : 0) : (va > vb ? -1 : va < vb ? 1 : 0);
    };
    const zeRows = rows.filter(r => r.group === 'zembed-1').sort(sortFn);
    const compRows = rows.filter(r => r.group !== 'zembed-1').sort(sortFn);
    const grouped = [...zeRows, ...compRows];
    const cheapest = rows.length > 0 ? rows.reduce((a, b) => a.totalCost < b.totalCost ? a : b) : null;

    let html = '', lastGroup = '';
    for (const r of grouped) {
        if (r.group !== lastGroup) {
            html += '<tr class="section-label"><td colspan="11">' + r.group + '</td></tr>';
            lastGroup = r.group;
        }
        const isZe = r.group === 'zembed-1';
        const sav = cheapest ? ((r.totalCost - cheapest.totalCost) / cheapest.totalCost * 100) : 0;
        const savTag = r === cheapest ? '<span class="tag tag-green">cheapest</span>'
            : sav <= 0 ? '<span class="tag tag-green">' + sav.toFixed(0) + '%</span>'
            : '<span class="tag tag-red">+' + sav.toFixed(0) + '%</span>';
        const secTag = r.security === 'vpc' ? ' <span class="tag tag-blue">VPC</span>' : '';
        html += '<tr class="' + (isZe ? 'is-ze' : '') + '">'
            + '<td><span class="model-dot" style="background:' + r.color + '"></span>' + r.name + '</td>'
            + '<td>' + r.infra + secTag + '</td>'
            + '<td class="num">' + r.dims + '</td>'
            + '<td>' + r.quantization + '</td>'
            + '<td class="num">' + fmtStorage(r.storageBytes) + '</td>'
            + '<td class="num">' + fmtDollars(r.embedCost) + '</td>'
            + '<td class="num">' + fmtDollars(r.queryCost) + '</td>'
            + '<td class="num">' + fmtDollars(r.storageCost) + '</td>'
            + '<td class="num"><strong>' + fmtDollars(r.totalCost) + '</strong> ' + savTag + '</td>'
            + '<td class="num">' + (r.accuracy * 100).toFixed(1) + '%</td>'
            + '<td class="num">' + r.latency_ms + ' ms</td>'
            + '</tr>';
    }
    document.getElementById('tableBody').innerHTML = html;
}

// ── Main ──
function recalc() {
    recalcTernary();
    renderScatter();
    renderTable();
}

document.querySelectorAll('#resultTable thead th').forEach(th => {
    th.addEventListener('click', () => {
        const col = th.dataset.col;
        if (sortCol === col) sortAsc = !sortAsc;
        else { sortCol = col; sortAsc = true; }
        renderTable();
    });
});

initFilters();
initTernaryToggles();
recalc();
</script>"""


def html_foot() -> str:
    return """
</div>
</body>
</html>"""


def build_html(configs_json: str, recall_curves_json: str) -> str:
    return (
        html_head()
        + html_usage_inputs()
        + html_ternary()
        + html_scatterplots()
        + html_table()
        + html_script(configs_json, recall_curves_json)
        + html_foot()
    )


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    ck = cache_key(CHECKPOINTS)
    if ck.exists():
        with open(ck, "rb") as f:
            all_split_evals: dict[str, dict[str, list[QueryEvals]]] = pickle.load(f)
    else:
        all_split_evals = {}
        for cp in CHECKPOINTS:
            all_split_evals[cp] = await load_evals(cp)
        CACHE_DIR.mkdir(exist_ok=True)
        with open(ck, "wb") as f:
            pickle.dump(all_split_evals, f)

    all_evals: dict[str, list[QueryEvals]] = {
        cp: [e for evals in splits.values() for e in evals]
        for cp, splits in all_split_evals.items()
    }
    query_id_sets = [set(e.query_id for e in evals) for evals in all_evals.values()]
    common: set[str] = set.intersection(*query_id_sets) if query_id_sets else set()

    recall_curves: dict[str, list[float]] = {}
    for cp, evals in all_evals.items():
        filtered = [e for e in evals if e.query_id in common]
        if not filtered:
            continue
        per_query = np.array([
            recalls_at_all_k(decode_embedding(e.similarities), e.qrels, K_MAX)
            for e in filtered
        ])
        recall_curves[cp] = [round(float(v), 5) for v in per_query.mean(axis=0)]

    OUTPUT_PATH.write_text(build_html(json.dumps(CONFIGURATIONS), json.dumps(recall_curves)))


if __name__ == "__main__":
    asyncio.run(main())