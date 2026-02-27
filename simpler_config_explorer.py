"""
Generate self-contained HTML for zembed-1 configuration recommender.

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

CHECKPOINTS: list[str] = [
    "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450",
    "aimodel-embed/modal/voyage-4.zeroentropy.dev",
    "aimodel-embed/modal/zeroentropy--voyage-4-nano-model-endpoint.modal.run",
    "aimodel-embed/openai/text-embedding-3-large",
    "aimodel-embed/openai/text-embedding-3-small",
]

CONFIGURATIONS: list[dict[str, object]] = [
    {"id": "zembed-1-f32-full", "name": "zembed-1", "quantization": "float32", "matryoshka": "full", "dims": 2048, "bits_per_dim": 32, "deployment": "api", "price_per_m_tokens": 0.06, "latency_ms": 45, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.0, "color": "#4f46e5"},
    {"id": "zembed-1-f32-half", "name": "zembed-1 (half dim)", "quantization": "float32", "matryoshka": "half", "dims": 1024, "bits_per_dim": 32, "deployment": "api", "price_per_m_tokens": 0.06, "latency_ms": 45, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.015, "color": "#4f46e5"},
    {"id": "zembed-1-f32-quarter", "name": "zembed-1 (quarter dim)", "quantization": "float32", "matryoshka": "quarter", "dims": 512, "bits_per_dim": 32, "deployment": "api", "price_per_m_tokens": 0.06, "latency_ms": 45, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.04, "color": "#4f46e5"},
    {"id": "zembed-1-f32-eighth", "name": "zembed-1 (eighth dim)", "quantization": "float32", "matryoshka": "eighth", "dims": 256, "bits_per_dim": 32, "deployment": "api", "price_per_m_tokens": 0.06, "latency_ms": 45, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.08, "color": "#4f46e5"},
    {"id": "zembed-1-int8-full", "name": "zembed-1 (int8)", "quantization": "int8", "matryoshka": "full", "dims": 2048, "bits_per_dim": 8, "deployment": "api", "price_per_m_tokens": 0.06, "latency_ms": 45, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.005, "color": "#4f46e5"},
    {"id": "zembed-1-int8-half", "name": "zembed-1 (int8, half dim)", "quantization": "int8", "matryoshka": "half", "dims": 1024, "bits_per_dim": 8, "deployment": "api", "price_per_m_tokens": 0.06, "latency_ms": 45, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.02, "color": "#4f46e5"},
    {"id": "zembed-1-bin-full", "name": "zembed-1 (binary)", "quantization": "binary", "matryoshka": "full", "dims": 2048, "bits_per_dim": 1, "deployment": "api", "price_per_m_tokens": 0.06, "latency_ms": 45, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.025, "color": "#4f46e5"},
    {"id": "zembed-1-bin-half", "name": "zembed-1 (binary, half dim)", "quantization": "binary", "matryoshka": "half", "dims": 1024, "bits_per_dim": 1, "deployment": "api", "price_per_m_tokens": 0.06, "latency_ms": 45, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.035, "color": "#4f46e5"},
    {"id": "zembed-1-f32-full-vpc-a10g", "name": "zembed-1", "quantization": "float32", "matryoshka": "full", "dims": 2048, "bits_per_dim": 32, "deployment": "vpc", "gpu": "A10G", "price_per_hour": 1.50, "max_tokens_per_sec": 50000, "latency_ms": 35, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.0, "color": "#4f46e5"},
    {"id": "zembed-1-f32-half-vpc-a10g", "name": "zembed-1 (half dim)", "quantization": "float32", "matryoshka": "half", "dims": 1024, "bits_per_dim": 32, "deployment": "vpc", "gpu": "A10G", "price_per_hour": 1.50, "max_tokens_per_sec": 50000, "latency_ms": 35, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.015, "color": "#4f46e5"},
    {"id": "zembed-1-int8-full-vpc-a10g", "name": "zembed-1 (int8)", "quantization": "int8", "matryoshka": "full", "dims": 2048, "bits_per_dim": 8, "deployment": "vpc", "gpu": "A10G", "price_per_hour": 1.50, "max_tokens_per_sec": 50000, "latency_ms": 35, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.005, "color": "#4f46e5"},
    {"id": "zembed-1-bin-half-vpc-a10g", "name": "zembed-1 (binary, half dim)", "quantization": "binary", "matryoshka": "half", "dims": 1024, "bits_per_dim": 1, "deployment": "vpc", "gpu": "A10G", "price_per_hour": 1.50, "max_tokens_per_sec": 50000, "latency_ms": 35, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.035, "color": "#4f46e5"},
    {"id": "zembed-1-f32-full-vpc-h100", "name": "zembed-1", "quantization": "float32", "matryoshka": "full", "dims": 2048, "bits_per_dim": 32, "deployment": "vpc", "gpu": "H100", "price_per_hour": 4.50, "max_tokens_per_sec": 200000, "latency_ms": 15, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.0, "color": "#4f46e5"},
    {"id": "zembed-1-f32-half-vpc-h100", "name": "zembed-1 (half dim)", "quantization": "float32", "matryoshka": "half", "dims": 1024, "bits_per_dim": 32, "deployment": "vpc", "gpu": "H100", "price_per_hour": 4.50, "max_tokens_per_sec": 200000, "latency_ms": 15, "checkpoint": "zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450", "accuracy_penalty": 0.015, "color": "#4f46e5"},
]

K_MAX = 100
DOWNLOAD_SEM = asyncio.Semaphore(64)
OUTPUT_PATH = Path("config_explorer.html")
CACHE_DIR = Path(".cache_evals")
RECALL_K_FOR_ACCURACY = 10


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


def build_html(configs_json: str, recall_curves_json: str) -> str:
    return _html_shell(configs_json, recall_curves_json)


def _html_shell(configs_json: str, recall_curves_json: str) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>zembed-1 configuration</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{
    font-family:'Inter',system-ui,sans-serif;
    background:#fafafa;
    color:#111;
    -webkit-font-smoothing:antialiased;
    -moz-osx-font-smoothing:grayscale;
    line-height:1.5;
}

.page{max-width:820px;margin:0 auto;padding:48px 24px 80px}

/* ── Breadcrumb trail ── */
.trail{
    min-height:20px;
    margin-bottom:24px;
    display:flex;
    gap:6px;
    flex-wrap:wrap;
    align-items:center;
}
.crumb{
    font-size:12px;
    font-weight:500;
    color:#666;
    letter-spacing:0.01em;
}
.crumb-sep{
    font-size:10px;
    color:#ccc;
    margin:0 2px;
}
.crumb-val{color:#111;font-weight:600}

/* ── Step (single visible at a time) ── */
.step-container{
    position:relative;
    overflow:hidden;
    min-height:120px;
}
.step{
    position:absolute;
    top:0;left:0;right:0;
    opacity:0;
    transform:translateX(40px);
    transition:opacity .3s ease-out,transform .3s ease-out;
    pointer-events:none;
}
.step.active{
    position:relative;
    opacity:1;
    transform:translateX(0);
    pointer-events:auto;
}
.step.exiting{
    opacity:0;
    transform:translateX(-40px);
}
.step-q{
    font-size:17px;
    font-weight:600;
    color:#111;
    margin-bottom:16px;
    letter-spacing:-0.01em;
}
.step-hint{
    font-size:12px;
    color:#999;
    margin-bottom:16px;
}
.pills{display:flex;gap:8px;flex-wrap:wrap}
.pill{
    padding:8px 20px;
    border:1px solid #ddd;
    border-radius:6px;
    font-size:13px;
    font-weight:500;
    font-family:inherit;
    background:#fff;
    color:#333;
    cursor:pointer;
    transition:border-color .15s,background .15s,color .15s;
    line-height:1.4;
    user-select:none;
}
.pill:hover{border-color:#999;background:#f5f5f5}
.pill.selected{
    border-color:#4f46e5;
    background:#4f46e5;
    color:#fff;
}

/* ── Recommendation ── */
.reco{
    margin-top:40px;
    padding:24px 0;
    border-top:1px solid #e5e5e5;
    opacity:0;
    transform:translateY(8px);
    transition:opacity .35s ease-out,transform .35s ease-out;
    pointer-events:none;
}
.reco.visible{opacity:1;transform:translateY(0);pointer-events:auto}
.reco-label{
    font-size:11px;
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:0.08em;
    color:#4f46e5;
    margin-bottom:8px;
}
.reco-name{
    font-size:22px;
    font-weight:700;
    letter-spacing:-0.02em;
    margin-bottom:2px;
}
.reco-desc{font-size:13px;color:#666;margin-bottom:20px}
.reco-stats{display:flex;gap:40px;flex-wrap:wrap}
.reco-stat-val{
    font-size:20px;
    font-weight:700;
    font-family:'JetBrains Mono','SF Mono',monospace;
    letter-spacing:-0.02em;
}
.reco-stat-label{
    font-size:11px;
    color:#999;
    font-weight:500;
    letter-spacing:0.02em;
}

/* ── Reset ── */
.reset-row{
    margin-top:20px;
    opacity:0;
    transition:opacity .3s;
    pointer-events:none;
}
.reset-row.visible{opacity:1;pointer-events:auto}
.reset-link{
    font-size:12px;
    color:#999;
    cursor:pointer;
    text-decoration:none;
    font-weight:500;
    border:none;
    background:none;
    font-family:inherit;
    padding:0;
}
.reset-link:hover{color:#4f46e5}

/* ── Table ── */
.results{
    margin-top:32px;
    opacity:0;
    transform:translateY(10px);
    transition:opacity .4s ease-out .1s,transform .4s ease-out .1s;
    pointer-events:none;
}
.results.visible{opacity:1;transform:translateY(0);pointer-events:auto}
.tbl{width:100%;border-collapse:collapse}
.tbl thead th{
    text-align:left;
    padding:8px 10px;
    font-size:10px;
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:0.06em;
    color:#aaa;
    border-bottom:1px solid #e5e5e5;
    cursor:pointer;
    user-select:none;
    white-space:nowrap;
}
.tbl thead th:hover{color:#4f46e5}
.tbl thead th.r{text-align:right}
.tbl tbody td{
    padding:10px 10px;
    font-size:13px;
    border-bottom:1px solid #f0f0f0;
    transition:background .2s;
}
.tbl tbody td.r{
    text-align:right;
    font-family:'JetBrains Mono','SF Mono',monospace;
    font-size:12px;
    font-variant-numeric:tabular-nums;
}
.tbl tbody tr{transition:background .2s}
.tbl tbody tr:hover{background:#f8f8f8}
.tbl tbody tr.is-reco{border-left:3px solid #4f46e5;background:#f9f8ff}
.tbl tbody tr.is-reco:hover{background:#f3f1ff}

/* ── Scatter ── */
.scatter-row{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:16px;
    margin-bottom:24px;
}
.scatter-wrap{
    height:260px;
    background:#fff;
    border:1px solid #eee;
    border-radius:8px;
    padding:12px;
}
.scatter-wrap canvas{max-height:236px}

@media(max-width:640px){
    .scatter-row{display:none}
    .reco-stats{gap:24px}
}
</style>
</head>
<body>
<div class="page">

<div class="trail" id="trail"></div>
<div class="step-container" id="stepContainer"></div>

<div class="reco" id="reco">
    <div class="reco-label">Recommended</div>
    <div class="reco-name" id="recoName"></div>
    <div class="reco-desc" id="recoDesc"></div>
    <div class="reco-stats">
        <div><div class="reco-stat-val" id="recoAcc"></div><div class="reco-stat-label">Recall@10</div></div>
        <div><div class="reco-stat-val" id="recoLat"></div><div class="reco-stat-label">Latency</div></div>
        <div><div class="reco-stat-val" id="recoDims"></div><div class="reco-stat-label">Dimensions</div></div>
        <div><div class="reco-stat-val" id="recoStor"></div><div class="reco-stat-label">per 1M docs</div></div>
    </div>
</div>

<div class="reset-row" id="resetRow">
    <button class="reset-link" onclick="reset()">Start over</button>
</div>

<div class="results" id="results">
    <div class="scatter-row">
        <div class="scatter-wrap"><canvas id="sc1"></canvas></div>
        <div class="scatter-wrap"><canvas id="sc2"></canvas></div>
    </div>
    <table class="tbl" id="tbl">
        <thead><tr>
            <th data-col="name">Configuration</th>
            <th data-col="dims" class="r">Dims</th>
            <th data-col="quantization">Quant</th>
            <th data-col="storageBytes" class="r">Storage</th>
            <th data-col="latency_ms" class="r">Latency</th>
            <th data-col="accuracy" class="r">Recall@10</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
    </table>
</div>

</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
const CONFIGS = """ + configs_json + """;
const CURVES = """ + recall_curves_json + """;
const RK = """ + str(RECALL_K_FOR_ACCURACY) + """;

const STEPS = {
    deploy:  {q:'Where does it run?', choices:[
        {val:'api', label:'ZeroEntropy API'},
        {val:'vpc', label:'Private VPC'},
    ]},
    gpu:     {q:'Which GPU?', hint:'Determines throughput and per-hour cost.', choices:[
        {val:'A10G', label:'A10G'},
        {val:'H100', label:'H100'},
    ]},
    latency: {q:'Latency budget?', choices:[
        {val:'tight', label:'< 20 ms'},
        {val:'normal', label:'< 50 ms'},
        {val:'relaxed', label:'No constraint'},
    ]},
    storage: {q:'Storage priority?', choices:[
        {val:'max', label:'Max accuracy'},
        {val:'balanced', label:'Balanced'},
        {val:'min', label:'Minimize'},
    ]},
};

let answers = {};
let flow = [];
let flowIdx = 0;
let sortCol = 'accuracy', sortAsc = false;
let c1 = null, c2 = null;
let recoId = null;

function acc(c) {
    if (c.accuracy_override !== undefined) return c.accuracy_override;
    if (c.checkpoint && CURVES[c.checkpoint]) return Math.max(0, CURVES[c.checkpoint][RK-1] - (c.accuracy_penalty||0));
    return 0;
}
function stor(c) { return 1e6 * c.dims * c.bits_per_dim / 8; }
function fmtB(b) {
    const g = b / 1073741824;
    return g >= 1000 ? (g/1000).toFixed(1)+' TB' : g >= 1 ? g.toFixed(1)+' GB' : (g*1024).toFixed(0)+' MB';
}

function buildFlow() {
    flow = ['deploy'];
    if (answers.deploy === 'vpc') flow.push('gpu');
    flow.push('latency', 'storage');
}

function renderTrail() {
    const el = document.getElementById('trail');
    let html = '';
    for (let i = 0; i < flowIdx; i++) {
        const key = flow[i];
        const choice = STEPS[key].choices.find(c => c.val === answers[key]);
        if (i > 0) html += '<span class="crumb-sep">/</span>';
        html += '<span class="crumb"><span class="crumb-val">' + (choice ? choice.label : '') + '</span></span>';
    }
    el.innerHTML = html;
}

function renderStep() {
    const container = document.getElementById('stepContainer');
    const prev = container.querySelector('.step.active');

    if (flowIdx >= flow.length) {
        if (prev) { prev.classList.add('exiting'); prev.classList.remove('active'); setTimeout(() => prev.remove(), 300); }
        resolve();
        return;
    }

    const key = flow[flowIdx];
    const step = STEPS[key];
    const div = document.createElement('div');
    div.className = 'step';
    let html = '<div class="step-q">' + step.q + '</div>';
    if (step.hint) html += '<div class="step-hint">' + step.hint + '</div>';
    html += '<div class="pills">';
    for (const ch of step.choices) {
        html += '<button class="pill" data-key="' + key + '" data-val="' + ch.val + '">' + ch.label + '</button>';
    }
    html += '</div>';
    div.innerHTML = html;
    container.appendChild(div);

    div.querySelectorAll('.pill').forEach(btn => {
        btn.addEventListener('click', () => pick(btn.dataset.key, btn.dataset.val, btn));
    });

    if (prev) {
        prev.classList.add('exiting');
        prev.classList.remove('active');
        setTimeout(() => prev.remove(), 300);
    }
    requestAnimationFrame(() => requestAnimationFrame(() => div.classList.add('active')));
}

function pick(key, val, btn) {
    answers[key] = val;
    btn.parentElement.querySelectorAll('.pill').forEach(p => p.classList.remove('selected'));
    btn.classList.add('selected');

    setTimeout(() => {
        buildFlow();
        flowIdx = flow.indexOf(key) + 1;
        renderTrail();
        renderStep();
    }, 150);
}

function filtered() {
    return CONFIGS.filter(c => {
        if (answers.deploy === 'api' && c.deployment !== 'api') return false;
        if (answers.deploy === 'vpc' && c.deployment !== 'vpc') return false;
        if (answers.deploy === 'vpc' && answers.gpu && c.gpu !== answers.gpu) return false;
        return true;
    }).map(c => ({...c, accuracy: acc(c), storageBytes: stor(c)}));
}

function meetsLat(c) {
    if (answers.latency === 'tight') return c.latency_ms < 20;
    if (answers.latency === 'normal') return c.latency_ms < 50;
    return true;
}

function resolve() {
    const all = filtered();
    const eligible = all.filter(meetsLat);
    const pool = eligible.length > 0 ? eligible : all;

    if (answers.storage === 'min') pool.sort((a,b) => a.storageBytes - b.storageBytes || b.accuracy - a.accuracy);
    else if (answers.storage === 'max') pool.sort((a,b) => b.accuracy - a.accuracy);
    else pool.sort((a,b) => (b.accuracy - b.storageBytes/1e13) - (a.accuracy - a.storageBytes/1e13));

    const best = pool[0];
    recoId = best.id;

    const qd = best.quantization === 'float32' ? 'Full precision' : best.quantization === 'int8' ? '8-bit quantized' : 'Binary quantized';
    const dd = best.deployment === 'api' ? 'ZeroEntropy API' : best.gpu + ' VPC';
    document.getElementById('recoName').textContent = best.name;
    document.getElementById('recoDesc').textContent = qd + ', ' + best.dims + ' dimensions, ' + dd;
    document.getElementById('recoAcc').textContent = (best.accuracy * 100).toFixed(1) + '%';
    document.getElementById('recoLat').textContent = best.latency_ms + ' ms';
    document.getElementById('recoDims').textContent = String(best.dims);
    document.getElementById('recoStor').textContent = fmtB(best.storageBytes);

    document.getElementById('reco').classList.add('visible');
    document.getElementById('resetRow').classList.add('visible');
    setTimeout(() => {
        document.getElementById('results').classList.add('visible');
        renderTable(all);
        renderScatter(all);
    }, 150);
}

function renderTable(rows) {
    const sorted = [...rows].sort((a,b) => {
        let va = a[sortCol], vb = b[sortCol];
        if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
        return sortAsc ? (va < vb ? -1 : 1) : (va > vb ? -1 : 1);
    });
    // Move reco to top
    const ri = sorted.findIndex(r => r.id === recoId);
    if (ri > 0) { const [r] = sorted.splice(ri, 1); sorted.unshift(r); }

    let html = '';
    for (const r of sorted) {
        const cls = r.id === recoId ? 'is-reco' : '';
        html += '<tr class="'+cls+'">'
            + '<td>'+r.name+'</td>'
            + '<td class="r">'+r.dims+'</td>'
            + '<td>'+r.quantization+'</td>'
            + '<td class="r">'+fmtB(r.storageBytes)+'</td>'
            + '<td class="r">'+r.latency_ms+' ms</td>'
            + '<td class="r">'+(r.accuracy*100).toFixed(1)+'%</td>'
            + '</tr>';
    }
    document.getElementById('tbody').innerHTML = html;
}

function renderScatter(rows) {
    const DOT = '#bbb';
    const RECO = '#4f46e5';
    const pts = rows.map(r => ({
        sg: r.storageBytes / 1073741824,
        lat: r.latency_ms,
        y: r.accuracy,
        label: r.name + ' (' + r.quantization + ')',
        color: r.id === recoId ? RECO : DOT,
        r: r.id === recoId ? 8 : 5,
        bw: r.id === recoId ? 2.5 : 0,
        bc: r.id === recoId ? '#fff' : 'transparent',
    }));

    const mkOpts = (xLabel) => ({
        responsive:true, maintainAspectRatio:false,
        animation:{duration:400,easing:'easeOutCubic'},
        plugins:{legend:{display:false},tooltip:{callbacks:{
            label:(i) => { const p=pts[i.dataIndex]; return p.label+': '+(p.y*100).toFixed(1)+'%'; }
        }}},
        scales:{
            x:{title:{display:true,text:xLabel,font:{size:11,family:'Inter'}},grid:{color:'#f0f0f0'}},
            y:{title:{display:true,text:'Recall@10',font:{size:11,family:'Inter'}},min:0,max:1,grid:{color:'#f0f0f0'}},
        },
    });
    const mkDs = (xfn) => [{
        data:pts.map(p => ({x:xfn(p),y:p.y})),
        backgroundColor:pts.map(p=>p.color),
        pointRadius:pts.map(p=>p.r),
        pointBorderColor:pts.map(p=>p.bc),
        pointBorderWidth:pts.map(p=>p.bw),
    }];

    if(c1){c1.data.datasets=mkDs(p=>p.sg);c1.options=mkOpts('Storage / 1M docs (GB)');c1.update();}
    else c1=new Chart(document.getElementById('sc1'),{type:'scatter',data:{datasets:mkDs(p=>p.sg)},options:mkOpts('Storage / 1M docs (GB)')});

    if(c2){c2.data.datasets=mkDs(p=>p.lat);c2.options=mkOpts('Latency (ms)');c2.update();}
    else c2=new Chart(document.getElementById('sc2'),{type:'scatter',data:{datasets:mkDs(p=>p.lat)},options:mkOpts('Latency (ms)')});
}

function reset() {
    answers = {}; flow = []; flowIdx = 0; recoId = null;
    document.getElementById('reco').classList.remove('visible');
    document.getElementById('results').classList.remove('visible');
    document.getElementById('resetRow').classList.remove('visible');
    document.getElementById('trail').innerHTML = '';
    const container = document.getElementById('stepContainer');
    const prev = container.querySelector('.step');
    if (prev) { prev.classList.add('exiting'); prev.classList.remove('active'); setTimeout(() => prev.remove(), 300); }
    buildFlow();
    setTimeout(renderStep, 100);
}

document.querySelectorAll('#tbl thead th').forEach(th => {
    th.addEventListener('click', () => {
        const col = th.dataset.col;
        if (sortCol === col) sortAsc = !sortAsc;
        else { sortCol = col; sortAsc = col === 'name' || col === 'quantization'; }
        renderTable(filtered());
    });
});

buildFlow();
renderStep();
</script>
</body>
</html>"""


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