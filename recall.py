"""
Generate self-contained HTML chart of Recall@K from S3 evals.

Usage:
    python -m ml.training.embedding.generate_recall_chart
"""

import asyncio
import json
import html as html_lib
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.ai import AIEmbedding, decode_embedding
from ml.s3_utils import s3_list_objects, s3_read_key
from ml.utils import async_iterate_with_prefetching, async_zip, unwrap, wrap_sem

CHECKPOINTS: list[str] = [
    "qwen3-embedding-4b-v0.2.3-biasadj2/epoch-004-step-1400",
    #"zembed-4b-v0.3.0-mix-v3.0/epoch-001-step-8450",
    "zembed-4b-v0.3.0-mix-v4.0/epoch-001-step-5450",
    "aimodel-embed/modal/voyage-4.zeroentropy.dev",
    "aimodel-embed/modal/zeroentropy--voyage-4-nano-model-endpoint.modal.run",
    "aimodel-embed/openai/text-embedding-3-large",
    "aimodel-embed/openai/text-embedding-3-small",
]

DISPLAY_NAMES: dict[str, str] = {
    "qwen3-embedding-4b-v0.2.3-biasadj2/epoch-004-step-1400": "Qwen3-Embedding-4B",
    "zembed-4b-v0.3.0-mix-v4.0/epoch-001-step-5450": "zembed-1",
    "aimodel-embed/modal/voyage-4.zeroentropy.dev": "Voyage 4",
    "aimodel-embed/modal/zeroentropy--voyage-4-nano-model-endpoint.modal.run": "Voyage 4 Nano",
    "aimodel-embed/openai/text-embedding-3-large": "OpenAI text-embedding-3-large",
    "aimodel-embed/openai/text-embedding-3-small": "OpenAI text-embedding-3-small",
}

COLORS: dict[str, str] = {
    "qwen3-embedding-4b-v0.2.3-biasadj2/epoch-004-step-1400": "#ff7d59",
    "zembed-4b-v0.3.0-mix-v4.0/epoch-001-step-5450": "#4f46e5",
    "aimodel-embed/modal/voyage-4.zeroentropy.dev": "#6366f1",
    "aimodel-embed/modal/zeroentropy--voyage-4-nano-model-endpoint.modal.run": "#a78bfa",
    "aimodel-embed/openai/text-embedding-3-large": "#10a37f",
    "aimodel-embed/openai/text-embedding-3-small": "#6ee7b7",
}

K_MAX = 100
DOWNLOAD_SEM = asyncio.Semaphore(64)
OUTPUT_PATH = Path("recall_chart.html")


from ml.training.embedding.evaluate import QueryEvals


async def load_evals(checkpoint: str) -> dict[str, list[QueryEvals]]:
    prefix = f"checkpoints/{checkpoint}/"
    eval_paths: list[str] = []
    async for path in s3_list_objects(prefix):
        if path.endswith("/evals.jsonl"):
            eval_paths.append(path)

    result: dict[str, list[QueryEvals]] = {}
    async for data, path in async_iterate_with_prefetching(
        [
            lambda p=path: async_zip(wrap_sem(s3_read_key(p), DOWNLOAD_SEM), p)
            for path in eval_paths
        ],
        max_concurrent=32,
    ):
        split_id = path.rsplit("/", 2)[-2]
        result[split_id] = [
            QueryEvals.model_validate_json(line)
            for line in unwrap(data).decode().strip().split("\n")
            if line.strip()
        ]
    return result


def recalls_at_all_k(
    similarities: AIEmbedding,
    qrels: dict[int, float],
    k_max: int,
) -> NDArray[np.float32]:
    total_relevancy = sum(qrels.values())
    assert total_relevancy > 0
    sorted_indices = np.argsort(-similarities, kind="stable")[:k_max]
    recalls = np.zeros(k_max, dtype=np.float32)
    hits = 0.0
    for k_idx, idx in enumerate(sorted_indices):
        hits += qrels.get(int(idx), 0.0)
        recalls[k_idx] = hits / total_relevancy
    if len(sorted_indices) < k_max:
        recalls[len(sorted_indices) :] = recalls[len(sorted_indices) - 1]
    return recalls


def build_html(
    checkpoint_recalls: dict[str, NDArray[np.float32]],
) -> str:
    k_values = list(range(1, K_MAX + 1))

    datasets: list[dict[str, object]] = []
    for cp in CHECKPOINTS:
        if cp not in checkpoint_recalls:
            continue
        datasets.append(
            {
                "label": DISPLAY_NAMES.get(cp, cp),
                "data": [round(float(v), 5) for v in checkpoint_recalls[cp]],
                "borderColor": COLORS.get(cp, "#888888"),
                "borderWidth": 2,
                "pointRadius": 0,
                "tension": 0.3,
            }
        )

    chart_data = json.dumps({"labels": k_values, "datasets": datasets})

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Recall@K — zembed-1 vs Competitors</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #fafafa;
        padding: 40px 20px;
    }
    .container {
        max-width: 960px;
        margin: 0 auto;
        background: white;
        border-radius: 16px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        padding: 32px 40px;
    }
    h2 {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .subtitle {
        font-size: 14px;
        color: #888;
        margin-bottom: 24px;
    }
    .chart-wrap {
        position: relative;
        height: 480px;
    }
    .controls {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .controls label {
        font-size: 13px;
        font-weight: 600;
        color: #555;
    }
    .controls select {
        padding: 6px 12px;
        border: 1px solid #ddd;
        border-radius: 6px;
        font-size: 13px;
        font-family: inherit;
        background: #fafafa;
    }
    .controls select:focus {
        outline: none;
        border-color: #4f46e5;
    }
    .mode-btn {
        padding: 6px 16px;
        border: 1px solid #ddd;
        border-radius: 6px;
        font-size: 13px;
        font-family: inherit;
        background: #fafafa;
        cursor: pointer;
        transition: all 0.15s;
    }
    .mode-btn:hover {
        background: #f0f0f0;
    }
    .mode-btn.active {
        background: #4f46e5;
        color: white;
        border-color: #4f46e5;
    }

    /* Title sections */
    .title-absolute, .title-delta {
        transition: opacity 0.3s, transform 0.3s;
    }
    .title-absolute.hidden, .title-delta.hidden {
        display: none;
    }

    /* Rotating word animation */
    .delta-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .word-rotator {
        display: inline-block;
        height: 1.3em;
        overflow: hidden;
        vertical-align: bottom;
        position: relative;
        min-width: 80px;
        text-align: center;
    }
    .word-rotator .word {
        display: block;
        height: 1.3em;
        line-height: 1.3em;
        font-weight: 800;
        position: absolute;
        width: 100%;
        left: 0;
        transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1),
                    opacity 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .word-rotator .word.worse {
        color: #dc2626;
    }
    .word-rotator .word.better {
        color: #059669;
    }
    /* States: visible, exiting-down, entering-from-top */
    .word-rotator .word.state-visible {
        transform: translateY(0);
        opacity: 1;
    }
    .word-rotator .word.state-exit {
        transform: translateY(110%);
        opacity: 0;
    }
    .word-rotator .word.state-above {
        transform: translateY(-110%);
        opacity: 0;
        transition: none; /* snap to top instantly */
    }

    .baseline-name {
        color: #4f46e5;
        font-weight: 800;
        font-style: italic;
    }
    

    /* Y-range flash indicator */
    .range-flash {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        border-radius: 8px;
        z-index: 10;
        opacity: 0;
        transition: opacity 0.15s;
    }
    .range-flash.active {
        opacity: 1;
        animation: flashPulse 0.6s ease-out forwards;
    }
    @keyframes flashPulse {
        0% {
            box-shadow: inset 0 0 0 3px rgba(79, 70, 229, 0.5);
            opacity: 1;
        }
        100% {
            box-shadow: inset 0 0 0 0px rgba(79, 70, 229, 0);
            opacity: 0;
        }
    }

    /* Range badge */
    .range-badge {
        position: absolute;
        top: 8px;
        right: 8px;
        background: #4f46e5;
        color: white;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        z-index: 11;
        opacity: 0;
        transform: translateY(-8px);
        transition: opacity 0.25s, transform 0.25s;
        pointer-events: none;
    }
    .range-badge.visible {
        opacity: 1;
        transform: translateY(0);
    }
</style>
</head>
<body>
<div class="container">
    <div class="title-absolute" id="titleAbsolute">
        <h2>Recall@K</h2>
        <p class="subtitle">Higher is better. Evaluated on ZeroEntropy internal benchmarks.</p>
    </div>
    <div class="title-delta hidden" id="titleDelta">
        <div class="delta-title">
            Which models are
            <span class="word-rotator" id="wordRotator">
                <span class="word worse state-visible" id="wordA">worse</span>
                <span class="word better state-above" id="wordB">better</span>
            </span>
            than
            <span class="baseline-name" id="baselineLabel"></span>?
        </div>
        <p class="subtitle">Δ Recall@K vs baseline. Above zero = better.</p>
    </div>
    <div class="controls">
        <button class="mode-btn active" id="btnAbsolute" onclick="setMode('absolute')">Absolute</button>
        <button class="mode-btn" id="btnDelta" onclick="setMode('delta')">Δ vs Baseline</button>
        <div id="baselineControl" style="display:none;">
            <label>Baseline: </label>
            <select id="baselineSelect" onchange="onBaselineChange()"></select>
        </div>
    </div>
    <div class="chart-wrap">
        <canvas id="recallChart"></canvas>
        <div class="range-flash" id="rangeFlash"></div>
        <div class="range-badge" id="rangeBadge"></div>
    </div>
</div>
<script>
const RAW_DATA = """ + chart_data + """;
const originalDatasets = RAW_DATA.datasets.map(d => ({ ...d, data: [...d.data] }));

let mode = 'absolute';
let chart;
let rotatorInterval = null;
let currentWord = 0; // 0 = worse showing, 1 = better showing
let prevYMin = 0;
let prevYMax = 1;

function initBaselineSelect() {
    const sel = document.getElementById('baselineSelect');
    originalDatasets.forEach((d, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = d.label;
        sel.appendChild(opt);
    });
}

function startRotator() {
    stopRotator();
    currentWord = 0;
    const wordA = document.getElementById('wordA');
    const wordB = document.getElementById('wordB');
    wordA.className = 'word worse state-visible';
    wordB.className = 'word better state-above';

    rotatorInterval = setInterval(() => {
        if (currentWord === 0) {
            // worse exits down, better enters from top
            wordA.className = 'word worse state-exit';
            wordB.className = 'word better state-above';
            // After snap-to-top (no transition), trigger enter
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    wordB.className = 'word better state-visible';
                });
            });
            currentWord = 1;
        } else {
            // better exits down, worse enters from top
            wordB.className = 'word better state-exit';
            wordA.className = 'word worse state-above';
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    wordA.className = 'word worse state-visible';
                });
            });
            currentWord = 0;
        }
    }, 2200);
}

function stopRotator() {
    if (rotatorInterval) {
        clearInterval(rotatorInterval);
        rotatorInterval = null;
    }
}

function flashRange(newMin, newMax) {
    // Flash border
    const flash = document.getElementById('rangeFlash');
    flash.classList.remove('active');
    void flash.offsetWidth; // force reflow
    flash.classList.add('active');

    // Show badge with new range
    const badge = document.getElementById('rangeBadge');
    const minStr = newMin >= 0 ? '+' + newMin.toFixed(3) : newMin.toFixed(3);
    const maxStr = newMax >= 0 ? '+' + newMax.toFixed(3) : newMax.toFixed(3);
    badge.textContent = 'y: [' + newMin.toFixed(3) + ', ' + newMax.toFixed(3) + ']';
    badge.classList.add('visible');

    setTimeout(() => {
        badge.classList.remove('visible');
    }, 1500);
}

function setMode(m) {
    mode = m;
    document.getElementById('btnAbsolute').classList.toggle('active', m === 'absolute');
    document.getElementById('btnDelta').classList.toggle('active', m === 'delta');
    document.getElementById('baselineControl').style.display = m === 'delta' ? '' : 'none';

    document.getElementById('titleAbsolute').classList.toggle('hidden', m !== 'absolute');
    document.getElementById('titleDelta').classList.toggle('hidden', m !== 'delta');

    if (m === 'delta') {
        startRotator();
    } else {
        stopRotator();
    }

    updateChart();
}

function onBaselineChange() {
    updateChart();
}

function updateChart() {
    const baseIdx = parseInt(document.getElementById('baselineSelect').value);
    const baseData = originalDatasets[baseIdx].data;

    document.getElementById('baselineLabel').textContent = originalDatasets[baseIdx].label;

    if (mode === 'delta') {
        let globalMin = 0;
        let globalMax = 0;
        chart.data.datasets.forEach((ds, i) => {
            const deltas = originalDatasets[i].data.map((v, k) => Math.round((v - baseData[k]) * 100000) / 100000);
            ds.data = deltas;
            ds.hidden = false;
            if (i === baseIdx) {
                // Baseline is the zero line — show it styled as reference
                ds.borderWidth = 3;
                ds.borderDash = [6, 4];
                ds.borderColor = originalDatasets[i].borderColor;
            } else {
                ds.borderWidth = 2;
                ds.borderDash = [];
                ds.borderColor = originalDatasets[i].borderColor;
                const lo = Math.min(...deltas);
                const hi = Math.max(...deltas);
                if (lo < globalMin) globalMin = lo;
                if (hi > globalMax) globalMax = hi;
            }
        });

        // Pad range by 10%
        const range = globalMax - globalMin || 0.01;
        const newMin = Math.floor((globalMin - range * 0.1) * 1000) / 1000;
        const newMax = Math.ceil((globalMax + range * 0.1) * 1000) / 1000;

        // Flash if range changed meaningfully
        if (Math.abs(newMin - prevYMin) > 0.001 || Math.abs(newMax - prevYMax) > 0.001) {
            flashRange(newMin, newMax);
        }

        chart.options.scales.y.min = newMin;
        chart.options.scales.y.max = newMax;
        chart.options.scales.y.title.text = 'Δ Recall@K';
        prevYMin = newMin;
        prevYMax = newMax;
    } else {
        chart.data.datasets.forEach((ds, i) => {
            ds.data = [...originalDatasets[i].data];
            ds.hidden = false;
            ds.borderWidth = 2;
            ds.borderDash = [];
            ds.borderColor = originalDatasets[i].borderColor;
        });
        chart.options.scales.y.min = 0;
        chart.options.scales.y.max = 1;
        chart.options.scales.y.title.text = 'Recall@K';
        prevYMin = 0;
        prevYMax = 1;
    }

    chart.update();
}

initBaselineSelect();

const ctx = document.getElementById('recallChart').getContext('2d');
chart = new Chart(ctx, {
    type: 'line',
    data: JSON.parse(JSON.stringify(RAW_DATA)),
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 600,
            easing: 'easeInOutCubic',
        },
        transitions: {
            active: {
                animation: {
                    duration: 300,
                }
            }
        },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    usePointStyle: true,
                    pointStyle: 'line',
                    padding: 20,
                    font: { size: 13 },
                },
            },
            tooltip: {
                itemSort: function(a, b) {
                    return b.parsed.y - a.parsed.y;
                },
                callbacks: {
                    title: function(items) {
                        return 'K = ' + items[0].label;
                    },
                    label: function(item) {
                        const prefix = mode === 'delta' ? (item.parsed.y >= 0 ? '+' : '') : '';
                        return item.dataset.label + ': ' + prefix + item.parsed.y.toFixed(4);
                    },
                },
            },
        },
        scales: {
            x: {
                title: { display: true, text: 'K', font: { size: 13 } },
                ticks: {
                    callback: function(val, idx) {
                        const v = idx + 1;
                        if (v === 1 || v % 10 === 0) return v;
                        return '';
                    },
                },
            },
            y: {
                title: { display: true, text: 'Recall@K', font: { size: 13 } },
                min: 0,
                max: 1,
            },
        },
    },
});
</script>
</body>
</html>"""

import hashlib
import pickle

CACHE_DIR = Path(".cache_evals")


def cache_key(checkpoints: list[str]) -> Path:
    h = hashlib.sha256(",".join(checkpoints).encode()).hexdigest()[:16]
    return CACHE_DIR / f"evals_{h}.pkl"


async def main() -> None:
    ck = cache_key(CHECKPOINTS)

    if ck.exists():
        with open(ck, "rb") as f:
            all_split_evals: dict[str, dict[str, list[QueryEvals]]] = pickle.load(f)
    else:
        all_split_evals = {}
        tasks = {cp: load_evals(cp) for cp in CHECKPOINTS}
        for cp, task in tasks.items():
            all_split_evals[cp] = await task
        CACHE_DIR.mkdir(exist_ok=True)
        with open(ck, "wb") as f:
            pickle.dump(all_split_evals, f)

    all_evals: dict[str, list[QueryEvals]] = {
        cp: [e for evals in splits.values() for e in evals]
        for cp, splits in all_split_evals.items()
    }

    query_id_sets = [set(e.query_id for e in evals) for evals in all_evals.values()]
    common: set[str] = set.intersection(*query_id_sets) if query_id_sets else set()

    checkpoint_recalls: dict[str, NDArray[np.float32]] = {}
    for cp, evals in all_evals.items():
        filtered = [e for e in evals if e.query_id in common]
        if not filtered:
            continue
        per_query = np.array(
            [
                recalls_at_all_k(decode_embedding(e.similarities), e.qrels, K_MAX)
                for e in filtered
            ]
        )
        checkpoint_recalls[cp] = per_query.mean(axis=0)

    OUTPUT_PATH.write_text(build_html(checkpoint_recalls))

if __name__ == "__main__":
    asyncio.run(main())