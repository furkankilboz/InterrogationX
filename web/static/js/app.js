'use strict';

let curData = null;
let curWIdx = 0;
let curFilter = 'ALL';
let flatPairs = [];

document.addEventListener('DOMContentLoaded', function () {
    loadCases();
    loadHistory();
});

/* ── Mode ──────────────────────────────────────── */
function setMode(m) {
    var pc = document.getElementById('pCase');
    var pp = document.getElementById('pCustom');
    var bc = document.getElementById('bCase');
    var bp = document.getElementById('bCustom');

    if (m === 'case') {
        pc.classList.remove('hidden');
        pp.classList.add('hidden');
        bc.classList.add('on');
        bp.classList.remove('on');
    } else {
        pc.classList.add('hidden');
        pp.classList.remove('hidden');
        bc.classList.remove('on');
        bp.classList.add('on');
    }
}

/* ── Cases ─────────────────────────────────────── */
async function loadCases() {
    try {
        var r = await fetch('/api/cases');
        var d = await r.json();
        var s = document.getElementById('caseSel');
        s.innerHTML = '<option value="">— select a case —</option>';

        (d.cases || []).forEach(function (c) {
            var o = document.createElement('option');
            o.value = c.case_id;
            o.textContent = 'Case ' + c.case_id + ': ' + c.description.substring(0, 65);
            s.appendChild(o);
        });

        s.addEventListener('change', function () {
            previewCase(s.value);
        });
    } catch (e) {
        console.error('loadCases:', e);
    }
}

async function previewCase(id) {
    if (!id) return;
    try {
        var r = await fetch('/api/case/' + id);
        var d = await r.json();
        showStmts(d.suspect_statement, d.witness_statements);
    } catch (e) {
        console.error('previewCase:', e);
    }
}

function showStmts(sus, wits) {
    document.getElementById('stmtCard').style.display = 'block';
    document.getElementById('dSus').textContent = sus;

    var w = document.getElementById('dWit');
    w.innerHTML = '';

    (wits || []).forEach(function (t, i) {
        var d = document.createElement('div');
        d.className = 'stmt';
        d.innerHTML = '<span class="rbadge wit">WITNESS ' + (i + 1) + '</span>'
            + '<p>' + esc(t) + '</p>';
        w.appendChild(d);
    });
}

/* ── Analyze ───────────────────────────────────── */
async function analyzeCase() {
    var id = document.getElementById('caseSel').value;
    if (!id) {
        alert('Select a case first.');
        return;
    }
    await runAnalysis('/api/analyze', { case_id: id });
}

async function analyzeCustom() {
    var sus = document.getElementById('inSuspect').value.trim();
    var wits = [
        document.getElementById('inW1').value.trim(),
        document.getElementById('inW2').value.trim(),
        document.getElementById('inW3').value.trim()
    ].filter(function (w) { return w.length > 0; });

    if (!sus) {
        alert('Enter the defendant statement.');
        return;
    }
    if (!wits.length) {
        alert('Enter at least one witness statement.');
        return;
    }
    showStmts(sus, wits);
    await runAnalysis('/api/analyze/custom', {
        suspect_statement: sus,
        witness_statements: wits
    });
}

async function runAnalysis(url, body) {
    setLoading(true);
    clearResults();
    try {
        var r = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!r.ok) {
            var e = await r.json();
            throw new Error(e.error || 'HTTP ' + r.status);
        }
        var d = await r.json();
        curData = d;
        renderAll(d);
        loadHistory();
    } catch (e) {
        alert('Analysis failed: ' + e.message);
        console.error(e);
    } finally {
        setLoading(false);
    }
}

/* ── Render ────────────────────────────────────── */
function renderAll(d) {
    renderMetrics(d);
    renderWTabs(d);
    buildFlat(d);
    renderPairs();
    renderHighlights(d);
    renderDf(d);
    if (d.witness_analyses && d.witness_analyses.length) {
        renderHeatmap(d.witness_analyses[0]);
    }
}

function renderMetrics(d) {
    var el = document.getElementById('sumCards');
    el.classList.remove('hidden');
    var vc = vrdClass(d.overall_verdict);

    el.innerHTML =
        '<div class="mcard v">'
        + '<div class="ml">Overall Verdict</div>'
        + '<div class="mv ' + vc + '">' + (d.overall_verdict || '—') + '</div>'
        + '</div>'
        + '<div class="mcard c">'
        + '<div class="ml">Contradiction Score</div>'
        + '<div class="mv" style="color:var(--red)">' + pct(d.overall_contradiction_score) + '</div>'
        + '<div class="ms">confidence-weighted across all witnesses</div>'
        + '</div>'
        + '<div class="mcard s">'
        + '<div class="ml">Corroboration Score</div>'
        + '<div class="mv" style="color:var(--green)">' + pct(d.overall_corroboration_score) + '</div>'
        + '<div class="ms">confidence-weighted across all witnesses</div>'
        + '</div>';
}

function renderWTabs(d) {
    var wrap = document.getElementById('wTabs');
    wrap.classList.remove('hidden');
    var bar = document.getElementById('wTabBar');
    var cont = document.getElementById('wTabContent');
    bar.innerHTML = '';
    cont.innerHTML = '';

    d.witness_analyses.forEach(function (wa, i) {
        var b = document.createElement('button');
        b.className = 'wtab' + (i === 0 ? ' on' : '');
        b.textContent = 'Witness ' + wa.witness_idx;
        b.onclick = function () { switchW(i, d); };
        bar.appendChild(b);

        var vc = vrdAbbr(wa.verdict);
        var p = document.createElement('div');
        p.className = 'wpanel' + (i === 0 ? ' on' : '');
        p.id = 'wp' + i;

        p.innerHTML =
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
            + '<span class="vpill ' + vc + '">' + wa.verdict + '</span>'
            + '<span class="muted">'
            + wa.contradictions.length + ' contradictions · '
            + wa.corroborations.length + ' corroborations · '
            + wa.partial_matches.length + ' partial'
            + '</span>'
            + '</div>'
            + '<div class="sbar-w">'
            + '<div class="sbar-l"><span>Contradiction</span><span>' + pct(wa.contradiction_score) + '</span></div>'
            + '<div class="sbar-t"><div class="sbar-f ctr" style="width:' + (wa.contradiction_score * 100) + '%"></div></div>'
            + '</div>'
            + '<div class="sbar-w">'
            + '<div class="sbar-l"><span>Corroboration</span><span>' + pct(wa.corroboration_score) + '</span></div>'
            + '<div class="sbar-t"><div class="sbar-f crr" style="width:' + (wa.corroboration_score * 100) + '%"></div></div>'
            + '</div>';

        cont.appendChild(p);
    });
}

function switchW(i, d) {
    curWIdx = i;
    document.querySelectorAll('.wtab').forEach(function (b, j) {
        b.classList.toggle('on', j === i);
    });
    document.querySelectorAll('.wpanel').forEach(function (p, j) {
        p.classList.toggle('on', j === i);
    });
    renderHeatmap(d.witness_analyses[i]);
}

/* ── Heatmap ───────────────────────────────────── */
function renderHeatmap(wa) {
    document.getElementById('hmSec').classList.remove('hidden');
    document.getElementById('hmLbl').textContent = 'Witness ' + wa.witness_idx;

    var m = wa.similarity_matrix;
    if (!m || !m.length) return;

    var R = m.length;
    var C = m[0].length;
    var SZ = 52;
    var PL = 10;
    var PT = 10;

    var cv = document.getElementById('hmCanvas');
    cv.width = C * SZ + PL;
    cv.height = R * SZ + PT;

    var ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);

    for (var r = 0; r < R; r++) {
        for (var c = 0; c < C; c++) {
            var v = m[r][c];
            ctx.fillStyle = hmColor(v);
            ctx.fillRect(PL + c * SZ, PT + r * SZ, SZ - 2, SZ - 2);

            ctx.fillStyle = v > .45 ? '#0B0E14' : '#D8DDE6';
            ctx.font = '11px Consolas';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(v.toFixed(2), PL + c * SZ + SZ / 2, PT + r * SZ + SZ / 2);
        }
    }
}

function hmColor(v) {
    v = Math.max(0, Math.min(1, v));
    if (v < .5) {
        return lerp('#0B0E14', '#4E8FE6', v * 2);
    } else {
        return lerp('#4E8FE6', '#2EC77B', (v - .5) * 2);
    }
}

function lerp(a, b, t) {
    function parse(s) {
        return [
            parseInt(s.slice(1, 3), 16),
            parseInt(s.slice(3, 5), 16),
            parseInt(s.slice(5, 7), 16)
        ];
    }
    var ca = parse(a);
    var cb = parse(b);
    var r = Math.round(ca[0] + (cb[0] - ca[0]) * t);
    var g = Math.round(ca[1] + (cb[1] - ca[1]) * t);
    var bl = Math.round(ca[2] + (cb[2] - ca[2]) * t);
    return 'rgb(' + r + ',' + g + ',' + bl + ')';
}

/* ── Pairs table ───────────────────────────────── */
function buildFlat(d) {
    flatPairs = [];
    d.witness_analyses.forEach(function (wa) {
        wa.pairwise_analysis.forEach(function (p) {
            flatPairs.push(Object.assign({}, p, { witness_idx: wa.witness_idx }));
        });
    });
}

function renderPairs() {
    document.getElementById('pairSec').classList.remove('hidden');

    var f = curFilter === 'ALL'
        ? flatPairs
        : flatPairs.filter(function (p) { return p.relationship === curFilter; });

    document.getElementById('pairBadge').textContent = f.length;
    var tb = document.getElementById('pairBody');
    tb.innerHTML = '';

    f.forEach(function (p) {
        var tr = document.createElement('tr');
        var confColor = p.confidence >= .6
            ? 'var(--red)'
            : (p.confidence >= .4 ? 'var(--orange)' : 'var(--text2)');

        tr.innerHTML =
            '<td>W' + p.witness_idx + '</td>'
            + '<td>' + esc(p.suspect_sentence) + '</td>'
            + '<td>' + esc(p.witness_sentence) + '</td>'
            + '<td><div class="sbar">'
            + '<div class="sbar-tk"><div class="sbar-fl" style="width:' + (p.similarity * 100) + '%"></div></div>'
            + '<span class="sbar-n">' + p.similarity.toFixed(2) + '</span>'
            + '</div></td>'
            + '<td style="color:' + confColor + ';font-weight:700;font-size:12px">'
            + (p.confidence * 100).toFixed(0) + '%</td>'
            + '<td><span class="rpill ' + p.relationship + '">'
            + p.relationship.replace('_', ' ') + '</span></td>'
            + '<td style="font-size:11px;color:var(--orange)">'
            + (p.reasons || []).join('; ') + '</td>';

        tb.appendChild(tr);
    });
}

function filterP(btn) {
    document.querySelectorAll('.fbtn').forEach(function (b) {
        b.classList.remove('on');
    });
    btn.classList.add('on');
    curFilter = btn.dataset.r;
    renderPairs();
}

/* ── Highlights ────────────────────────────────── */
function renderHighlights(d) {
    var contras = [];
    var corrs = [];

    d.witness_analyses.forEach(function (wa) {
        wa.contradictions.forEach(function (c) {
            contras.push(Object.assign({}, c, { witness_idx: wa.witness_idx }));
        });
        wa.corroborations.forEach(function (c) {
            corrs.push(Object.assign({}, c, { witness_idx: wa.witness_idx }));
        });
    });

    contras.sort(function (a, b) { return (b.confidence || 0) - (a.confidence || 0); });
    corrs.sort(function (a, b) { return b.similarity - a.similarity; });

    var cs = document.getElementById('topC');
    var cl = document.getElementById('cList');
    if (contras.length) {
        cs.classList.remove('hidden');
        cl.innerHTML = '';
        contras.slice(0, 6).forEach(function (c) {
            cl.appendChild(hlItem(c, 'ctr'));
        });
    }

    var ss = document.getElementById('topS');
    var sl = document.getElementById('sList');
    if (corrs.length) {
        ss.classList.remove('hidden');
        sl.innerHTML = '';
        corrs.slice(0, 6).forEach(function (c) {
            sl.appendChild(hlItem(c, 'crr'));
        });
    }
}

function hlItem(p, type) {
    var d = document.createElement('div');
    d.className = 'hitem ' + type;
    var confPct = ((p.confidence || 0) * 100).toFixed(0);
    var confCls = p.confidence >= .6
        ? 'color:var(--red)'
        : (p.confidence >= .4 ? 'color:var(--orange)' : 'color:var(--text2)');

    d.innerHTML =
        '<div class="hi-grid">'
        + '<div class="hi-s"><span class="role">Defendant</span>' + esc(p.suspect_sentence) + '</div>'
        + '<div class="hi-s"><span class="role">Witness ' + p.witness_idx + '</span>' + esc(p.witness_sentence) + '</div>'
        + '</div>'
        + '<div class="hi-m">'
        + '<span class="hi-sim">Similarity: ' + p.similarity.toFixed(3) + '</span>'
        + '<span class="hi-conf" style="' + confCls + ';background:var(--surface)">' + confPct + '% confidence</span>'
        + (p.reasons || []).map(function (r) {
            return '<span class="hi-r">' + esc(r) + '</span>';
        }).join('')
        + '</div>';

    return d;
}

/* ── Summary DataFrame ─────────────────────────── */
function renderDf(d) {
    var rows = d.witness_analyses.map(function (wa) {
        return {
            'Witness': 'Witness ' + wa.witness_idx,
            'Contradictions': wa.contradictions.length,
            'Corroborations': wa.corroborations.length,
            'Partial': wa.partial_matches.length,
            'Contradiction %': wa.contradiction_score,
            'Corroboration %': wa.corroboration_score,
            'Verdict': wa.verdict
        };
    });

    if (!rows.length) return;

    document.getElementById('dfSec').classList.remove('hidden');
    var ks = Object.keys(rows[0]);

    document.getElementById('dfHead').innerHTML =
        '<tr>' + ks.map(function (k) { return '<th>' + k + '</th>'; }).join('') + '</tr>';

    document.getElementById('dfBody').innerHTML = rows.map(function (row) {
        return '<tr>' + ks.map(function (k) {
            var v = row[k];
            if (k === 'Verdict') {
                var vc = vrdAbbr(v);
                return '<td><span class="vpill ' + vc + '">' + v + '</span></td>';
            }
            if (typeof v === 'number') {
                return '<td>' + (v.toFixed ? v.toFixed(3) : v) + '</td>';
            }
            return '<td>' + esc(String(v)) + '</td>';
        }).join('') + '</tr>';
    }).join('');
}

/* ── History ───────────────────────────────────── */
async function loadHistory() {
    try {
        var r = await fetch('/api/history');
        var d = await r.json();
        var el = document.getElementById('histList');

        if (!d.analyses || !d.analyses.length) {
            el.innerHTML = '<span class="muted">No history yet.</span>';
            return;
        }

        el.innerHTML = '';
        d.analyses.slice(0, 20).forEach(function (a) {
            var div = document.createElement('div');
            div.className = 'hist-i';
            var vc = vrdClass(a.overall_verdict);

            div.innerHTML =
                '<div class="hist-id">Case ' + a.case_id + '</div>'
                + '<div class="hist-v ' + vc + '">' + (a.overall_verdict || '—') + '</div>'
                + '<div class="hist-d">' + (a.created_at || '').slice(0, 16) + '</div>';

            el.appendChild(div);
        });
    } catch (e) {
        console.error(e);
    }
}

/* ── Utility ───────────────────────────────────── */
function setLoading(on) {
    document.getElementById('spinner').classList.toggle('hidden', !on);
    ['goBtn', 'goBtn2'].forEach(function (id) {
        var b = document.getElementById(id);
        if (b) b.disabled = on;
    });
}

function clearResults() {
    ['sumCards', 'wTabs', 'hmSec', 'pairSec', 'topC', 'topS', 'dfSec'].forEach(function (id) {
        document.getElementById(id).classList.add('hidden');
    });
    flatPairs = [];
}

function vrdClass(v) {
    if (!v) return 'vrd-I';
    if (v.includes('HIGHLY')) return 'vrd-HC';
    if (v.includes('STRONGLY')) return 'vrd-SC';
    if (v.includes('CONTRADICT')) return 'vrd-C';
    if (v.includes('CORROBOR')) return 'vrd-CO';
    return 'vrd-I';
}

function vrdAbbr(v) {
    if (!v) return 'I';
    if (v.includes('HIGHLY')) return 'HC';
    if (v.includes('STRONGLY')) return 'SC';
    if (v.includes('CONTRADICT')) return 'C';
    if (v.includes('CORROBOR')) return 'CO';
    return 'I';
}

function pct(v) {
    return (v != null ? (v * 100).toFixed(1) : '0.0') + '%';
}

function esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
