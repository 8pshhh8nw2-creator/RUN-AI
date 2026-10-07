"""Hero generativi per RUNAI: particelle di luce 3D su canvas (zero dipendenze, niente CDN).

Uso nelle pagine Streamlit (al posto di image_url=SVG_xxx):

    from hero_engine import header_block
    header_block("TELEMETRIA", "Il tuo allenamento", "Dati che corrono", scene="home")

Scene disponibili: home, analisi, stats, kpi, ml, plan, cv
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

SCENES = ("home", "analisi", "stats", "kpi", "ml", "plan", "cv")
_JS = Path(__file__).with_name("hero_engine.js").read_text(encoding="utf-8")

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%;background:transparent;overflow:hidden}
canvas{display:block;width:100%;height:100%}</style></head>
<body><canvas id="c"></canvas><script>window.SCENE="__SCENE__";</script><script>__JS__</script></body></html>"""


def hero_html(scene="home"):
    return _PAGE.replace("__SCENE__", scene).replace("__JS__", _JS)


def hero(scene="home", height=460):
    """Rende un hero animato (iframe trasparente: si fonde con lo sfondo della pagina)."""
    components.html(hero_html(scene), height=height)


def header_block(kicker, title, subtitle, scene="home", height=460, **_legacy):
    """Sostituto drop-in del vecchio header_block: stessi parametri testuali, scena animata a destra."""
    st.markdown("<div class='telemetry-bar'></div>", unsafe_allow_html=True)
    col_txt, col_img = st.columns([1.0, 1.6])
    with col_txt:
        st.markdown(f"""
        <div class="app-header">
            <div class="app-kicker"><span class="dot"></span>{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_img:
        hero(scene, height)


def preview_html():
    """Pagina unica con selettore di scena, per guardare tutto nel browser."""
    btns = "".join(f'<button onclick="setScene(\'{s}\')">{s}</button>' for s in SCENES)
    return (_PAGE.replace("__SCENE__", "home").replace("__JS__", _JS)
            .replace("<body>", "<body style='background:#04070D'>"
                     "<style>.bar{position:fixed;top:10px;left:10px;display:flex;gap:6px;z-index:9}"
                     ".bar button{background:#0b1730;color:#7EC8FF;border:1px solid #2F8FE0;border-radius:8px;"
                     "padding:6px 12px;font:600 12px Inter,sans-serif;cursor:pointer}"
                     "html,body{background:#04070D!important}</style>"
                     f"<div class='bar'>{btns}</div>"))


if __name__ == "__main__":
    Path("hero_preview.html").write_text(preview_html(), encoding="utf-8")
    print("scritto hero_preview.html")




/* RUNAI hero engine — particelle di luce 3D su canvas, zero dipendenze. */
(() => {
const C = document.getElementById('c'), X = C.getContext('2d');
let W = 0, H = 0, DPR = 1, T = 0, last = performance.now();
const M = { x: 0, y: 0, tx: 0, ty: 0 };
addEventListener('mousemove', e => { M.tx = e.clientX / innerWidth - .5; M.ty = e.clientY / innerHeight - .5; });
const cam = { yaw: 0, pitch: .5, d: 800, f: 800, s: 1, cx: .5, cy: .5 };

function resize() {
  DPR = Math.min(2, window.devicePixelRatio || 1);
  W = innerWidth || 900; H = innerHeight || 500;
  C.width = W * DPR; C.height = H * DPR;
  X.setTransform(DPR, 0, 0, DPR, 0, 0);
  cam.s = Math.min(W / 900, H / 520);
}
addEventListener('resize', resize);

// ---------- utilità ----------
function rng(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let q = Math.imul(a ^ a >>> 15, 1 | a);
  q = q + Math.imul(q ^ q >>> 7, 61 | q) ^ q; return ((q ^ q >>> 14) >>> 0) / 4294967296; }; }
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const ease = v => 1 - Math.pow(1 - clamp(v), 3);
const mod1 = v => ((v % 1) + 1) % 1;
const lerp = (a, b, f) => a + (b - a) * f;

function P(x, y, z) {
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
  const k = cam.f / (cam.f + z2 + cam.d);
  return [W * cam.cx + x1 * k * cam.s, H * cam.cy + y1 * k * cam.s, k, z2];
}

const RAMP = [[0, [10, 60, 170]], [.35, [30, 130, 235]], [.65, [0, 229, 255]], [1, [235, 250, 255]]];
function ramp(v) {
  v = clamp(v);
  for (let i = 0; i < RAMP.length - 1; i++) {
    const [a, ca] = RAMP[i], [b, cb] = RAMP[i + 1];
    if (v <= b) { const f = (v - a) / (b - a); return [0, 1, 2].map(j => Math.round(ca[j] + (cb[j] - ca[j]) * f)); }
  }
  return [235, 250, 255];
}
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

function sprite(rgb) {
  const s = document.createElement('canvas'); s.width = s.height = 64;
  const g = s.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)');
  gr.addColorStop(.18, `rgba(${rgb},0.95)`);
  gr.addColorStop(.5, `rgba(${rgb},0.25)`);
  gr.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64); return s;
}
const S = { cy: sprite('0,229,255'), bl: sprite('47,143,224'), lt: sprite('126,200,255'),
            wh: sprite('230,247,255'), am: sprite('255,184,77') };
function dot(sp, x, y, r, a = 1) {
  if (r < .3) return;
  X.globalAlpha = clamp(a); X.drawImage(sp, x - r, y - r, r * 2, r * 2); X.globalAlpha = 1;
}
function line(a, b, w, col) {
  X.strokeStyle = col; X.lineWidth = w; X.beginPath(); X.moveTo(a[0], a[1]); X.lineTo(b[0], b[1]); X.stroke();
}
function poly(pts, w, col) {
  X.strokeStyle = col; X.lineWidth = w; X.lineJoin = 'round'; X.lineCap = 'round';
  X.beginPath(); pts.forEach((p, i) => i ? X.lineTo(p[0], p[1]) : X.moveTo(p[0], p[1])); X.stroke();
}
const gauss = R => R() + R() + R() - 1.5;

// ---------- HOME: la rotta come fiume di luce ----------
const home = (() => {
  const R = rng(3), ps = [];
  for (let i = 0; i < 2200; i++)
    ps.push({ u: R(), sp: .015 + R() * .03, a: R() * 6.28, rad: 2 + Math.abs(gauss(R)) * 20, w: (R() - .5) * 8, h: R() });
  const route = u => { const a = u * 6.2832;
    return [250 * Math.cos(a) + 70 * Math.cos(3 * a), -40 * Math.sin(2 * a) - 35 * Math.cos(a) - 20, 170 * Math.sin(a) + 55 * Math.sin(2 * a + 1)]; };
  const spine = []; for (let i = 0; i <= 240; i++) spine.push(route(i / 240));
  return { title: 'RUNAI · ROTTA', trail: .2,
    frame(t) {
      cam.yaw = t * .12 + M.x * .9; cam.pitch = .55 + M.y * .4; cam.d = 650; cam.cy = .52;
      for (let gx = -420; gx <= 420; gx += 28) for (let gz = -300; gz <= 300; gz += 28) {
        const p = P(gx, 135 + 5 * Math.sin(gx * .02 + gz * .03 + t), gz);
        dot(S.bl, p[0], p[1], 3.2 * p[2] * cam.s, .22);
      }
      poly(spine.map(v => P(...v)), 6, 'rgba(47,143,224,0.12)');
      poly(spine.map(v => P(...v)), 1.6, 'rgba(126,200,255,0.55)');
      for (const p of ps) {
        const u = mod1(p.u + t * p.sp * .5), b = route(u), ang = p.a + t * p.w * .3 + u * 30;
        const ox = Math.cos(ang) * p.rad, oy = Math.sin(ang) * p.rad;
        const q = P(b[0] + ox * .7, b[1] + oy, b[2] + ox * .7);
        dot(p.h > .75 ? S.cy : S.lt, q[0], q[1], (1.6 + p.h * 3) * q[2] * cam.s, .35 + p.h * .5);
      }
      const hu = mod1(t * .05), hb = route(hu), hp = P(...hb), gp = P(hb[0], 135, hb[2]);
      line(hp, gp, 1.5, 'rgba(0,229,255,0.5)');
      dot(S.wh, hp[0], hp[1], 20 * hp[2] * cam.s, 1);
      dot(S.cy, hp[0], hp[1], (36 + 8 * Math.sin(t * 4)) * hp[2] * cam.s, .6);
      const ph = (t * .5) % 1;
      X.strokeStyle = `rgba(0,229,255,${.8 * (1 - ph)})`; X.lineWidth = 2; X.beginPath();
      X.ellipse(gp[0], gp[1], (10 + 60 * ph) * gp[2] * cam.s, (4 + 20 * ph) * gp[2] * cam.s, 0, 0, 6.2832); X.stroke();
    } };
})();

// ---------- ANALISI: ECG a strati nello spazio ----------
function ecgf(p) {
  const g = (d, s) => Math.exp(-d * d / (2 * s * s));
  return .15 * g(p - .15, .03) - .2 * g(p - .27, .012) + g(p - .30, .012) - .3 * g(p - .33, .012) + .3 * g(p - .55, .05);
}
const analisi = { title: 'RUNAI · BIOMETRIA', trail: 0,
  frame(t) {
    cam.yaw = -.35 + Math.sin(t * .2) * .15 + M.x * .5; cam.pitch = .62 + M.y * .25; cam.d = 760; cam.cy = .55;
    const L = 30, N = 90;
    for (let j = 0; j < L; j++) {
      const z = lerp(-280, 280, j / (L - 1)), pts = [];
      for (let i = 0; i <= N; i++) {
        const x = lerp(-400, 400, i / N), env = Math.exp(-x * x / (2 * 220 * 220));
        const ph = mod1((x + 400) / 800 * 2.2 - t * .35 + z * .0035);
        const y = -(ecgf(ph) * 85 * env + 6 * Math.sin(x * .03 + t * 2 + z * .05) * (.3 + env)) + 40;
        pts.push(P(x, y, z).concat([Math.abs(y - 40) / 90]));
      }
      for (let i = 0; i < N; i++) {
        const a = pts[i], b = pts[i + 1], v = (a[4] + b[4]) / 2, k = (a[2] + b[2]) / 2;
        line(a, b, 1 + k * .9, rgba(ramp(.15 + v * .9), clamp(.12 + (k - .6) * 1.1 + v * .4)));
      }
      for (const p of pts) if (p[4] > .55) dot(S.wh, p[0], p[1], 9 * p[2] * cam.s, p[4] * .7);
    }
  } };

// ---------- STATS: colonne di luce ----------
const stats = (() => {
  const R = rng(11), cols = 14, rows = 6, data = [];
  for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++)
    data.push(40 + 110 * clamp((.5 + .5 * Math.sin(i * .55 + j * 1.3)) * (.55 + .45 * Math.cos(j * .7 - i * .2)) + (R() - .5) * .25));
  return { title: 'RUNAI · VOLUME', trail: 0,
    frame(t) {
      cam.yaw = -.6 + Math.sin(t * .15) * .25 + M.x * .5; cam.pitch = .5 + M.y * .3; cam.d = 820; cam.cy = .6;
      const items = [];
      for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
        const g = ease(t * .7 - (i + j) * .06), v = data[i * rows + j] * g * (1 + .04 * Math.sin(t * 2 + i + j));
        const x = (i - (cols - 1) / 2) * 46, z = (j - (rows - 1) / 2) * 60;
        items.push({ b: P(x, 100, z), top: P(x, 100 - v, z), v: v / 150 });
      }
      items.sort((a, b) => b.b[3] - a.b[3]);
      for (const it of items) {
        const k = it.b[2], c = ramp(.2 + it.v * .8);
        line(it.b, it.top, 6 * k * cam.s, rgba(c, .16));
        line(it.b, it.top, 2.2 * k * cam.s, rgba(c, .75));
        dot(S.bl, it.b[0], it.b[1], 12 * k * cam.s, .35);
        dot(it.v > .6 ? S.wh : S.cy, it.top[0], it.top[1], (8 + it.v * 10) * k * cam.s, .9);
      }
      for (let j = 0; j < rows; j += 2) {
        const pts = []; for (let i = 0; i < cols; i++) {
          const v = data[i * rows + j] * ease(t * .7 - (i + j) * .06), x = (i - (cols - 1) / 2) * 46, z = (j - (rows - 1) / 2) * 60;
          pts.push(P(x, 100 - v, z));
        }
        poly(pts, 1.4, 'rgba(0,229,255,0.45)');
      }
    } };
})();

// ---------- KPI: anelli orbitali ----------
const kpi = (() => {
  const R = rng(5), radii = [110, 150, 190, 230, 270], rings = [];
  radii.forEach((r, ri) => { const n = 90 + ri * 30, a = [];
    for (let i = 0; i < n; i++) a.push({ f: i / n, sz: 1 + R() * 2.2, j: (R() - .5) * 8 }); rings.push({ r, a, dir: ri % 2 ? -1 : 1 }); });
  const streak = []; for (let i = 0; i < 260; i++) streak.push({ a: R() * 6.28, o: R(), sp: .12 + R() * .25, h: R() });
  return { title: 'RUNAI · INDICE DI FORMA', trail: .2,
    frame(t) {
      cam.yaw = M.x * .5 + Math.sin(t * .3) * .15; cam.pitch = .95 + M.y * .3; cam.d = 760; cam.cy = .5;
      const shown = .824 * ease(t / 2.4);
      rings.forEach((rg, ri) => rg.a.forEach(p => {
        const th = (p.f + t * .02 * rg.dir * (1 + ri * .15)) * 6.2832;
        const q = P(rg.r * Math.cos(th), Math.sin(th * 3 + t) * 6 + p.j, rg.r * Math.sin(th));
        const main = ri === 2, inGauge = main && p.f < shown;
        dot(inGauge ? S.cy : (main ? S.bl : S.lt), q[0], q[1], (p.sz * (inGauge ? 2.4 : 1.2)) * q[2] * cam.s * 2.2, inGauge ? .95 : (main ? .3 : .28));
      }));
      const ha = shown * 6.2832 + (-0), hq = P(190 * Math.cos(ha + 0), 0, 190 * Math.sin(ha + 0));
      dot(S.wh, hq[0], hq[1], 22 * hq[2] * cam.s, 1); dot(S.cy, hq[0], hq[1], 44 * hq[2] * cam.s, .5);
      for (const s of streak) { const r = 30 + mod1(s.o + t * s.sp * .3) * 270, q = P(r * Math.cos(s.a), 0, r * Math.sin(s.a));
        dot(S.lt, q[0], q[1], 2.4 * q[2] * cam.s, (1 - r / 300) * .5); }
      X.globalCompositeOperation = 'source-over'; X.textAlign = 'center';
      X.shadowColor = '#00E5FF'; X.shadowBlur = 24; X.fillStyle = '#E8F6FF';
      X.font = `700 ${Math.round(64 * cam.s)}px "JetBrains Mono", monospace`;
      X.fillText((82.4 * ease(t / 2.4)).toFixed(1), W / 2, H / 2 + 14 * cam.s);
      X.shadowBlur = 0; X.fillStyle = 'rgba(126,200,255,0.8)'; X.font = `600 ${Math.round(13 * cam.s)}px Inter, sans-serif`;
      X.fillText('INDICE DI FORMA', W / 2, H / 2 + 42 * cam.s); X.textAlign = 'start';
      X.globalCompositeOperation = 'lighter';
    } };
})();

// ---------- ML: rete neurale 3D con impulsi ----------
const ml = (() => {
  const R = rng(8), layers = [7, 9, 9, 5, 3], nodes = [], idx = [];
  layers.forEach((n, l) => { idx.push([]); for (let i = 0; i < n; i++) {
    idx[l].push(nodes.length); nodes.push({ l, p: [(l - 2) * 150, (i - (n - 1) / 2) * 38, (R() - .5) * 170], act: 0 }); } });
  const links = []; for (let l = 0; l < layers.length - 1; l++) idx[l].forEach(a => idx[l + 1].forEach(b => links.push([a, b])));
  const pulses = []; const mk = () => { const path = []; for (let l = 0; l < layers.length; l++) path.push(idx[l][Math.floor(R() * layers[l])]);
    return { path, s: R() * 4, sp: .5 + R() * .6 }; };
  for (let i = 0; i < 40; i++) pulses.push(mk());
  return { title: 'RUNAI · MODELLO PREDITTIVO', trail: 0,
    frame(t, dt) {
      cam.yaw = t * .15 + M.x; cam.pitch = .35 + M.y * .3; cam.d = 800; cam.cy = .5;
      nodes.forEach(n => n.act = Math.max(0, n.act - dt * 1.4));
      for (const [a, b] of links) line(P(...nodes[a].p), P(...nodes[b].p), 1, 'rgba(126,200,255,0.07)');
      for (const pl of pulses) {
        pl.s += dt * pl.sp; if (pl.s >= 4) Object.assign(pl, mk(), { s: 0 });
        const i = Math.floor(pl.s), f = pl.s - i, a = nodes[pl.path[i]], b = nodes[pl.path[i + 1]];
        if (f < .1) a.act = 1;
        for (let k = 0; k < 7; k++) { const ff = clamp(f - k * .04), q = P(lerp(a.p[0], b.p[0], ff), lerp(a.p[1], b.p[1], ff), lerp(a.p[2], b.p[2], ff));
          dot(S.cy, q[0], q[1], (9 - k) * q[2] * cam.s, (1 - k / 7) * .9); }
      }
      nodes.forEach(n => { const q = P(...n.p), out = n.l === 4;
        dot(out ? S.am : S.lt, q[0], q[1], (6 + n.act * 16 + (out ? 6 : 0)) * q[2] * cam.s, .5 + n.act * .5);
        dot(S.wh, q[0], q[1], 3 * q[2] * cam.s, 1); });
    } };
})();

// ---------- PLAN: montagna wireframe e sentiero ----------
const plan = (() => {
  const Hf = (x, z) => -(190 * Math.exp(-((x - 80) ** 2 + z * z) / (2 * 130 * 130)) + 75 * Math.exp(-((x + 230) ** 2 + (z - 60) ** 2) / (2 * 100 * 100))
    + 50 * Math.exp(-((x - 320) ** 2 + (z + 90) ** 2) / (2 * 80 * 80)) + 10 * Math.sin(x * .05) * Math.cos(z * .06)) + 110;
  const gx = 56, gz = 40, grid = [];
  for (let i = 0; i < gx; i++) { grid.push([]); for (let j = 0; j < gz; j++) { const x = (i - gx / 2) * 15, z = (j - gz / 2) * 15; grid[i].push([x, Hf(x, z), z]); } }
  const path = []; for (let i = 0; i <= 220; i++) { const s = i / 220, a = -.4 + s * 9, r = 300 * (1 - s) + 6;
    const x = 80 + r * Math.cos(a), z = r * .6 * Math.sin(a); path.push([x, Hf(x, z) - 3, z]); }
  return { title: 'RUNAI · PIANO ALLENAMENTO', trail: 0,
    frame(t) {
      cam.yaw = -.5 + Math.sin(t * .1) * .3 + M.x * .6; cam.pitch = .62 + M.y * .25; cam.d = 900; cam.cy = .56;
      const seg = (a, b) => { const pa = P(...a), pb = P(...b), h = clamp((110 - (a[1] + b[1]) / 2) / 300);
        line(pa, pb, 1, rgba(ramp(.1 + h * 1.1), .1 + h * .6)); };
      for (let i = 0; i < gx; i++) for (let j = 0; j < gz; j++) { if (i < gx - 1) seg(grid[i][j], grid[i + 1][j]); if (j < gz - 1) seg(grid[i][j], grid[i][j + 1]); }
      const pp = path.map(v => P(...v));
      poly(pp, 7, 'rgba(0,229,255,0.15)'); poly(pp, 2.2, 'rgba(0,229,255,0.9)');
      const hd = pp[Math.floor(mod1(t * .07) * 220)];
      dot(S.wh, hd[0], hd[1], 18 * hd[2] * cam.s, 1); dot(S.cy, hd[0], hd[1], 34 * hd[2] * cam.s, .6);
      const top = path[220], tp = P(...top), fp = P(top[0], top[1] - 70, top[2]);
      line(tp, fp, 2, 'rgba(232,246,255,0.9)'); dot(S.cy, fp[0], fp[1], (22 + 5 * Math.sin(t * 3)) * fp[2] * cam.s, .9); dot(S.wh, fp[0], fp[1], 9 * fp[2] * cam.s, 1);
    } };
})();

// ---------- CV: tracking su point cloud ----------
const cv = (() => {
  const trail = [], pts = [];
  for (let i = 0; i < 40; i++) for (let j = 0; j < 26; j++) pts.push([(i - 19.5) * 22, 110, (j - 12.5) * 22]);
  const pos = t => [230 * Math.sin(t * .6), -Math.abs(Math.sin(t * 5)) * 22, 120 * Math.sin(t * .9 + 1)];
  return { title: 'RUNAI · COMPUTER VISION', trail: 0,
    frame(t) {
      cam.yaw = .5 * Math.sin(t * .12) + M.x * .6; cam.pitch = .5 + M.y * .3; cam.d = 760; cam.cy = .56;
      const xs = mod1(t * .18) * 900 - 450;
      for (const p of pts) { const b = Math.exp(-((p[0] - xs) ** 2) / (2 * 55 * 55)), q = P(...p);
        dot(b > .25 ? S.cy : S.bl, q[0], q[1], (2 + b * 6) * q[2] * cam.s, .2 + b * .8); }
      const a = P(xs, 110, -290), b = P(xs, 110, 290), c = P(xs, -120, 290), d = P(xs, -120, -290);
      poly([a, b, c, d, a], 1.2, 'rgba(0,229,255,0.35)');
      const o = pos(t); trail.push(o); if (trail.length > 90) trail.shift();
      trail.forEach((v, i) => { const q = P(...v), f = i / trail.length; dot(S.cy, q[0], q[1], (3 + f * 9) * q[2] * cam.s, f * .8); });
      const q = P(...o), g = P(o[0], 110, o[2]);
      line(q, g, 1.2, 'rgba(126,200,255,0.5)');
      dot(S.wh, q[0], q[1], 16 * q[2] * cam.s, 1); dot(S.cy, q[0], q[1], 34 * q[2] * cam.s, .6);
      for (let r = 0; r < 3; r++) { const ph = mod1(t * .6 + r / 3);
        X.strokeStyle = `rgba(0,229,255,${.7 * (1 - ph)})`; X.lineWidth = 1.6; X.beginPath();
        X.ellipse(g[0], g[1], (8 + 90 * ph) * g[2] * cam.s, (3 + 34 * ph) * g[2] * cam.s, 0, 0, 6.2832); X.stroke(); }
      const w = 46 * q[2] * cam.s, l = 14 * cam.s, x0 = q[0] - w, x1 = q[0] + w, y0 = q[1] - w * 1.5, y1 = q[1] + w * 1.2;
      X.strokeStyle = '#00E5FF'; X.lineWidth = 2.4; X.beginPath();
      X.moveTo(x0, y0 + l); X.lineTo(x0, y0); X.lineTo(x0 + l, y0); X.moveTo(x1 - l, y0); X.lineTo(x1, y0); X.lineTo(x1, y0 + l);
      X.moveTo(x1, y1 - l); X.lineTo(x1, y1); X.lineTo(x1 - l, y1); X.moveTo(x0 + l, y1); X.lineTo(x0, y1); X.lineTo(x0, y1 - l); X.stroke();
      X.globalCompositeOperation = 'source-over'; X.fillStyle = '#00E5FF'; X.fillRect(x0, y0 - 20, 92 * cam.s, 17 * cam.s);
      X.fillStyle = '#040A18'; X.font = `700 ${Math.round(11 * cam.s)}px "JetBrains Mono", monospace`; X.fillText('ATLETA · 98%', x0 + 6, y0 - 7 * cam.s - 1);
      X.globalCompositeOperation = 'lighter';
    } };
})();

const scenes = { home, analisi, stats, kpi, ml, plan, cv };
let cur = scenes[window.SCENE] || home;
window.setScene = n => { cur = scenes[n] || home; X.clearRect(0, 0, W, H); };

function loop(now) {
  const dt = Math.min(.05, (now - last) / 1000); last = now; T += dt;
  M.x += (M.tx - M.x) * .05; M.y += (M.ty - M.y) * .05;
  X.globalCompositeOperation = 'source-over';
  if (cur.trail > 0) { X.globalCompositeOperation = 'destination-out'; X.fillStyle = `rgba(0,0,0,${cur.trail})`; X.fillRect(0, 0, W, H); }
  else X.clearRect(0, 0, W, H);
  X.globalCompositeOperation = 'lighter';
  cur.frame(T, dt);
  X.globalCompositeOperation = 'source-over';
  X.fillStyle = 'rgba(126,200,255,0.75)'; X.font = `600 ${Math.round(11 * Math.max(.9, cam.s))}px Inter, sans-serif`;
  X.fillText('●  ' + cur.title, 18, H - 16);
  requestAnimationFrame(loop);
}
resize(); requestAnimationFrame(loop);
})();
