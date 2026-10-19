import math
import time
from flask import Flask, jsonify, request

app = Flask(__name__)

ROWS, COLS = 6, 7
EMPTY, HUMAN, AI = 0, 1, 2
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


# ---------------- Game logic ----------------
def valid_moves(b):
    return [c for c in range(COLS) if b[0][c] == EMPTY]


def drop(b, col, p):
    for r in range(ROWS - 1, -1, -1):
        if b[r][col] == EMPTY:
            b[r][col] = p
            return


def undo(b, col):
    for r in range(ROWS):
        if b[r][col] != EMPTY:
            b[r][col] = EMPTY
            return


def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS


def win_cells(b, p):
    """Return the 4 winning cells (as flat indexes) or None."""
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in DIRS:
                if in_bounds(r + 3 * dr, c + 3 * dc):
                    cells = [(r + i * dr, c + i * dc) for i in range(4)]
                    if all(b[y][x] == p for y, x in cells):
                        return [y * COLS + x for y, x in cells]
    return None


# ---------------- AI: minimax + alpha-beta ----------------
def score_window(w):
    s = 0
    if w.count(AI) == 3 and w.count(EMPTY) == 1:
        s += 5
    elif w.count(AI) == 2 and w.count(EMPTY) == 2:
        s += 2
    if w.count(HUMAN) == 3 and w.count(EMPTY) == 1:
        s -= 4
    return s


def evaluate(b):
    score = [b[r][COLS // 2] for r in range(ROWS)].count(AI) * 3
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in DIRS:
                if in_bounds(r + 3 * dr, c + 3 * dc):
                    score += score_window([b[r + i * dr][c + i * dc] for i in range(4)])
    return score


def minimax(b, depth, alpha, beta, maximizing, prune, stats):
    stats["nodes"] += 1
    if win_cells(b, AI):
        return None, 1_000_000 + depth
    if win_cells(b, HUMAN):
        return None, -1_000_000 - depth
    moves = sorted(valid_moves(b), key=lambda c: abs(c - COLS // 2))
    if not moves:
        return None, 0
    if depth == 0:
        return None, evaluate(b)

    best = -math.inf if maximizing else math.inf
    best_col = moves[0]
    for c in moves:
        drop(b, c, AI if maximizing else HUMAN)
        _, s = minimax(b, depth - 1, alpha, beta, not maximizing, prune, stats)
        undo(b, c)
        if maximizing:
            if s > best:
                best, best_col = s, c
            alpha = max(alpha, best)
        else:
            if s < best:
                best, best_col = s, c
            beta = min(beta, best)
        if prune and alpha >= beta:
            break
    return best_col, best


# ---------------- API ----------------
@app.post("/api/move")
def api_move():
    d = request.get_json(silent=True) or {}
    flat = d.get("board")
    if not (isinstance(flat, list) and len(flat) == ROWS * COLS and all(v in (0, 1, 2) for v in flat)):
        return jsonify(error="Bad board"), 400
    try:
        col = int(d.get("col"))
        depth = max(1, min(int(d.get("depth", 4)), 6))
    except (TypeError, ValueError):
        return jsonify(error="Bad input"), 400
    prune = bool(d.get("prune", True))
    if not prune:
        depth = min(depth, 5)  # plain minimax at depth 6 is too slow for a web server

    b = [flat[r * COLS:(r + 1) * COLS] for r in range(ROWS)]
    if col not in valid_moves(b):
        return jsonify(error="Invalid move"), 400

    out = {"status": "play", "cells": [], "nodes": 0, "ms": 0, "depth": depth, "prune": prune}
    drop(b, col, HUMAN)
    cells = win_cells(b, HUMAN)
    if cells:
        out.update(status="human", cells=cells)
    elif not valid_moves(b):
        out.update(status="draw")
    else:
        stats = {"nodes": 0}
        t = time.time()
        ai_col, _ = minimax(b, depth, -math.inf, math.inf, True, prune, stats)
        out.update(nodes=stats["nodes"], ms=round((time.time() - t) * 1000, 1))
        drop(b, ai_col, AI)
        cells = win_cells(b, AI)
        if cells:
            out.update(status="ai", cells=cells)
        elif not valid_moves(b):
            out.update(status="draw")
    out["board"] = [v for row in b for v in row]
    return jsonify(out)


@app.get("/")
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect Four AI</title>
<style>
body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f8fafc;color:#0f172a;display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}
@media (prefers-color-scheme:dark){body{background:#0f172a;color:#f1f5f9}.cell{background:#0f172a!important}select,button{background:#1e293b;color:#f1f5f9}}
h1{font-size:1.3rem;margin:0}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:center}
select,button{font:inherit;padding:6px 10px;border:1px solid #94a3b8;border-radius:8px;cursor:pointer}
#board{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;background:#1e3a8a;padding:10px;border-radius:14px;width:min(94vw,520px)}
.cell{aspect-ratio:1;border-radius:50%;background:#fff;cursor:pointer}
.cell.h{background:#dc2626!important}.cell.a{background:#facc15!important}
.cell.w{outline:4px solid #22c55e;outline-offset:-4px}
#status{font-size:1.1rem;font-weight:600}
table{border-collapse:collapse;font-size:.85rem}
th,td{border:1px solid #94a3b8;padding:4px 10px;text-align:center}
.wrap{max-width:100%;overflow-x:auto}
</style></head><body>
<h1>Connect Four AI (Python: Minimax + Alpha-Beta)</h1>
<div class="bar">
 <label>Difficulty
  <select id="lvl"><option value="2">Easy (depth 2)</option><option value="4" selected>Medium (depth 4)</option><option value="6">Hard (depth 6)</option></select>
 </label>
 <label><input type="checkbox" id="prune" checked> Alpha-Beta pruning</label>
 <button id="new">New Game</button>
</div>
<div id="status"></div>
<div id="board"></div>
<div class="wrap"><table><thead><tr><th>#</th><th>Depth</th><th>Pruning</th><th>Nodes</th><th>Time (ms)</th></tr></thead><tbody id="log"></tbody></table></div>
<script>
let b,over,busy,n;
const el=document.getElementById('board'),st=document.getElementById('status'),log=document.getElementById('log');
for(let i=0;i<42;i++){const d=document.createElement('div');d.className='cell';d.onclick=()=>play(i%7);el.appendChild(d)}
function draw(w){[...el.children].forEach((d,i)=>{d.className='cell'+(b[i]==1?' h':b[i]==2?' a':'')+(w&&w.includes(i)?' w':'')})}
function reset(){b=Array(42).fill(0);over=false;busy=false;n=0;log.innerHTML='';st.textContent='Your turn (red). Click a column.';draw()}
async function play(col){
  if(over||busy||b[col]!==0)return;
  busy=true;st.textContent='AI is thinking...';
  try{
    const r=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({board:b,col:col,depth:+document.getElementById('lvl').value,prune:document.getElementById('prune').checked})});
    const d=await r.json();
    if(!r.ok){st.textContent=d.error||'Error';busy=false;return}
    b=d.board;draw(d.cells);
    if(d.nodes){n++;log.insertAdjacentHTML('beforeend',`<tr><td>${n}</td><td>${d.depth}</td><td>${d.prune?'On':'Off'}</td><td>${d.nodes}</td><td>${d.ms}</td></tr>`)}
    if(d.status==='play')st.textContent='Your turn (red). Click a column.';
    else{over=true;st.textContent={human:'You win!',ai:'AI wins!',draw:'Draw!'}[d.status]}
  }catch(e){st.textContent='Server error, try again.'}
  busy=false;
}
document.getElementById('new').onclick=reset;
reset();
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
