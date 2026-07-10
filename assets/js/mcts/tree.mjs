// Draw an MCTS search tree as SVG from the per-node arrays the search returns.
//
//   parents[i]  parent node id of node i, or -1 if node i is not in the tree
//               (node 0 is the root; its parent is -1 but it is always present)
//   visits[i]   visit count of node i (node radius, and picks the best line)
//
// Nodes are sized by visit count and laid out as a tidy top-down tree (each
// subtree occupies a contiguous horizontal band, so no edges cross). The
// principal variation (follow the most-visited child from the root) is marked
// with class "pv".
//
// opts.grow (seconds): if > 0, each node/edge gets an animation-delay in the
// order the search actually created the nodes. Node ids are assigned in
// creation order, so replaying them in id order reproduces how the tree was
// built. The CSS keyframes that use the delay live with the consumer
// (demo.mjs); here we only emit the delays.
//
// Returns an HTML string: an <svg> plus a caption.

export function treeHTML(parents, visits, { width = 300, height = 250, pad = 12, grow = 0, caption } = {}) {
  const n = parents.length;
  const children = Array.from({ length: n }, () => []);
  for (let i = 1; i < n; i++) if (parents[i] >= 0) children[parents[i]].push(i);

  const depth = new Int32Array(n).fill(-1);
  const xpos = new Float64Array(n);
  let leaf = 0;
  (function assign(u, d) {
    depth[u] = d;
    const ch = children[u];
    if (ch.length === 0) {
      xpos[u] = leaf++;
      return;
    }
    let sum = 0;
    for (const c of ch) {
      assign(c, d + 1);
      sum += xpos[c];
    }
    xpos[u] = sum / ch.length;
  })(0, 0);

  const nodes = [];
  for (let i = 0; i < n; i++) if (depth[i] >= 0) nodes.push(i);
  const maxLeaf = Math.max(1, leaf - 1);
  let maxDepth = 1;
  for (const i of nodes) if (depth[i] > maxDepth) maxDepth = depth[i];
  const maxVisits = Math.max(1, visits[0]);

  // birth order == ascending node id == the order `nodes` was collected
  const rank = new Int32Array(n);
  nodes.forEach((id, k) => (rank[id] = k));
  const total = Math.max(1, nodes.length);
  const delay = (id) => (grow > 0 ? ` style="animation-delay:${((rank[id] / total) * grow).toFixed(3)}s"` : "");

  // principal variation: from the root, always follow the most-visited child
  const pv = new Set([0]);
  for (let u = 0; children[u].length; ) {
    let best = children[u][0];
    for (const c of children[u]) if (visits[c] > visits[best]) best = c;
    pv.add(best);
    u = best;
  }

  const px = (i) => pad + (xpos[i] / maxLeaf) * (width - 2 * pad);
  const py = (i) => pad + (depth[i] / maxDepth) * (height - 2 * pad);

  let s = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">`;
  for (const u of nodes)
    for (const c of children[u]) {
      const cls = pv.has(u) && pv.has(c) ? "edge pv" : "edge";
      s += `<line class="${cls}"${delay(c)} x1="${px(u).toFixed(1)}" y1="${py(u).toFixed(1)}" x2="${px(c).toFixed(1)}" y2="${py(c).toFixed(1)}" />`;
    }
  for (const i of nodes) {
    const r = (1.1 + 3.6 * Math.sqrt(visits[i] / maxVisits)).toFixed(2);
    s += `<circle class="${pv.has(i) ? "node pv" : "node"}"${delay(i)} cx="${px(i).toFixed(1)}" cy="${py(i).toFixed(1)}" r="${r}" />`;
  }
  const cap = caption === undefined ? `search tree · ${nodes.length} nodes · best line highlighted` : caption;
  s += `</svg>` + (cap ? `<div class="caption">${cap}</div>` : "");
  return s;
}
