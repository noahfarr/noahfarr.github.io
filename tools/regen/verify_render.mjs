// Pixel-accuracy check: for every fixture, rasterize pgx's reference SVG and our
// JS renderer's SVG with the SAME Chromium engine, then pixel-diff with
// ImageMagick. AE (absolute error) == 0 means identical pixels.
//
//   node web/verify_render.mjs        (after: uv run python web/gen_reference.py)

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { renderSVG } from "../../assets/js/mcts/render.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REF = path.join(HERE, "reference");
const TMP = path.join(HERE, ".verify_tmp");
const PROFILE = path.join(TMP, "chrome-profile");
mkdirSync(TMP, { recursive: true });

const CHROMIUM = process.env.CHROMIUM || "chromium-browser";
const GAMES = { tic_tac_toe: [0, 1, 2, 3], connect_four: [0, 1, 2, 3], chess: [0, 1, 2, 3] };

const sizeOf = (svg) => ({
  w: Math.round(+/width="([\d.]+)"/.exec(svg)[1]),
  h: Math.round(+/height="([\d.]+)"/.exec(svg)[1]),
});

const wrap = (svg) =>
  `<!doctype html><html><head><meta charset="utf-8">` +
  `<style>*{margin:0;padding:0}svg{display:block}</style></head><body>${svg}</body></html>`;

function rasterize(svg, tag) {
  const { w, h } = sizeOf(svg);
  const html = path.join(TMP, `${tag}.html`);
  const png = path.join(TMP, `${tag}.png`);
  writeFileSync(html, wrap(svg));
  execSync(
    `${CHROMIUM} --headless=new --disable-gpu --no-sandbox --hide-scrollbars ` +
      `--no-first-run --no-default-browser-check --force-device-scale-factor=1 ` +
      `--user-data-dir="${PROFILE}" --default-background-color=FFFFFFFF ` +
      `--window-size=${w},${h} --screenshot="${png}" "file://${html}"`,
    { stdio: "ignore" }
  );
  return { png, w, h };
}

function pixelDiff(a, b, out) {
  const r = execSync(`compare -metric AE "${a}" "${b}" "${out}" 2>&1 || true`, { encoding: "utf8" });
  const m = /^([\d.e+]+)/.exec(r.trim());
  return m ? Math.round(parseFloat(m[1])) : NaN;
}

let fails = 0;
for (const [game, idxs] of Object.entries(GAMES)) {
  for (const n of idxs) {
    const rec = JSON.parse(readFileSync(path.join(REF, game, `${n}.json`), "utf8"));
    const refSvg = readFileSync(path.join(REF, game, `${n}.svg`), "utf8");
    const mySvg = renderSVG(rec);
    writeFileSync(path.join(TMP, `${game}_${n}_mine.svg`), mySvg);

    const ref = rasterize(refSvg, `${game}_${n}_ref`);
    const mine = rasterize(mySvg, `${game}_${n}_mine`);
    const ae = pixelDiff(ref.png, mine.png, path.join(TMP, `${game}_${n}_diff.png`));
    const total = ref.w * ref.h;
    const ok = ae === 0;
    if (!ok) fails++;
    console.log(`${ok ? "✓" : "✗"} ${game}/${n}  ${ref.w}x${ref.h}  AE=${ae}/${total}`);
  }
}
console.log(fails ? `\n${fails} fixture(s) differ (see ${TMP}/*_diff.png)` : `\nALL PIXEL-ACCURATE ✓`);
process.exit(fails ? 1 : 0);
