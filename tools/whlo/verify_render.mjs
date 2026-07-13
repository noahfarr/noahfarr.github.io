// Pixel-accuracy check: for every fixture, rasterize pgx's reference SVG and our
// JS renderer's SVG with the SAME Chromium engine, then pixel-diff with
// ImageMagick. AE (absolute error) == 0 means identical pixels.
//
//   node verify_render.mjs        (after: uv run python generate_reference.py)

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { renderSVG } from "../../assets/js/mcts/render.mjs";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REFERENCE_DIR = path.join(SCRIPT_DIR, "reference");
const TEMP_DIR = path.join(SCRIPT_DIR, ".verify_tmp");
const CHROME_PROFILE = path.join(TEMP_DIR, "chrome-profile");
mkdirSync(TEMP_DIR, { recursive: true });

const CHROMIUM = process.env.CHROMIUM || "chromium-browser";
const GAMES = { tic_tac_toe: [0, 1, 2, 3], connect_four: [0, 1, 2, 3], chess: [0, 1, 2, 3] };

const svgSize = (svg) => ({
  width: Math.round(+/width="([\d.]+)"/.exec(svg)[1]),
  height: Math.round(+/height="([\d.]+)"/.exec(svg)[1]),
});

const wrapHtml = (svg) =>
  `<!doctype html><html><head><meta charset="utf-8">` +
  `<style>*{margin:0;padding:0}svg{display:block}</style></head><body>${svg}</body></html>`;

function rasterize(svg, name) {
  const { width, height } = svgSize(svg);
  const htmlPath = path.join(TEMP_DIR, `${name}.html`);
  const pngPath = path.join(TEMP_DIR, `${name}.png`);
  writeFileSync(htmlPath, wrapHtml(svg));
  execSync(
    `${CHROMIUM} --headless=new --disable-gpu --no-sandbox --hide-scrollbars ` +
      `--no-first-run --no-default-browser-check --force-device-scale-factor=1 ` +
      `--user-data-dir="${CHROME_PROFILE}" --default-background-color=FFFFFFFF ` +
      `--window-size=${width},${height} --screenshot="${pngPath}" "file://${htmlPath}"`,
    { stdio: "ignore" }
  );
  return { pngPath, width, height };
}

function pixelDiff(referencePng, ourPng, diffPath) {
  const output = execSync(`compare -metric AE "${referencePng}" "${ourPng}" "${diffPath}" 2>&1 || true`, { encoding: "utf8" });
  const match = /^([\d.e+]+)/.exec(output.trim());
  return match ? Math.round(parseFloat(match[1])) : NaN;
}

let failures = 0;
for (const [game, indices] of Object.entries(GAMES)) {
  for (const index of indices) {
    const record = JSON.parse(readFileSync(path.join(REFERENCE_DIR, game, `${index}.json`), "utf8"));
    const referenceSvg = readFileSync(path.join(REFERENCE_DIR, game, `${index}.svg`), "utf8");
    const ourSvg = renderSVG(record);
    writeFileSync(path.join(TEMP_DIR, `${game}_${index}_mine.svg`), ourSvg);

    const reference = rasterize(referenceSvg, `${game}_${index}_ref`);
    const ours = rasterize(ourSvg, `${game}_${index}_mine`);
    const absoluteError = pixelDiff(reference.pngPath, ours.pngPath, path.join(TEMP_DIR, `${game}_${index}_diff.png`));
    const totalPixels = reference.width * reference.height;
    const identical = absoluteError === 0;
    if (!identical) failures++;
    console.log(`${identical ? "✓" : "✗"} ${game}/${index}  ${reference.width}x${reference.height}  AE=${absoluteError}/${totalPixels}`);
  }
}
console.log(failures ? `\n${failures} fixture(s) differ (see ${TEMP_DIR}/*_diff.png)` : `\nALL PIXEL-ACCURATE ✓`);
process.exit(failures ? 1 : 0);
