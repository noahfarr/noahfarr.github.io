// Pixel-accurate JavaScript port of pgx's SVG board rendering
// (pgx/_src/visualizer.py + pgx/_src/dwg/*). Given the raw pgx state fields a
// game exposes, these produce SVG geometrically identical to
// `pgx.Visualizer(color_theme="light").get_dwg(state)`, verified by
// rasterizing both and pixel-diffing (see web/verify_render.mjs).
//
// Supported: tic_tac_toe, connect_four, chess. Light theme.
//
// Each renderer takes a record { env_id, board:number[], color?:number } and
// returns a standalone SVG string.

import { CHESS_PIECES } from "./chess_pieces.mjs";

const SVG_OPEN = (w, h) =>
  `<svg baseProfile="full" height="${h}" version="1.1" width="${w}" ` +
  `xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" ` +
  `xmlns:xlink="http://www.w3.org/1999/xlink"><defs />`;

// --- tic_tac_toe -----------------------------------------------------------
// GRID_SIZE 60, board 3x3. Light ColorSet: background=white, grid=black.
// Both marks are drawn in grid_color (black): 0 -> cross, 1 -> ring.
function renderTicTacToe(board) {
  const G = 60,
    BW = 3,
    BH = 3,
    W = (BW + 1) * G,
    H = (BH + 1) * G;
  const grid = "black";
  let s = SVG_OPEN(W, H);
  s += `<g transform="scale(1.0)">`;
  s += `<rect fill="white" height="${H}" width="${W}" x="0" y="0" />`;
  s += `<g transform="translate(${G / 2},${G / 2})">`;

  // horizontal grid lines + endpoint dots
  s += `<g fill="${grid}" id="hlines" stroke="${grid}">`;
  for (let y = 1; y < BH; y++) {
    const yy = G * y;
    s += `<line stroke-width="${G * 0.05}" x1="0" x2="${G * BW}" y1="${yy}" y2="${yy}" />`;
    s += `<circle cx="0" cy="${yy}" r="${G * 0.014}" />`;
    s += `<circle cx="${G * BW}" cy="${yy}" r="${G * 0.014}" />`;
  }
  s += `</g>`;
  // vertical grid lines + endpoint dots
  s += `<g fill="${grid}" id="vline" stroke="${grid}">`;
  for (let x = 1; x < BW; x++) {
    const xx = G * x;
    s += `<line stroke-width="${G * 0.05}" x1="${xx}" x2="${xx}" y1="0" y2="${G * BH}" />`;
    s += `<circle cx="${xx}" cy="0" r="${G * 0.014}" />`;
    s += `<circle cx="${xx}" cy="${G * BH}" r="${G * 0.014}" />`;
  }
  s += `</g>`;

  // marks, in board order
  for (let i = 0; i < board.length; i++) {
    const mark = board[i];
    const x = i % BW,
      y = Math.floor(i / BH);
    if (mark === 0) {
      const w = G * 0.05;
      s += `<line stroke="${grid}" stroke-width="${w}" x1="${(x + 0.1) * G}" x2="${(x + 0.9) * G}" y1="${(y + 0.1) * G}" y2="${(y + 0.9) * G}" />`;
      s += `<line stroke="${grid}" stroke-width="${w}" x1="${(x + 0.1) * G}" x2="${(x + 0.9) * G}" y1="${(y + 0.9) * G}" y2="${(y + 0.1) * G}" />`;
    } else if (mark === 1) {
      s += `<circle cx="${(x + 0.5) * G}" cy="${(y + 0.5) * G}" fill="none" r="${0.4 * G}" stroke="${grid}" stroke-width="${0.05 * G}" />`;
    }
  }
  s += `</g></g></svg>`;
  return s;
}

// --- connect_four ----------------------------------------------------------
// GRID_SIZE 35, BOARD_WIDTH/HEIGHT 7. Light ColorSet: p1=black, p2=white,
// grid=black, text=gray, background=white. Board is 42 (6 rows) but the grid is
// drawn 7x7, exactly as pgx does.
function renderConnectFour(board) {
  const G = 35,
    BW = 7,
    BH = 7,
    W = (BW + 1) * G,
    H = (BH + 1) * G;
  const text = "gray",
    grid = "black";
  let s = SVG_OPEN(W, H);
  s += `<rect fill="white" height="${BH * G}" width="${BW * G}" x="0" y="0" />`; // root bg (added to dwg)
  s += `<g transform="scale(1.0)">`;
  s += `<rect fill="white" height="${H}" width="${W}" x="0" y="0" />`;
  s += `<g transform="translate(${G / 2},${G / 2})">`;

  s += `<g id="vline" stroke="${text}">`;
  for (let x = 1; x < BW; x++) s += `<line stroke-width="1px" x1="${G * x}" x2="${G * x}" y1="0" y2="${G * (BH - 1)}" />`;
  s += `</g>`;
  s += `<g id="vline" stroke="${text}">`;
  for (let y = 1; y < BH; y++) s += `<line stroke-width="0.1px" x1="0" x2="${G * BW}" y1="${G * y}" y2="${G * y}" />`;
  s += `</g>`;

  const bar = 6;
  s += `<rect fill="${grid}" height="${bar}" stroke="${grid}" width="${BW * G}" x="0" y="${(BH - 1) * G}" />`;
  s += `<rect fill="${grid}" height="${BH * G}" stroke="${grid}" width="${bar}" x="${-bar}" y="0" />`;
  s += `<rect fill="${grid}" height="${BH * G}" stroke="${grid}" width="${bar}" x="${G * BW}" y="0" />`;

  for (let xy = 0; xy < board.length; xy++) {
    const stone = board[xy];
    if (stone === -1) continue;
    const sy = Math.floor(xy / BH) * G + G / 2;
    const sx = (xy % BW) * G + G / 2;
    const fill = stone === 0 ? "black" : "white";
    s += `<circle cx="${sx}" cy="${sy}" fill="${fill}" r="${G / 3}" stroke="black" />`;
  }
  s += `</g></g></svg>`;
  return s;
}

// --- chess ------------------------------------------------------------------
// GRID_SIZE 50, 8x8. Light ColorSet: dark squares gray, light white, grid black.
// board[i] > 0 white (1..6 = P,N,B,R,Q,K), < 0 black. When color==1 pgx flips
// the board (negate + mirror columns) so it always draws from white's side.
const PIECES = ["", "wP", "wN", "wB", "wR", "wQ", "wK", "P", "N", "B", "R", "Q", "K"];
const FILE_CHAR = ["a", "b", "c", "d", "e", "f", "g", "h"];

function flipChessBoard(board) {
  // board2[r*8 + c] = -board[r*8 + (7 - c)]   (pgx _flip on the board field)
  const out = new Array(64);
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) out[r * 8 + c] = -board[r * 8 + (7 - c)];
  return out;
}

function renderChess(board, color) {
  const G = 50,
    BW = 8,
    BH = 8,
    W = (BW + 1) * G,
    H = (BH + 1) * G;
  const grid = "black";
  if (color === 1) board = flipChessBoard(board);

  let s = SVG_OPEN(W, H);
  s += `<rect fill="white" height="${H}" width="${(BW + 1.5) * G}" x="0" y="0" />`; // root bg
  s += `<g transform="scale(1.0)">`;
  s += `<rect fill="white" height="${H}" width="${W}" x="0" y="0" />`;
  s += `<g transform="translate(10,0) translate(${G / 2},${G / 2})">`;

  // squares
  for (let i = 0; i < BW * BH; i++) {
    const fill = Math.floor(i / BH) % 2 !== i % 2 ? "gray" : "white";
    const x = i % BW,
      y = Math.floor(i / BH);
    s += `<rect fill="${fill}" height="${G}" width="${G}" x="${x * G}" y="${y * G}" />`;
  }
  // border
  s += `<rect fill="none" height="${BH * G}" stroke="${grid}" stroke-width="3px" width="${BW * G}" x="0" y="0" />`;
  // coordinates (rank + file interleaved, matching pgx order)
  s += `<g fill="${grid}" id="cord">`;
  for (let i = 0; i < BW; i++) {
    s += `<text font-family="Serif" font-size="20px" x="${-0.3 * G}" y="${(i + 0.6) * G}">${8 - i}</text>`;
    s += `<text font-family="Serif" font-size="20px" x="${(i + 0.4) * G}" y="${8.35 * G}">${FILE_CHAR[i]}</text>`;
  }
  s += `</g>`;
  // pieces
  s += `<g>`;
  for (let i = 0; i < 64; i++) {
    let pi = board[i];
    if (pi === 0) continue;
    if (pi < 0) pi = -pi + 6;
    const uri = CHESS_PIECES[PIECES[pi]];
    const x = Math.floor(i / BH),
      y = 7 - (i % BH);
    s += `<image height="${G * 0.8}" width="${G * 0.8}" x="${x * G + 5}" xlink:href="${uri}" y="${y * G + 5}" />`;
  }
  s += `</g>`;

  s += `</g></g></svg>`;
  return s;
}

export function renderSVG(rec) {
  switch (rec.env_id) {
    case "tic_tac_toe":
      return renderTicTacToe(rec.board);
    case "connect_four":
      return renderConnectFour(rec.board);
    case "chess":
      return renderChess(rec.board, rec.color ?? 0);
    default:
      throw new Error(`no renderer for env_id '${rec.env_id}'`);
  }
}
