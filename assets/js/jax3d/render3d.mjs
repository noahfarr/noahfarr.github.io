// Minimal from-scratch WebGL renderer for the jax3d car scene: flat-shaded
// boxes and spheres under one directional light, orbit camera. Z is up (jax3d
// gravity is -z). No external libraries.
//
// createRenderer(canvas) -> { render(items, camera), resize() }
//   items:  [{ geom:'cube'|'sphere', pos:[3], quat:[x,y,z,w], scale:[3], color:[3] }]
//   camera: { eye:[3], target:[3], fovDeg }

// ---- tiny column-major mat4/mat3 helpers ----------------------------------
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2),
    nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(eye, center, up) {
  let z0 = eye[0] - center[0],
    z1 = eye[1] - center[1],
    z2 = eye[2] - center[2];
  let zl = Math.hypot(z0, z1, z2) || 1;
  z0 /= zl;
  z1 /= zl;
  z2 /= zl;
  let x0 = up[1] * z2 - up[2] * z1,
    x1 = up[2] * z0 - up[0] * z2,
    x2 = up[0] * z1 - up[1] * z0;
  let xl = Math.hypot(x0, x1, x2) || 1;
  x0 /= xl;
  x1 /= xl;
  x2 /= xl;
  const y0 = z1 * x2 - z2 * x1,
    y1 = z2 * x0 - z0 * x2,
    y2 = z0 * x1 - z1 * x0;
  return new Float32Array([
    x0,
    y0,
    z0,
    0,
    x1,
    y1,
    z1,
    0,
    x2,
    y2,
    z2,
    0,
    -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]),
    -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
    -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]),
    1,
  ]);
}
function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
function modelMat(pos, q, s) {
  const [x, y, z, w] = q,
    x2 = x + x,
    y2 = y + y,
    z2 = z + z;
  const xx = x * x2,
    xy = x * y2,
    xz = x * z2,
    yy = y * y2,
    yz = y * z2,
    zz = z * z2,
    wx = w * x2,
    wy = w * y2,
    wz = w * z2;
  return new Float32Array([
    (1 - (yy + zz)) * s[0],
    (xy + wz) * s[0],
    (xz - wy) * s[0],
    0,
    (xy - wz) * s[1],
    (1 - (xx + zz)) * s[1],
    (yz + wx) * s[1],
    0,
    (xz + wy) * s[2],
    (yz - wx) * s[2],
    (1 - (xx + yy)) * s[2],
    0,
    pos[0],
    pos[1],
    pos[2],
    1,
  ]);
}
function normalMat(q, s) {
  const [x, y, z, w] = q,
    x2 = x + x,
    y2 = y + y,
    z2 = z + z;
  const xx = x * x2,
    xy = x * y2,
    xz = x * z2,
    yy = y * y2,
    yz = y * z2,
    zz = z * z2,
    wx = w * x2,
    wy = w * y2,
    wz = w * z2;
  const i0 = 1 / s[0],
    i1 = 1 / s[1],
    i2 = 1 / s[2];
  return new Float32Array([
    (1 - (yy + zz)) * i0,
    (xy + wz) * i0,
    (xz - wy) * i0,
    (xy - wz) * i1,
    (1 - (xx + zz)) * i1,
    (yz + wx) * i1,
    (xz + wy) * i2,
    (yz - wx) * i2,
    (1 - (xx + yy)) * i2,
  ]);
}

// ---- geometry --------------------------------------------------------------
function cubeGeometry() {
  // 6 faces, per-face normal, unit cube spanning -1..1 (scale by half-extents)
  const faces = [
    [
      [1, 0, 0],
      [
        [1, -1, -1],
        [1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
      ],
    ],
    [
      [-1, 0, 0],
      [
        [-1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
      ],
    ],
    [
      [0, 1, 0],
      [
        [-1, 1, -1],
        [-1, 1, 1],
        [1, 1, 1],
        [1, 1, -1],
      ],
    ],
    [
      [0, -1, 0],
      [
        [1, -1, -1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, -1, -1],
      ],
    ],
    [
      [0, 0, 1],
      [
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
      ],
    ],
    [
      [0, 0, -1],
      [
        [-1, 1, -1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
      ],
    ],
  ];
  const pos = [],
    nrm = [];
  for (const [n, v] of faces)
    for (const [a, b, c] of [
      [0, 1, 2],
      [0, 2, 3],
    ])
      for (const k of [a, b, c]) {
        pos.push(...v[k]);
        nrm.push(...n);
      }
  return { pos: new Float32Array(pos), nrm: new Float32Array(nrm), count: pos.length / 3 };
}
function sphereGeometry(lat = 18, lon = 28) {
  const pos = [],
    nrm = [],
    idx = [];
  for (let i = 0; i <= lat; i++) {
    const t = (i / lat) * Math.PI,
      st = Math.sin(t),
      ct = Math.cos(t);
    for (let j = 0; j <= lon; j++) {
      const p = (j / lon) * 2 * Math.PI,
        sp = Math.sin(p),
        cp = Math.cos(p);
      const x = st * cp,
        y = st * sp,
        z = ct;
      pos.push(x, y, z);
      nrm.push(x, y, z);
    }
  }
  for (let i = 0; i < lat; i++)
    for (let j = 0; j < lon; j++) {
      const a = i * (lon + 1) + j,
        b = a + lon + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  return { pos: new Float32Array(pos), nrm: new Float32Array(nrm), idx: new Uint16Array(idx), count: idx.length };
}

const VS = `
attribute vec3 aPos; attribute vec3 aNormal;
uniform mat4 uMVP; uniform mat3 uNormalMat;
varying vec3 vN;
void main(){ vN = uNormalMat * aNormal; gl_Position = uMVP * vec4(aPos, 1.0); }`;
const FS = `
precision mediump float;
varying vec3 vN;
uniform vec3 uColor; uniform vec3 uLightDir;
void main(){
  vec3 N = normalize(vN);
  float diff = max(dot(N, normalize(uLightDir)), 0.0);
  // hemispheric ambient (sky brighter than ground) + directional
  float amb = mix(0.32, 0.5, 0.5 + 0.5 * N.z);
  vec3 col = uColor * (amb + 0.7 * diff);
  gl_FragColor = vec4(pow(col, vec3(0.9)), 1.0);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}

export function createRenderer(canvas) {
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
  if (!gl) throw new Error("WebGL not available");

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);

  const loc = {
    aPos: gl.getAttribLocation(prog, "aPos"),
    aNormal: gl.getAttribLocation(prog, "aNormal"),
    uMVP: gl.getUniformLocation(prog, "uMVP"),
    uNormalMat: gl.getUniformLocation(prog, "uNormalMat"),
    uColor: gl.getUniformLocation(prog, "uColor"),
    uLightDir: gl.getUniformLocation(prog, "uLightDir"),
  };

  function makeMesh(g) {
    const posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, g.pos, gl.STATIC_DRAW);
    const nrmBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
    gl.bufferData(gl.ARRAY_BUFFER, g.nrm, gl.STATIC_DRAW);
    let idxBuf = null;
    if (g.idx) {
      idxBuf = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, g.idx, gl.STATIC_DRAW);
    }
    return { posBuf, nrmBuf, idxBuf, count: g.count };
  }
  const meshes = { cube: makeMesh(cubeGeometry()), sphere: makeMesh(sphereGeometry()) };

  gl.enable(gl.DEPTH_TEST);

  function bind(mesh) {
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.posBuf);
    gl.enableVertexAttribArray(loc.aPos);
    gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.nrmBuf);
    gl.enableVertexAttribArray(loc.aNormal);
    gl.vertexAttribPointer(loc.aNormal, 3, gl.FLOAT, false, 0, 0);
    if (mesh.idxBuf) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.idxBuf);
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr),
      h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function render(items, camera) {
    resize();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.09, 0.1, 0.12, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const aspect = canvas.width / Math.max(1, canvas.height);
    const proj = perspective((camera.fovDeg * Math.PI) / 180, aspect, 0.1, 300);
    const view = lookAt(camera.eye, camera.target, [0, 0, 1]);
    const vp = mul(proj, view);
    gl.uniform3fv(loc.uLightDir, new Float32Array([0.4, 0.5, 1.0]));

    let current = null;
    for (const it of items) {
      const mesh = meshes[it.geom];
      if (mesh !== current) {
        bind(mesh);
        current = mesh;
      }
      const mvp = mul(vp, modelMat(it.pos, it.quat, it.scale));
      gl.uniformMatrix4fv(loc.uMVP, false, mvp);
      gl.uniformMatrix3fv(loc.uNormalMat, false, normalMat(it.quat, it.scale));
      gl.uniform3fv(loc.uColor, new Float32Array(it.color));
      if (mesh.idxBuf) gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
      else gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
    }
  }

  return { render, resize, gl };
}
