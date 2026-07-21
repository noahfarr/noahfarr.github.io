/* tslint:disable */
/* eslint-disable */

export class CompileResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly manifest_json: string;
    readonly wasm_bytes: Uint8Array;
}

/**
 * WebGPU-backend compile result: a JSON execution plan (shaders, step
 * program, arena size, const layout, I/O manifest) plus one concatenated
 * constants blob (kept out of the JSON to avoid base64 for weights).
 */
export class WebGpuCompileResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    const_bytes: Uint8Array;
    plan_json: string;
}

/**
 * Compile StableHLO text (jax.export `.mlir_module()` output) to a wasm
 * module + manifest. Throws a JS error with a rendered diagnostic on
 * failure.
 */
export function compile(text: string): CompileResult;

/**
 * Compile StableHLO text for the WebGPU backend. The JS layer owns device
 * setup and execution. Throws a rendered diagnostic on failure (including
 * "op X is not supported on the WebGPU backend" for uncovered constructs).
 */
export function compile_webgpu(text: string): WebGpuCompileResult;

/**
 * Compile with explicit wasm feature flags (from runtime detection).
 */
export function compile_with(text: string, simd128: boolean, relaxed_simd: boolean): CompileResult;

/**
 * Probe module for `WebAssembly.validate`-based relaxed-SIMD detection.
 */
export function relaxed_simd_probe(): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_compileresult_free: (a: number, b: number) => void;
    readonly __wbg_get_webgpucompileresult_const_bytes: (a: number) => [number, number];
    readonly __wbg_get_webgpucompileresult_plan_json: (a: number) => [number, number];
    readonly __wbg_set_webgpucompileresult_const_bytes: (a: number, b: number, c: number) => void;
    readonly __wbg_set_webgpucompileresult_plan_json: (a: number, b: number, c: number) => void;
    readonly __wbg_webgpucompileresult_free: (a: number, b: number) => void;
    readonly compile: (a: number, b: number) => [number, number, number];
    readonly compile_webgpu: (a: number, b: number) => [number, number, number];
    readonly compile_with: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly compileresult_manifest_json: (a: number) => [number, number];
    readonly compileresult_wasm_bytes: (a: number) => [number, number];
    readonly relaxed_simd_probe: () => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
