import { modelIdentity, parseManifest } from './manifest.js';
import { canonical, object, probability } from './schema-core.js';
import type { Manifest } from './types.js';

export async function sha256(value: string | ArrayBuffer): Promise<string> {
  const bytes = typeof value === 'string' ? new TextEncoder().encode(value) : value;
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
}
export async function readBrowserBundle(manifestUrl: string, progress: (file: string) => void) {
  const download = async (url: string): Promise<ArrayBuffer> => {
    const response = await fetch(url, { signal: AbortSignal.timeout(120_000) });
    if (!response.ok) throw new Error(`Cannot load ${new URL(url).pathname}: HTTP ${response.status}.`);
    return response.arrayBuffer();
  };
  const bytes = await download(manifestUrl);
  const manifest = parseManifest(JSON.parse(new TextDecoder().decode(bytes)));
  if (manifest.id !== `local-${(await sha256(modelIdentity(manifest))).slice(0, 20)}`) throw new Error('Model manifest checksum mismatch.');
  const assets = new Map<string, ArrayBuffer>();
  for (const [file, hash] of Object.entries(manifest.files)) {
    progress(file);
    const url = new URL(file, manifestUrl).href;
    const data = await download(url);
    if (await sha256(data) !== hash) throw new Error(`Checksum mismatch for ${file}; model bundle is damaged.`);
    assets.set(url, data);
  }
  return { manifest, assets, manifestSha256: await sha256(bytes), download };
}

/** Validates the existing FP16 calibration export against this exact model and runtime. */
export async function applyWebGPUCalibration(manifest: Manifest, value: unknown, manifestSha256: string, runtime: unknown): Promise<string> {
  const policy = object(value, 'WebGPU policy');
  if (policy.format !== 'jev-distill-browser-policy-experiment' || policy.version !== 1) throw new Error('Unsupported WebGPU calibration export.');
  const provenance = object(policy.provenance, 'WebGPU provenance');
  const encoder = object(provenance.encoder, 'WebGPU encoder');
  const features = manifest.features;
  if (features.kind !== 'minilm' || provenance.sourceModelId !== manifest.id || provenance.sourceManifestSha256 !== manifestSha256 ||
      provenance.headSha256 !== await sha256(JSON.stringify(manifest.head)) || provenance.tokenizerSha256 !== manifest.files['encoder/tokenizer.json'] ||
      encoder.modelId !== features.modelId || encoder.revision !== features.revision || encoder.dtype !== 'fp16' ||
      typeof encoder.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(encoder.sha256) ||
      canonical(provenance.runtime) !== canonical(runtime) || provenance.device !== 'webgpu' ||
      provenance.preprocessing !== manifest.preprocessing || provenance.pooling !== 'mean' || provenance.normalize !== true || provenance.maxTokens !== features.maxTokens ||
      (provenance.inputPrefix ?? '') !== (features.inputPrefix ?? '')) {
    throw new Error('WebGPU calibration does not match this model, encoder, or browser runtime.');
  }
  const calibration = object(policy.calibration, 'WebGPU calibration');
  if (calibration.method !== 'temperature' || calibration.status !== 'fitted' || typeof calibration.temperature !== 'number' ||
      !Number.isFinite(calibration.temperature) || calibration.temperature < 0.05 || calibration.temperature > 20) throw new Error('Invalid WebGPU calibration.');
  const acceptance = object(policy.acceptance, 'WebGPU recommendation');
  if (acceptance.mode !== 'automatic' || !['ready', 'insufficient_data', 'target_not_met'].includes(String(acceptance.status)) ||
      typeof acceptance.targetAccuracy !== 'number' || !(acceptance.targetAccuracy > 0 && acceptance.targetAccuracy <= 1)) throw new Error('Invalid WebGPU threshold recommendation.');
  probability(policy.threshold, 'WebGPU recommended threshold');
  manifest.calibration = { method: 'temperature', status: 'fitted', temperature: calibration.temperature };
  manifest.thresholdRecommendation = { threshold: acceptance.status === 'ready' ? policy.threshold as number : null,
    status: acceptance.status as 'ready' | 'insufficient_data' | 'target_not_met', targetAccuracy: acceptance.targetAccuracy };
  manifest.id += '-fp16-webgpu';
  return encoder.sha256;
}
