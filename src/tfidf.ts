import type { FeatureExtractor, TfidfConfig, Vector } from './types.js';

function tokens(text: string, maxTokens: number): string[] {
  const words = text.toLowerCase().normalize('NFKC').match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) ?? [];
  if (words.length > maxTokens) throw new Error(`Input has ${words.length} tokens; this model supports ${maxTokens}. Extract relevant fields first.`);
  return words;
}
function terms(text: string, maxTokens: number): string[] {
  const words = tokens(text, maxTokens);
  return [...words.map(w => `w:${w}`), ...words.slice(1).map((w, i) => `b:${words[i]} ${w}`)];
}
export function fitTfidf(texts: string[], maxFeatures: number, maxTokens = 4096): TfidfConfig {
  const df = new Map<string, number>();
  for (const text of texts) for (const term of new Set(terms(text, maxTokens))) df.set(term, (df.get(term) ?? 0) + 1);
  const vocabulary = [...df.keys()].sort((a, b) => df.get(b)! - df.get(a)! || a.localeCompare(b)).slice(0, maxFeatures);
  if (vocabulary.length === 0) throw new Error('No word features found in the training data.');
  return { kind: 'tfidf', vocabulary, idf: vocabulary.map(term => Math.log((1 + texts.length) / (1 + df.get(term)!)) + 1), maxTokens };
}
export function tfidfExtractor(config: TfidfConfig): FeatureExtractor {
  const lookup = new Map(config.vocabulary.map((term, i) => [term, i]));
  return {
    config,
    async encode(texts) {
      return texts.map(text => {
        const counts = new Map<number, number>();
        for (const term of terms(text, config.maxTokens)) {
          const index = lookup.get(term);
          if (index !== undefined) counts.set(index, (counts.get(index) ?? 0) + 1);
        }
        const indices = Uint32Array.from([...counts.keys()].sort((a, b) => a - b));
        const values = Float32Array.from(indices, index => (1 + Math.log(counts.get(index)!)) * config.idf[index]);
        const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
        for (let i = 0; i < values.length; i++) values[i] /= norm;
        return { indices, values } satisfies Vector;
      });
    },
    async dispose() {},
  };
}
