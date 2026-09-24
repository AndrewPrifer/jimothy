import type { FeatureExtractor, MiniLMConfig, Vector } from './types.js';

/** One embedding path for training, Node inference and browser inference. */
export function encoderExtractor(config: MiniLMConfig, pipe: any): FeatureExtractor {
  return { config,
    async encode(texts) {
      const inputs: string[] = [];
      const groups: { start: number; count: number }[] = [];
      const tokenize = (text: string, special = true): number[] => pipe.tokenizer(text, {
        truncation: false, padding: false, return_tensor: false, add_special_tokens: special,
      }).input_ids;
      for (let i = 0; i < texts.length; i++) {
        const start = inputs.length;
        const ids = tokenize(texts[i]);
        if (ids.length <= config.maxTokens) inputs.push(texts[i]);
        else {
          if (config.longInput !== 'chunk') throw new Error(`Input ${i + 1} has ${ids.length} wordpieces; MiniLM supports ${config.maxTokens}. Extract relevant fields first.`);
          const content = tokenize(texts[i], false);
          const window = config.maxTokens - tokenize('').length;
          if (window < 1) throw new Error('Encoder token limit cannot fit content and special tokens.');
          if (!content.length) throw new Error(`Input ${i + 1} has no content tokens to split.`);
          for (let offset = 0; offset < content.length;) {
            let end = Math.min(offset + window, content.length);
            let chunk = '';
            // Decoding a token window can change token boundaries; check the text the pipeline will actually receive.
            while (end > offset) {
              chunk = pipe.tokenizer.decode(content.slice(offset, end), { skip_special_tokens: true, clean_up_tokenization_spaces: false });
              if (chunk && tokenize(chunk).length <= config.maxTokens) break;
              end--;
            }
            if (end === offset) throw new Error(`Input ${i + 1} cannot be split into valid token windows.`);
            inputs.push(chunk);
            offset = end;
          }
        }
        groups.push({ start, count: inputs.length - start });
      }
      const rows: number[][] = [];
      for (let start = 0; start < inputs.length; start += 32) {
        const batch = (await pipe(inputs.slice(start, start + 32), { pooling: 'mean', normalize: true })).tolist() as number[][];
        for (const row of batch) {
          if (row.length !== config.dimensions || row.some(v => !Number.isFinite(v))) throw new Error('Encoder produced invalid embeddings.');
          rows.push(row);
        }
      }
      if (rows.length !== inputs.length) throw new Error('Encoder returned the wrong number of embeddings.');
      return groups.map(({ start, count }): Vector => {
        const values = Float32Array.from(rows[start]);
        if (count > 1) {
          for (let j = 1; j < count; j++) for (let k = 0; k < values.length; k++) values[k] += rows[start + j][k];
          const norm = Math.hypot(...values);
          if (!norm) throw new Error('Chunk embeddings cancel to a zero vector.');
          for (let k = 0; k < values.length; k++) values[k] /= norm;
        }
        return { indices: Uint32Array.from(values, (_, i) => i), values };
      });
    },
    async dispose() { await pipe.dispose(); },
  };
}
