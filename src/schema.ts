import { createHash } from 'node:crypto';
export * from './schema-core.js';

export function digest(value: string | Uint8Array): string {
  return createHash('sha256').update(value).digest('hex');
}
