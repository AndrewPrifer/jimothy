// Synthetic smoke-test data. Categories are generation targets, never human gold labels.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: { out: { type: 'string', default: 'datasets/email-300' }, model: { type: 'string', default: 'openai/gpt-5-nano' } } });
const out = resolve(values.out), model = values.model;
const task = JSON.parse(await readFile(new URL('../examples/email/task.json', import.meta.url), 'utf8'));
const categories = Object.keys(task.questions.category.criteria);
const themes = ['home and everyday errands', 'work and professional life', 'travel and local events', 'hobbies and creative projects', 'education and volunteering', 'fitness and outdoor activities', 'household services and technology', 'food and entertainment', 'clubs and neighbourhood life', 'small businesses and family plans'];
const emailSchema = { type: 'object', additionalProperties: false, properties: {
  from: { type: 'string' }, subject: { type: 'string' }, body: { type: 'string' },
}, required: ['from', 'subject', 'body'] };
const schema = { type: 'object', additionalProperties: false, properties: Object.fromEntries(categories.map(c => [c, { type: 'array', items: emailSchema, minItems: 5, maxItems: 5 }])), required: categories };
await mkdir(join(out, 'generation'), { recursive: true });
const batches = [], inputs = [], targets = [], train = [], test = [], seen = new Set();
const started = performance.now();
for (let batch = 0; batch < themes.length; batch++) {
  const prompt = `Generate 30 distinct fictional English emails for a local email-classifier smoke test: exactly five for each category below. Theme for this batch: ${themes[batch]}. Vary senders, sentence structures, formality and intent; mix short and moderately long messages. Each body must be 20–65 words, subject under 12 words. Use fictional names and example.com addresses. Do not place category names, answer labels, or explanations in emails. Include some realistic boundary cases (e.g. a receipt with an offer footer, a personal buying discussion, an informational newsletter, and a discussion digest), while keeping the main purpose discernible. Do not repeat a template with only names changed. Return the specified JSON object.\n${JSON.stringify(task.questions.category)}`;
  const fingerprint = createHash('sha256').update(JSON.stringify({ model, prompt, schema })).digest('hex');
  const path = join(out, 'generation', `${String(batch + 1).padStart(2, '0')}.json`);
  let saved;
  try { saved = JSON.parse(await readFile(path, 'utf8')); if (saved.fingerprint !== fingerprint) throw new Error('Generation cache differs; choose another --out directory.'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (!saved) {
    const key = process.env.AI_GATEWAY_API_KEY;
    if (!key) throw new Error('Set AI_GATEWAY_API_KEY.');
    for (let attempt = 0; ; attempt++) {
      const response = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', {
        method: 'POST', redirect: 'error', signal: AbortSignal.timeout(120_000),
        headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], reasoning_effort: 'minimal', max_completion_tokens: 7000,
          response_format: { type: 'json_schema', json_schema: { name: 'email_batch', strict: true, schema } } }),
      });
      if (!response.ok) {
        await response.body?.cancel();
        if (attempt < 2 && (response.status === 429 || response.status >= 500)) { await new Promise(r => setTimeout(r, 1000 * 2 ** attempt)); continue; }
        throw new Error(`GPT generation failed: HTTP ${response.status}. No credentials or response body logged.`);
      }
      const raw = await response.json();
      if (raw.choices?.[0]?.finish_reason !== 'stop') throw new Error('Generation was incomplete; no dataset was exported.');
      saved = { fingerprint, model: raw.model ?? model, prompt, data: JSON.parse(raw.choices[0].message.content), usage: raw.usage, providerMetadata: raw.provider_metadata ?? raw.providerMetadata };
      await writeFile(path, JSON.stringify(saved, null, 2) + '\n', { flag: 'wx' });
      break;
    }
  }
  batches.push(saved);
  for (const category of categories) {
    const emails = saved.data[category];
    if (!Array.isArray(emails) || emails.length !== 5) throw new Error(`Batch ${batch + 1}: expected five ${category} emails.`);
    emails.forEach((email, i) => {
      if (Object.keys(email).sort().join(',') !== 'body,from,subject' || Object.values(email).some(v => typeof v !== 'string' || !v.trim())) throw new Error('Invalid generated email.');
      const text = JSON.stringify(email), contentKey = email.body.toLowerCase().replace(/\s+/g, ' ').trim();
      if (text.length > 1800 || seen.has(contentKey)) throw new Error('Duplicate or excessively long generated email.');
      seen.add(contentKey);
      const row = { id: `email-${String(inputs.length + 1).padStart(3, '0')}`, state: email };
      inputs.push(row); (i === 4 ? test : train).push(row);
      targets.push({ id: row.id, intendedCategory: category, generator: saved.model });
    });
  }
  console.error(`Generated ${inputs.length}/300 emails`);
}
const jsonl = rows => rows.map(row => JSON.stringify(row)).join('\n') + '\n';
await writeFile(join(out, 'inputs.jsonl'), jsonl(inputs));
await writeFile(join(out, 'train.inputs.jsonl'), jsonl(train));
await writeFile(join(out, 'test.inputs.jsonl'), jsonl(test));
await writeFile(join(out, 'generation-targets.jsonl'), jsonl(targets));
await writeFile(join(out, 'task.json'), JSON.stringify(task, null, 2) + '\n');
const usage = batches.reduce((sum, batch) => ({ inputTokens: sum.inputTokens + (batch.usage?.prompt_tokens ?? 0), outputTokens: sum.outputTokens + (batch.usage?.completion_tokens ?? 0) }), { inputTokens: 0, outputTokens: 0 });
const report = { model, count: inputs.length, train: train.length, test: test.length, categories: Object.fromEntries(categories.map(c => [c, 50])), usage, elapsedMs: performance.now() - started,
  note: 'Synthetic generation targets are not human reference labels. The 60 test inputs were selected before teacher labeling and model training.' };
await writeFile(join(out, 'generation-report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ directory: out, ...report }, null, 2));
