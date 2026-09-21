import { predictHead } from './linear.js';
import { stateText } from './schema-core.js';
import type { Answer, FeatureExtractor, Json, Manifest, Prediction } from './types.js';

export class LocalClassifier {
  #manifest: Manifest;
  #extractor: FeatureExtractor;
  #disposed = false;
  constructor(manifest: Manifest, extractor: FeatureExtractor) {
    this.#manifest = manifest;
    this.#extractor = extractor;
  }
  get metadata() {
    return structuredClone({ id: this.#manifest.id, task: this.#manifest.task,
      backend: this.#manifest.features.kind, thresholdRecommendation: this.#manifest.thresholdRecommendation, createdAt: this.#manifest.createdAt });
  }
  async predict(state: Json): Promise<Prediction> { return (await this.predictBatch([state]))[0]; }
  async predictBatch(states: Json[]): Promise<Prediction[]> {
    if (this.#disposed) throw new Error('Classifier has been disposed.');
    const vectors = await this.#extractor.encode(states.map(s => stateText(s).text));
    const { labels, question } = this.#manifest.task;
    return vectors.map(vector => {
      const p = predictHead(this.#manifest.head, vector, this.#manifest.calibration.temperature);
      const winner = p.indexOf(Math.max(...p)), maxProbability = p[winner];
      const probabilities = Object.fromEntries(labels.map((label, i) => [label, p[i]]));
      let answer: Answer;
      if (question.type === 'choice') answer = { type: 'choice', choice: labels[winner], probabilities };
      else if (question.type === 'boolean') answer = { type: 'boolean', probability: p[1] };
      else if (question.type === 'noul') answer = { type: 'noul', noul: p[1] };
      else answer = { type: 'score', score: p.reduce((sum, value, i) => sum + value * i, 0), probabilities };
      return { answer, maxProbability };
    });
  }
  /** A fixed-task counterpart to Jev's evaluate shape. No prompt or provider is called. */
  async evaluate({ state }: { state: Json }) {
    const prediction = await this.predict(state);
    return {
      model: this.#manifest.id,
      answers: { [this.#manifest.task.questionId]: prediction.answer },
    };
  }
  async dispose(): Promise<void> {
    if (!this.#disposed) { this.#disposed = true; await this.#extractor.dispose(); }
  }
}
