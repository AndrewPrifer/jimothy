export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type QuestionType = 'choice' | 'boolean' | 'noul' | 'score';
export interface Question {
  type: QuestionType;
  instructions: Json;
  criteria?: Json;
}
export interface Task {
  questionId: string;
  question: Question;
  labels: string[];
}
export interface Example {
  state: Json;
  text: string;
  target: number[];
  teacher?: number[];
  humanLabel?: number;
  id?: string;
  group?: string;
  teacherModel?: string;
}
export interface Vector { indices: Uint32Array; values: Float32Array }
export interface TfidfConfig {
  kind: 'tfidf';
  vocabulary: string[];
  idf: number[];
  maxTokens: number;
}
export interface MiniLMConfig {
  kind: 'minilm';
  dimensions: 384;
  maxTokens: 256;
  longInput?: 'chunk';
  directory: 'encoder';
  modelId: string;
  revision: string;
  dtype: 'q8';
}
export type FeatureConfig = TfidfConfig | MiniLMConfig;
export interface Head { weights: number[][]; bias: number[] }
export interface Calibration {
  method: 'temperature';
  temperature: number;
  status: 'fitted' | 'insufficient_data';
}
/** Advisory cutoff on maxProbability, never applied during inference. */
export interface ThresholdRecommendation {
  threshold: number | null;
  status: 'ready' | 'insufficient_data' | 'target_not_met';
  targetAccuracy: number;
}
export interface Manifest {
  format: 'jev-distill';
  version: 3;
  id: string;
  createdAt: string;
  task: Task;
  preprocessing: 'canonical-json-v1';
  features: FeatureConfig;
  head: Head;
  calibration: Calibration;
  thresholdRecommendation: ThresholdRecommendation;
  files: Record<string, string>;
}
export type Answer =
  | { type: 'choice'; choice: string; probabilities: Record<string, number> }
  | { type: 'boolean'; probability: number }
  | { type: 'noul'; noul: number }
  | { type: 'score'; score: number; probabilities: Record<string, number> };
export interface Prediction {
  answer: Answer;
  maxProbability: number;
}
export interface FeatureExtractor {
  config: FeatureConfig;
  encode(texts: string[]): Promise<Vector[]>;
  dispose(): Promise<void>;
}
