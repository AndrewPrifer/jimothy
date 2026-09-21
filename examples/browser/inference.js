// Small pure helpers shared by the demo worker and its smoke tests.
export function parseInputs(value, mode) {
  if (typeof value !== 'string' || value.length > 100_000) throw new Error('Please use a shorter input (under 100,000 characters).');
  const texts = mode === 'batch' ? value.split(/\r?\n/).map(line => line.trim()).filter(Boolean) : [value.trim()];
  if (!texts.length || texts.some(text => !text)) throw new Error('Enter a message to classify.');
  if (texts.length > 50) throw new Error('Use up to 50 messages per batch, one per line.');
  return texts;
}

// Presentation only: probabilities and answer formatting come from the SDK.
export function makePrediction({ answer, maxProbability }, { task, id }) {
  return { choice: answer.choice, maxProbability,
    ranked: Object.entries(answer.probabilities).map(([label, probability]) => ({ label, probability })).sort((a, b) => b.probability - a.probability).slice(0, 5),
    response: { model: id, answers: { [task.questionId]: answer } },
  };
}
