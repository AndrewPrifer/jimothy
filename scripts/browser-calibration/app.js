const button = document.querySelector('#run'), status = document.querySelector('#status'), results = document.querySelector('#results');
button.onclick = () => {
  button.disabled = true;
  const worker = new Worker('/worker.js', { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.progress) { status.textContent = data.progress; return; }
    results.textContent = JSON.stringify(data, null, 2);
    if (data.complete || data.error) {
      status.textContent = data.error ? `Failed: ${data.error}` : `Complete. Saved to ${data.saved}`;
      worker.terminate();
    }
  };
  worker.onerror = event => { status.textContent = `Failed: ${event.message}`; worker.terminate(); };
  worker.postMessage({ start: true });
};
