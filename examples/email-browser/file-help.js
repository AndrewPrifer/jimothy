if (location.protocol === 'file:') {
  document.getElementById('status').textContent = '';
  const error = document.getElementById('error');
  error.textContent = 'Run npm run demo:email and open http://127.0.0.1:4320.';
  error.hidden = false;
}
