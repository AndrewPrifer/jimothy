// A classic script can explain file:// use even when module loading is blocked.
if (location.protocol === 'file:') {
  const error = document.getElementById('error');
  error.textContent = 'Run npm run demo:browser and open http://127.0.0.1:4319.';
  error.hidden = false;
  document.getElementById('status').textContent = 'Server required';
}
