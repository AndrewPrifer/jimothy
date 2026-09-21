const copyStatus = document.getElementById('copy-status');
for (const button of document.querySelectorAll('[data-copy]')) {
  let reset;
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      button.dataset.copied = 'true';
      copyStatus.textContent = 'Install command copied.';
      clearTimeout(reset);
      reset = setTimeout(() => { delete button.dataset.copied; copyStatus.textContent = ''; }, 2000);
    } catch {
      copyStatus.textContent = 'Select and copy the install command.';
      const range = document.createRange();
      range.selectNodeContents(button.querySelector('code'));
      const selection = window.getSelection();
      selection.removeAllRanges(); selection.addRange(range);
    }
  });
}
for (const tablist of document.querySelectorAll('[role="tablist"]')) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  function select(tab) {
    for (const item of tabs) {
      const active = item === tab;
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
      document.getElementById(item.getAttribute('aria-controls')).hidden = !active;
    }
  }
  for (const tab of tabs) {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', event => {
      const index = tabs.indexOf(tab);
      const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length
        : event.key === 'ArrowLeft' ? (index - 1 + tabs.length) % tabs.length
          : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
      if (next === null) return;
      event.preventDefault(); select(tabs[next]); tabs[next].focus();
    });
  }
}
