document.addEventListener('DOMContentLoaded', function () {
    let serverAddress = '127.0.0.1';
    const outputEl = document.getElementById('output');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const notifArea = document.getElementById('notif-area');
    const copyBtn = document.getElementById('copyBtn');

    function showLoading(show) {
      loadingSpinner.style.display = show ? 'block' : 'none';
    }
    function showNotif(msg, type) {
      notifArea.innerHTML = `<div class="notif notif-${type}">${msg}</div>`;
      setTimeout(() => notifArea.innerHTML = '', 2500);
    }
    function setSummary(text) {
      outputEl.innerHTML = text
        .trim()
        .replace(/\n/g, '<br>')
        .replace(/  /g, '&nbsp;&nbsp;');
      outputEl.scrollTop = outputEl.scrollHeight;
    }
    copyBtn.addEventListener('click', () => {
      const text = outputEl.innerText;
      if (text && text.trim() && text !== 'Summary will appear here...') {
        navigator.clipboard.writeText(text);
        showNotif('Summary copied!', 'success');
      } else {
        showNotif('Nothing to copy.', 'error');
      }
    });

    document.getElementById('summarizeBtn').addEventListener('click', async () => {
      notifArea.innerHTML = '';
      setSummary('');
      showLoading(true);
      try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            try {
              return document.body?.innerText || 'EMPTY';
            } catch (e) {
              return 'SCRIPT_ERROR';
            }
          }
        });
        const pageText = results?.[0]?.result || '';
        if (pageText === 'SCRIPT_ERROR') {
          setSummary('Could not access page content (script error).');
          showLoading(false);
          showNotif('Could not access page content.', 'error');
          return;
        }
        if (!pageText || pageText.trim() === 'EMPTY') {
          setSummary('No page text found.');
          showLoading(false);
          showNotif('No page text found.', 'error');
          return;
        }
        showNotif('Summarizing...', 'success');
        const res = await fetch(`http://${serverAddress}:7864/summarize_stream_status`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: pageText }),
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let resultText = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          resultText += chunk;
          setSummary(resultText);
        }
        showLoading(false);
        showNotif('Summary complete!', 'success');
      } catch (err) {
        showLoading(false);
        setSummary('Failed to get summary.\n' + err.message);
        showNotif('Failed to get summary.', 'error');
        console.error(' Fetch error:', err);
      }
    });
});