const $ = id => document.getElementById(id);

document.addEventListener('DOMContentLoaded', () => {
  const runsBody = $('runsBody');
  const fileDetailsContainer = $('fileDetailsContainer');
  const filesBody = $('filesBody');
  const runIdLabel = $('runIdLabel');
  const btnReport = $('btnReport');

  // Match system dark mode or saved setting
  const savedDarkMode = localStorage.getItem('darkMode') === 'true';
  if (savedDarkMode) {
    document.body.classList.add('dark-mode');
  }

  // Init token
  const tokenInput = $('apiToken');
  if (tokenInput) {
    tokenInput.value = sessionStorage.getItem('apiAuthToken') || '';
    tokenInput.addEventListener('input', () => {
      sessionStorage.setItem('apiAuthToken', tokenInput.value.trim());
    });
  }

  fetchRuns();

  $('btnLoadRuns').addEventListener('click', fetchRuns);

  async function fetchRuns() {
    const token = tokenInput ? tokenInput.value.trim() : '';
    if (token) sessionStorage.setItem('apiAuthToken', token);
    
    const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
    runsBody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 24px; color: var(--text-tertiary);">Loading runs...</td></tr>`;
    try {
      const res = await fetch('/api/project/runs', { headers });
      if (!res.ok) throw new Error('Failed to fetch runs');
      const data = await res.json();
      renderRuns(data.runs || []);
    } catch (e) {
      runsBody.innerHTML = `<tr><td colspan="8" style="color: var(--danger); text-align: center; padding: 24px;">Error loading runs: ${e.message}</td></tr>`;
    }
  }

  function renderRuns(runs) {
    if (runs.length === 0) {
      runsBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-tertiary); padding: 24px;">No runs found.</td></tr>`;
      return;
    }
    
    runsBody.innerHTML = runs.map(r => `
      <tr onclick="loadRunDetails(${r.id})">
        <td style="font-family: var(--font-mono); font-weight: 500;">#${r.id}</td>
        <td>${new Date(r.started_at).toLocaleString()}</td>
        <td>${r.total_files}</td>
        <td style="color: var(--success); font-weight: 500;">${r.passed}</td>
        <td style="color: var(--danger); font-weight: 500;">${r.failed}</td>
        <td style="color: var(--text-tertiary);">${r.skipped}</td>
        <td style="font-weight: 500;">${r.submitted_by || 'Local CLI'}</td>
        <td><span style="color: var(--accent); font-weight: 500;">View Details &rarr;</span></td>
      </tr>
    `).join('');
  }

  window.loadRunDetails = async function(runId) {
    const token = $('apiToken').value.trim();
    const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
    try {
      const res = await fetch(`/api/project/status/${runId}`, { headers });
      if (!res.ok) throw new Error('Failed to fetch run details');
      const data = await res.json();
      
      runIdLabel.textContent = `(#${runId})`;
      btnReport.href = `/api/project/report/${runId}` + (token ? `?token=${encodeURIComponent(token)}` : '');
      fileDetailsContainer.style.display = 'block';
      
      if (!data.files || data.files.length === 0) {
        filesBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-tertiary); padding: 24px;">No files in this run.</td></tr>`;
        return;
      }
      
      filesBody.innerHTML = data.files.map(f => {
        let statusColor = 'var(--text-secondary)';
        if (f.status === 'done') statusColor = 'var(--success)';
        if (f.status === 'failed') statusColor = 'var(--danger)';
        if (f.status === 'pending') statusColor = 'var(--warning)';
        
        return `
        <tr>
          <td style="font-family: var(--font-mono); font-size: 13px;">${f.path}</td>
          <td><span style="display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; background: ${statusColor}15; color: ${statusColor}; border: 1px solid ${statusColor}30;">${f.status.toUpperCase()}</span></td>
          <td>${f.complexity !== undefined ? f.complexity : '-'}</td>
          <td>${f.llm_called ? '✅ Yes' : '❌ No'}</td>
          <td style="font-size: 13px; font-weight: 500;">${f.attribution || '-'}</td>
          <td>${f.duration_ms || 0}ms</td>
        </tr>
      `}).join('');
      
      fileDetailsContainer.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
      alert('Error loading details: ' + e.message);
    }
  }
});
