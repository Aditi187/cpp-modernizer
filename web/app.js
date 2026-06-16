(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const esc = s => {
    const d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
  };
  const fmt_ms = ms => ms < 1000 ? ms + 'ms' : (ms / 1000).toFixed(1) + 's';

  let lastResult = null;

// Dark mode toggle
const savedDarkMode = localStorage.getItem('darkMode') === 'true';
if (savedDarkMode) {
  document.body.classList.add('dark-mode');
}

$('btnDarkMode').addEventListener('click', () => {
  const isDark = document.body.classList.toggle('dark-mode');
  localStorage.setItem('darkMode', isDark);
});

$('btnModernize').addEventListener('click', runModernize);

    // Auth token logic
    const tokenInput = $('authToken');
    if (tokenInput) {
      let savedToken = sessionStorage.getItem('apiAuthToken');
      if (!savedToken) {
        savedToken = prompt("Please enter your API Auth Token (see .env API_AUTH_TOKEN):") || "";
        sessionStorage.setItem('apiAuthToken', savedToken.trim());
      }
      tokenInput.value = savedToken;
      tokenInput.addEventListener('input', () => {
        sessionStorage.setItem('apiAuthToken', tokenInput.value.trim());
      });
    }

    async function runModernize() {
      const code = $('codeInput').value.trim();
      if (!code) { toast('Paste some C++ code first', 'error'); return; }
  
      const btn = $('btnModernize');
      btn.disabled = true;
      btn.textContent = 'Processing...';
      
      resetResults();
      resetPipeline();
      setPipeline('analyzer', 'active');
  
      try {
        const token = sessionStorage.getItem('apiAuthToken');
        const headers = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const resp = await fetch('/modernize/stream', {
          method: 'POST',
          headers,
          body: JSON.stringify({
            code,
            filename: 'input.cpp',
            skip_verify: $('skipVerify').checked,
          }),
        });

      if (!resp.ok) {
        if (resp.status === 401) {
          // Clear bad token so user is prompted again
          sessionStorage.removeItem('apiAuthToken');
          throw new Error('Invalid API token. Reload the page to re-enter your token.');
        }
        if (resp.status === 429) {
          throw new Error('Rate limit exceeded. Please wait a moment and try again.');
        }
        throw new Error(`Server error (HTTP ${resp.status}). Check that the API server is running.`);
      }

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            const d = JSON.parse(raw);
            if (d.node === 'done') {
              // Mark the verifier stage complete before rendering results
              setPipeline('verifier', 'completed');
              lastResult = d.response;
            } else if (d.node === 'done_error') {
              // Pipeline ended with an error - already shown via d.error event
              lastResult = null;
            } else if (d.error) {
              throw new Error(d.error);
            } else if (d.node) {
              advancePipeline(d.node);
            }
          } catch (e) { console.warn('SSE parse:', e); throw e; }
        }
      }

      if (!lastResult) throw new Error('Pipeline returned no result');
      finishPipeline(lastResult.success);
      renderResults(lastResult);
      toast('Modernization complete!', 'success');

    } catch (err) {
      finishPipeline(false);
      toast(`Error: ${err.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Modernize';
    }
  }

  // =========================================================================
  // Pipeline
  // =========================================================================
  const STAGES = ['analyzer', 'planner', 'modernizer', 'semantic_guard', 'verifier'];

  function setPipeline(name, cls) {
    const el = document.querySelector(`.pipeline-step[data-step="${name}"]`);
    if (!el) return;
    el.classList.remove('active', 'completed', 'failed');
    if (cls) el.classList.add(cls);
    const lbl = $('pipelineStageLabel');
    if (lbl) {
      if (cls === 'active') lbl.textContent = name.charAt(0).toUpperCase() + name.slice(1) + '...';
    }
  }

  function advancePipeline(completed) {
    setPipeline(completed, 'completed');
    const i = STAGES.indexOf(completed);
    if (i !== -1 && i < STAGES.length - 1) setPipeline(STAGES[i + 1], 'active');
  }

  function resetPipeline() {
    STAGES.forEach(s => {
      const el = document.querySelector(`.pipeline-step[data-step="${s}"]`);
      if (el) el.className = 'pipeline-step';
    });
    const lbl = $('pipelineStageLabel');
    if (lbl) lbl.textContent = 'Idle';
  }

  function finishPipeline(ok) {
    STAGES.forEach(s => {
      const el = document.querySelector(`.pipeline-step[data-step="${s}"]`);
      if (el) { el.classList.remove('active'); el.classList.add(ok ? 'completed' : 'failed'); }
    });
    const lbl = $('pipelineStageLabel');
    if (lbl) lbl.textContent = ok ? 'Complete' : 'Failed';
  }

  // =========================================================================
  // Results & Diff
  // =========================================================================
  function renderResults(data) {
    const score = data.score || 0;
    const pct = (score * 100).toFixed(1) + '%';
    const cls = score >= 0.8 ? 'good' : score >= 0.5 ? 'warn' : 'bad';

    const mScore = $('mScoreVal');
    if (mScore) { mScore.textContent = pct; mScore.className = `metric-value ${cls}`; }

    const mSafety = $('mSafetyVal');
    if (mSafety) {
      const s = data.safety_rating || '—';
      const sc = s === 'SAFE' ? 'good' : s === 'REVIEW' ? 'warn' : 'bad';
      mSafety.textContent = s;
      mSafety.className = `metric-value ${sc}`;
    }

    const mP = $('mPatterns');
    if (mP) mP.textContent = data.legacy_patterns_found ?? '—';
    const mT = $('mTime');
    if (mT) mT.textContent = fmt_ms(data.processing_time_ms || 0);

    $('metricsPanel').style.opacity = '1';

    // Show attribution badge
    if (data.attribution) {
      const container = $('attributionContainer');
      const badge = $('attributionBadge');
      const attribution = data.attribution;
      
      let badgeText = 'Rule-based only';
      let badgeClass = 'rule-based';
      let icon = '✓';
      
      if (attribution.includes('llm')) {
        if (attribution.includes('verified')) {
          badgeText = 'LLM + Verified';
        } else {
          const modelMatch = attribution.match(/llm:([^\s]+)/);
          badgeText = modelMatch ? `LLM: ${modelMatch[1]}` : 'LLM';
        }
        badgeClass = 'llm';
        icon = '🤖';
      }
      
      badge.textContent = icon + ' ' + badgeText;
      badge.className = `attribution-badge ${badgeClass}`;
      container.style.display = 'flex';
    }

    // Show verification details
    if (data.compiler_status || data.safety_rating) {
      const verDetails = $('verificationDetails');
      verDetails.style.display = 'block';
      
      const compilerEl = $('verificationCompiler');
      if (compilerEl) {
        compilerEl.textContent = data.compiler_status || '—';
        compilerEl.style.color = data.compiler_status?.includes('SUCCESS') ? 'var(--success)' : 'var(--danger)';
      }
      
      const semanticEl = $('verificationSemantic');
      if (semanticEl) {
        semanticEl.textContent = data.safety_rating || '—';
        semanticEl.style.color = data.safety_rating === 'SAFE' ? 'var(--success)' : data.safety_rating === 'REVIEW' ? 'var(--warning)' : 'var(--danger)';
      }
      
      // Populate compiler output if available
      const compilerOutput = $('compilerOutput');
      if (compilerOutput && data.compiler_output) {
        compilerOutput.textContent = typeof data.compiler_output === 'string' 
          ? data.compiler_output 
          : JSON.stringify(data.compiler_output, null, 2);
      }
      
      const sanitizerList = $('sanitizerFindings');
      const sanitizerContainer = $('sanitizerContainer');
      if (sanitizerList && sanitizerContainer) {
        if (data.sanitizer_findings && data.sanitizer_findings.length > 0) {
          sanitizerList.innerHTML = data.sanitizer_findings.map(f => `<li>${esc(f)}</li>`).join('');
          sanitizerContainer.style.display = 'flex';
        } else {
          sanitizerContainer.style.display = 'none';
        }
      }
    }

    renderSplitDiff(data.original_code, data.modernized_code);
    renderUnifiedDiff(data.diff ? data.diff.diff_preview : null);
    
    // Switch to split view initially
    $('btnSplit').click();
  }

  function resetResults() {
    $('metricsPanel').style.opacity = '0';
    $('attributionContainer').style.display = 'none';
    $('verificationDetails').style.display = 'none';
    $('diffContainer').innerHTML = `
      <div class="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#d0d7de" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
        <p>Modernizing code...</p>
      </div>`;
  }

  function codeToHtml(code) {
    if (!code) return '<p class="empty-state">No code</p>';
    return '<div class="code-render">' + code.split('\n').map((line, i) =>
      `<div class="cl"><span class="cn">${i + 1}</span><span class="ct">${esc(line)}</span></div>`
    ).join('') + '</div>';
  }

  function renderSplitDiff(orig, mod) {
    const html = `
      <div class="diff-split">
        <div class="diff-col">
          <div class="diff-col-header">Before (Legacy C)</div>
          ${codeToHtml(orig)}
        </div>
        <div class="diff-col">
          <div class="diff-col-header">After (C++17)</div>
          ${codeToHtml(mod)}
        </div>
      </div>
    `;
    $('diffContainer').dataset.split = html;
  }

  function renderUnifiedDiff(text) {
    if (!text) { $('diffContainer').dataset.unified = '<p class="empty-state">No diff available</p>'; return; }
    
    const html = '<div class="code-render">' + text.split('\n').map((l, i) => {
      let cls = '';
      if (l.startsWith('+') && !l.startsWith('+++')) cls = 'c-add';
      else if (l.startsWith('-') && !l.startsWith('---')) cls = 'c-rem';
      return `<div class="cl ${cls}"><span class="cn">${i + 1}</span><span class="ct">${esc(l)}</span></div>`;
    }).join('') + '</div>';
    
    $('diffContainer').dataset.unified = html;
  }

  $('btnSplit').addEventListener('click', () => {
    $('diffContainer').style.display = 'block';
    $('verificationDetails').style.display = 'none';
    $('diffContainer').innerHTML = $('diffContainer').dataset.split || '';
    $('btnSplit').classList.add('active');
    $('btnUnified').classList.remove('active');
    $('btnDetails').classList.remove('active');
  });

  $('btnUnified').addEventListener('click', () => {
    $('diffContainer').style.display = 'block';
    $('verificationDetails').style.display = 'none';
    $('diffContainer').innerHTML = $('diffContainer').dataset.unified || '';
    $('btnUnified').classList.add('active');
    $('btnSplit').classList.remove('active');
    $('btnDetails').classList.remove('active');
  });

  $('btnDetails').addEventListener('click', () => {
    $('diffContainer').style.display = 'none';
    $('verificationDetails').style.display = 'block';
    // ensure content is shown if header was not clicked
    const content = document.querySelector('.verification-content');
    if (content) content.style.display = 'block';
    $('btnDetails').classList.add('active');
    $('btnSplit').classList.remove('active');
    $('btnUnified').classList.remove('active');
  });

  // Verification details header toggle
  const verificationHeader = document.querySelector('.verification-header');
  if (verificationHeader) {
    verificationHeader.addEventListener('click', function() {
      this.classList.toggle('expanded');
      const content = document.querySelector('.verification-content');
      if (content) {
        content.style.display = content.style.display === 'none' ? 'block' : 'none';
      }
    });
  }

  $('btnCopy').addEventListener('click', () => {
    if (!lastResult) return;
    navigator.clipboard.writeText(lastResult.modernized_code)
      .then(() => toast('Copied to clipboard', 'success'));
  });

  function toast(msg, type = 'info') {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    $('toastContainer').appendChild(t);
    setTimeout(() => {
      t.style.opacity = '0';
      t.style.transform = 'translateY(4px)';
      t.style.transition = 'all 0.25s';
      setTimeout(() => t.remove(), 250);
    }, 3000);
  }

})();
