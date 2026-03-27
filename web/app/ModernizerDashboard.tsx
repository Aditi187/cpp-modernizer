'use client';

import React, { useState, useEffect } from 'react';
import styles from './modernizer.module.css';

export default function ModernizerDashboard() {
  const [legacyCode, setLegacyCode] = useState('// Paste your legacy C++ code here...\n\nint main() {\n  int* p = (int*)malloc(10 * sizeof(int));\n  free(p);\n  return 0;\n}');
  const [modernizedCode, setModernizedCode] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState<string[]>(['[System] Ready for input...']);

  const addLog = (msg: string) => {
    setLogs(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  const handleModernize = async () => {
    setIsProcessing(true);
    setModernizedCode('');
    addLog('Initiating modernization sequence...');

    try {
      const response = await fetch('https://cpp-modernizer.vercel.app/modernize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: legacyCode,
          source_file: 'web_input.cpp'
        }),
      });

      if (!response.ok) throw new Error('Modernization engine failed');

      const data = await response.json();
      setModernizedCode(data.modernized_code);
      addLog('Modernization COMPLETE. Applying final diff...');
    } catch (error) {
      addLog(`ERROR: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>◈</span>
          AI MODERNIZER <span className={styles.version}>v1.0</span>
        </div>
        <div className={styles.status}>
          <span className={styles.pulse}></span> System: Optimal
        </div>
      </header>

      {/* Main Workspace */}
      <main className={styles.workspace}>
        {/* Editor Section */}
        <div className={styles.editorArea}>
          <div className={`${styles.pane} glass`}>
            <div className={styles.paneHeader}>
              <span>LEGACY SOURCE</span>
              <span className={styles.lang}>C++98 / C++03</span>
            </div>
            <textarea
              className={styles.editor}
              value={legacyCode}
              onChange={(e) => setLegacyCode(e.target.value)}
              spellCheck={false}
            />
          </div>

          <div className={styles.actionCenter}>
            <button 
              className={`${styles.modernizeBtn} ${isProcessing ? styles.loading : ''}`}
              onClick={handleModernize}
              disabled={isProcessing}
            >
              {isProcessing ? 'PROCESSING...' : 'MODERNIZE ❯'}
            </button>
          </div>

          <div className={`${styles.pane} glass`}>
            <div className={styles.paneHeader}>
              <span>MODERNIZED OUTPUT</span>
              <span className={styles.langAccent}>C++17 / C++20</span>
            </div>
            <textarea
              className={`${styles.editor} ${styles.output}`}
              value={modernizedCode}
              readOnly
              placeholder="Modernized code will appear here..."
              spellCheck={false}
            />
          </div>
        </div>

        {/* Terminal Section */}
        <footer className={`${styles.terminal} glass`}>
          <div className={styles.terminalHeader}>CORE DIAGNOSTICS</div>
          <div className={styles.logList}>
            {logs.map((log, i) => (
              <div key={i} className={styles.logLine}>{log}</div>
            ))}
          </div>
        </footer>
      </main>
    </div>
  );
}
