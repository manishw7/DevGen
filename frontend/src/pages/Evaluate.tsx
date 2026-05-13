import { useState } from 'react';
import { Calculator, Copy, Check } from 'lucide-react';
import { evaluateCER } from '../api';

interface CERResult {
  cer: number;
  edit_distance: number;
  reference_length: number;
  prediction_length: number;
}

export default function Evaluate() {
  const [prediction, setPrediction] = useState('');
  const [reference, setReference] = useState('');
  const [result, setResult] = useState<CERResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  async function handleEvaluate() {
    if (!prediction.trim() || !reference.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await evaluateCER(prediction, reference);
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Evaluation failed');
    } finally {
      setLoading(false);
    }
  }

  function getCERBadge(cer: number) {
    if (cer <= 0.05) return { cls: 'success', label: 'Excellent' };
    if (cer <= 0.15) return { cls: 'warning', label: 'Acceptable' };
    return { cls: 'error', label: 'Needs Improvement' };
  }

  function copyResult() {
    if (!result) return;
    const text = `CER: ${(result.cer * 100).toFixed(2)}%\nEdit Distance: ${result.edit_distance}\nReference Length: ${result.reference_length}`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div>
      <div className="page-header animate-in">
        <h1>CER Evaluation</h1>
        <p>Calculate Character Error Rate between OCR predictions and ground truth using Levenshtein Distance</p>
      </div>

      <div className="result-container animate-in animate-in-delay-1">
        {/* Prediction Input */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Prediction (OCR Output)</div>
          </div>
          <textarea
            value={prediction}
            onChange={(e) => setPrediction(e.target.value)}
            placeholder="Paste the OCR-predicted Devanagari text here..."
            style={{
              width: '100%',
              minHeight: '180px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              padding: '16px',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-mono)',
              fontSize: '16px',
              lineHeight: '1.6',
              resize: 'vertical',
              outline: 'none',
            }}
          />
        </div>

        {/* Reference Input */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Reference (Ground Truth)</div>
          </div>
          <textarea
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            placeholder="Paste the correct ground truth Devanagari text here..."
            style={{
              width: '100%',
              minHeight: '180px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              padding: '16px',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-mono)',
              fontSize: '16px',
              lineHeight: '1.6',
              resize: 'vertical',
              outline: 'none',
            }}
          />
        </div>
      </div>

      {/* Action */}
      <div className="animate-in animate-in-delay-2" style={{ marginTop: '24px', display: 'flex', justifyContent: 'center' }}>
        <button
          className="btn btn-primary"
          onClick={handleEvaluate}
          disabled={loading || !prediction.trim() || !reference.trim()}
          style={{ padding: '14px 40px', fontSize: '15px' }}
        >
          <Calculator size={18} />
          {loading ? 'Calculating...' : 'Calculate CER'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="card animate-in" style={{ marginTop: '24px', borderColor: 'var(--error)' }}>
          <p style={{ color: 'var(--error)' }}>⚠️ {error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="card animate-in" style={{ marginTop: '24px' }}>
          <div className="card-header">
            <div className="card-title">
              <Calculator size={20} />
              Evaluation Result
            </div>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span className={`badge ${getCERBadge(result.cer).cls}`}>
                {getCERBadge(result.cer).label}
              </span>
              <button className="btn btn-ghost" onClick={copyResult}>
                {copied ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
          </div>

          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon purple"><Calculator size={20} /></div>
              <div className="stat-value" style={{ color: 'var(--text-accent)' }}>
                {(result.cer * 100).toFixed(2)}%
              </div>
              <div className="stat-label">Character Error Rate</div>
              {/* Visual bar */}
              <div className="confidence-bar" style={{ marginTop: '12px', height: '8px', borderRadius: '4px' }}>
                <div
                  className={`confidence-bar-fill ${result.cer <= 0.05 ? 'high' : result.cer <= 0.15 ? 'medium' : 'low'}`}
                  style={{ width: `${Math.min(result.cer * 100, 100)}%` }}
                />
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.edit_distance}</div>
              <div className="stat-label">Edit Distance (Levenshtein)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.reference_length}</div>
              <div className="stat-label">Reference Characters</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.prediction_length}</div>
              <div className="stat-label">Predicted Characters</div>
            </div>
          </div>

          {/* Formula */}
          <div style={{
            marginTop: '16px',
            padding: '16px',
            background: 'var(--bg-secondary)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)',
            fontFamily: 'var(--font-mono)',
            fontSize: '14px',
            color: 'var(--text-secondary)',
            textAlign: 'center',
          }}>
            CER = (S + D + I) / N = {result.edit_distance} / {result.reference_length} = {(result.cer * 100).toFixed(2)}%
          </div>
        </div>
      )}
    </div>
  );
}
