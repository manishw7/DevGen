import { useState } from 'react';
import { BookOpen, Search, AlertCircle } from 'lucide-react';
import { extractNER } from '../api';

interface Entities {
  dates: string[];
  citizenship_numbers: string[];
  killa_numbers: string[];
  paana_numbers: string[];
  districts: string[];
  provinces: string[];
  wards: string[];
  raw_numbers: string[];
  _extraction_confidence: number;
  _field_count: number;
}

interface NERResult {
  entities: Entities;
  summary: string;
}

const ENTITY_CONFIG: { key: keyof Entities; label: string; emoji: string; color: string }[] = [
  { key: 'dates',               label: 'Dates',            emoji: '📅', color: '#60a5fa' },
  { key: 'citizenship_numbers', label: 'Citizenship Nos.', emoji: '🪪', color: '#a78bfa' },
  { key: 'killa_numbers',       label: 'Killa (Plot) Nos.',emoji: '📋', color: '#34d399' },
  { key: 'paana_numbers',       label: 'Paana (Sheet) Nos.',emoji:'📄', color: '#fbbf24' },
  { key: 'districts',           label: 'Districts',        emoji: '📍', color: '#f87171' },
  { key: 'provinces',           label: 'Provinces',        emoji: '🗺️', color: '#fb923c' },
  { key: 'wards',               label: 'Wards',            emoji: '🏘️', color: '#a3e635' },
  { key: 'raw_numbers',         label: 'Other IDs',        emoji: '🔢', color: '#94a3b8' },
];

const SAMPLE_TEXTS = [
  'नागरिकता नं. 12-34-56-78901 जारी मिति २०७८/०३/१५ काठमाडौं जिल्ला वडा नं. ७',
  'पाना नं. 45 कित्ता 123 प्रदेश नं. ३ ललितपुर जिल्ला',
  'citizenship no 987654321 issued date 2078-06-20 Bhaktapur district ward 12',
];

export default function NERPage() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<NERResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleExtract() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await extractNER(text);
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'NER extraction failed');
    } finally {
      setLoading(false);
    }
  }

  function loadSample(s: string) {
    setText(s);
    setResult(null);
    setError(null);
  }

  const confPct = result ? Math.round(result.entities._extraction_confidence * 100) : 0;

  return (
    <div>
      <div className="page-header animate-in">
        <h1>NER Extraction</h1>
        <p>Extract structured fields — dates, ID numbers, districts — from recognized Devanagari text</p>
      </div>

      {/* Sample Prompts */}
      <div className="animate-in animate-in-delay-1" style={{ marginBottom: '16px' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
          Try a sample document text:
        </p>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {SAMPLE_TEXTS.map((s, i) => (
            <button
              key={i}
              className="btn btn-secondary"
              style={{ fontSize: '11px', padding: '6px 12px' }}
              onClick={() => loadSample(s)}
            >
              Sample {i + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Text Input */}
      <div className="card animate-in animate-in-delay-1" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <div className="card-title">
            <BookOpen size={20} />
            Recognized Text Input
          </div>
        </div>
        <textarea
          value={text}
          onChange={(e) => { setText(e.target.value); setResult(null); }}
          placeholder="Paste or type recognized Devanagari text from a Lal-Purja or Nagarikta document..."
          style={{
            width: '100%',
            minHeight: '140px',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            padding: '16px',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-mono)',
            fontSize: '16px',
            lineHeight: '1.8',
            resize: 'vertical',
            outline: 'none',
            marginBottom: '16px',
          }}
        />
        <button
          className="btn btn-primary"
          onClick={handleExtract}
          disabled={loading || !text.trim()}
        >
          <Search size={16} />
          {loading ? 'Extracting...' : 'Extract Entities'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="card animate-in" style={{ marginBottom: '24px', borderColor: 'var(--error)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--error)' }}>
            <AlertCircle size={18} />
            {error}
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="animate-in">
          {/* Confidence Bar */}
          <div className="card" style={{ marginBottom: '24px' }}>
            <div className="card-header">
              <div className="card-title">Extraction Confidence</div>
              <span className="badge info">{result.entities._field_count} fields found</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{ flex: 1 }}>
                <div className="confidence-bar" style={{ height: '10px', borderRadius: '5px' }}>
                  <div
                    className={`confidence-bar-fill ${confPct >= 60 ? 'high' : confPct >= 30 ? 'medium' : 'low'}`}
                    style={{ width: `${confPct}%` }}
                  />
                </div>
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 700, color: 'var(--text-accent)', minWidth: '50px', textAlign: 'right' }}>
                {confPct}%
              </span>
            </div>
          </div>

          {/* Entity Cards */}
          <div className="stats-grid">
            {ENTITY_CONFIG.map(({ key, label, emoji, color }) => {
              const vals = result.entities[key];
              if (!Array.isArray(vals) || vals.length === 0) return null;
              return (
                <div className="stat-card" key={key} style={{ '--accent-start': color } as any}>
                  <div className="stat-icon" style={{ background: `${color}22`, color }}>
                    <span style={{ fontSize: '20px' }}>{emoji}</span>
                  </div>
                  <div style={{ marginTop: '8px' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>
                      {label}
                    </div>
                    {vals.map((v, i) => (
                      <div
                        key={i}
                        style={{
                          display: 'inline-block',
                          background: `${color}18`,
                          border: `1px solid ${color}44`,
                          borderRadius: 'var(--radius-sm)',
                          padding: '4px 10px',
                          margin: '2px 4px 2px 0',
                          fontFamily: 'var(--font-mono)',
                          fontSize: '13px',
                          color,
                        }}
                      >
                        {v}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* No entities */}
          {result.entities._field_count === 0 && (
            <div className="empty-state card">
              <BookOpen size={40} />
              <h3>No Structured Entities Found</h3>
              <p>The text doesn't appear to contain recognizable Nepali document fields like dates, ID numbers, or district names.</p>
            </div>
          )}

          {/* Raw Summary */}
          <div className="card" style={{ marginTop: '24px' }}>
            <div className="card-title" style={{ marginBottom: '12px' }}>
              <BookOpen size={18} />
              Plain Text Summary
            </div>
            <pre style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              color: 'var(--text-secondary)',
              whiteSpace: 'pre-wrap',
              lineHeight: '1.8',
            }}>
              {result.summary || 'No entities found.'}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
