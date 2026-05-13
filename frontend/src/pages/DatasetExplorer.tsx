import { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';
import { fetchPaginatedSamples, fetchRandomSamples } from '../api';

interface Sample {
  text: string;
  image_base64: string;
  index: number;
}

interface PaginatedResult {
  split: string;
  page: number;
  page_size: number;
  total_samples: number;
  total_pages: number;
  samples: Sample[];
}

export default function DatasetExplorer() {
  const [data, setData] = useState<PaginatedResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [split, setSplit] = useState('train');
  const [page, setPage] = useState(0);
  const [mode, setMode] = useState<'browse' | 'random'>('browse');

  useEffect(() => {
    loadData();
  }, [split, page, mode]);

  async function loadData() {
    setLoading(true);
    setError(null);
    try {
      if (mode === 'browse') {
        const result = await fetchPaginatedSamples(split, page, 20);
        setData(result);
      } else {
        const samples = await fetchRandomSamples(split, 20);
        setData({
          split,
          page: 0,
          page_size: 20,
          total_samples: samples.length,
          total_pages: 1,
          samples,
        });
      }
    } catch (e: any) {
      setError(e.message || 'Failed to load dataset');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-header animate-in">
        <h1>Dataset Explorer</h1>
        <p>Browse and explore the IIIT-INDIC-HW-WORDS-Hindi handwriting dataset</p>
      </div>

      {/* Controls */}
      <div className="animate-in animate-in-delay-1" style={{ display: 'flex', gap: '16px', marginBottom: '24px', alignItems: 'center', flexWrap: 'wrap' }}>
        {/* Split Tabs */}
        <div className="tabs">
          {['train', 'validation', 'test'].map((s) => (
            <button
              key={s}
              className={`tab ${split === s ? 'active' : ''}`}
              onClick={() => { setSplit(s); setPage(0); }}
            >
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>

        {/* Mode Toggle */}
        <div className="tabs">
          <button
            className={`tab ${mode === 'browse' ? 'active' : ''}`}
            onClick={() => { setMode('browse'); setPage(0); }}
          >
            Browse
          </button>
          <button
            className={`tab ${mode === 'random' ? 'active' : ''}`}
            onClick={() => setMode('random')}
          >
            Random
          </button>
        </div>

        {mode === 'random' && (
          <button className="btn btn-secondary" onClick={loadData}>
            <RefreshCw size={14} /> Shuffle
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="card" style={{ marginBottom: '24px', borderColor: 'var(--error)' }}>
          <p style={{ color: 'var(--error)' }}>⚠️ {error}</p>
          <button className="btn btn-primary" onClick={loadData} style={{ marginTop: '12px' }}>Retry</button>
        </div>
      )}

      {/* Loading */}
      {loading && <div className="loading-spinner" />}

      {/* Grid */}
      {!loading && data && (
        <>
          <div className="dataset-grid animate-in">
            {data.samples.map((s, i) => (
              <div key={i} className="sample-card">
                <div className="sample-card-image">
                  <img
                    src={`data:image/jpeg;base64,${s.image_base64}`}
                    alt={s.text}
                    loading="lazy"
                  />
                </div>
                <div className="sample-card-body">
                  <div className="sample-card-label">{s.text}</div>
                  <div className="sample-card-index">
                    #{s.index} · {split}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {mode === 'browse' && data.total_pages > 1 && (
            <div className="pagination">
              <button
                className="btn btn-secondary"
                disabled={page === 0}
                onClick={() => setPage(page - 1)}
              >
                <ChevronLeft size={16} />
              </button>
              <span className="page-info">
                Page {page + 1} / {data.total_pages}
              </span>
              <button
                className="btn btn-secondary"
                disabled={page >= data.total_pages - 1}
                onClick={() => setPage(page + 1)}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
