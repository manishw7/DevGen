import { useState, useEffect } from 'react';
import { Sparkles, Download, RefreshCw, Zap, Info, PenTool, Type, Sliders } from 'lucide-react';
import { generateHandwriting, fetchGenerateInfo, generateLDMHandwriting, fetchLDMInfo } from '../api';

interface GeneratedImage {
  image_base64: string;
  index: number;
}

export default function GenerativeSuite() {
  const [activeTab, setActiveTab] = useState<'gan' | 'ldm'>('ldm');
  
  // GAN State
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [loadingGan, setLoadingGan] = useState(false);
  const [count, setCount] = useState(8);
  const [ganInfo, setGanInfo] = useState<any>(null);
  const [errorGan, setErrorGan] = useState('');

  // LDM State
  const [ldmText, setLdmText] = useState('नमस्ते');
  const [ldmFont, setLdmFont] = useState('font.ttf');
  const [conditioningScale, setConditioningScale] = useState(1.5);
  const [ldmInfo, setLdmInfo] = useState<any>(null);
  const [loadingLdm, setLoadingLdm] = useState(false);
  const [errorLdm, setErrorLdm] = useState('');
  const [ldmOutput, setLdmOutput] = useState<string | null>(null);
  const [ldmControl, setLdmControl] = useState<string | null>(null);

  useEffect(() => {
    fetchGenerateInfo().then(setGanInfo).catch(() => {});
    fetchLDMInfo().then(setLdmInfo).catch(() => {});
  }, []);

  const handleGenerateGan = async () => {
    setLoadingGan(true);
    setErrorGan('');
    try {
      const result = await generateHandwriting(count);
      setImages(result.images);
    } catch (e: any) {
      setErrorGan(e.message || 'GAN Generation failed');
    } finally {
      setLoadingGan(false);
    }
  };

  const handleGenerateLdm = async () => {
    if (!ldmText) return;
    setLoadingLdm(true);
    setErrorLdm('');
    try {
      const result = await generateLDMHandwriting(ldmText, ldmFont, conditioningScale);
      setLdmOutput(result.generated_image_base64);
      setLdmControl(result.control_image_base64);
    } catch (e: any) {
      setErrorLdm(e.message || 'LDM Generation failed. Is the checkpoint available?');
    } finally {
      setLoadingLdm(false);
    }
  };

  const downloadImage = (b64: string, name: string) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${b64}`;
    link.download = name;
    link.click();
  };

  const downloadAllGan = () => {
    images.forEach((img, i) => {
      setTimeout(() => downloadImage(img.image_base64, `devgen_gan_${i}.png`), i * 100);
    });
  };

  return (
    <div style={{ padding: '32px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <Sparkles size={28} style={{ color: '#a78bfa' }} />
          <h1 style={{ fontSize: '28px', fontWeight: 700, margin: 0, background: 'linear-gradient(135deg, #a78bfa, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Generative Suite
          </h1>
        </div>
        <p style={{ color: '#94a3b8', fontSize: '15px', margin: 0 }}>
          Synthesize realistic Devanagari handwriting using Latent Diffusion or GANs.
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '32px', borderBottom: '1px solid rgba(148, 163, 184, 0.1)', paddingBottom: '16px' }}>
        <button
          onClick={() => setActiveTab('ldm')}
          style={{
            background: activeTab === 'ldm' ? 'rgba(139, 92, 246, 0.15)' : 'transparent',
            border: activeTab === 'ldm' ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid transparent',
            color: activeTab === 'ldm' ? '#c4b5fd' : '#94a3b8',
            padding: '10px 20px', borderRadius: '8px', cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 600,
            transition: 'all 0.2s'
          }}
        >
          <PenTool size={18} /> LDM ControlNet (Text-to-Handwriting)
        </button>
        <button
          onClick={() => setActiveTab('gan')}
          style={{
            background: activeTab === 'gan' ? 'rgba(139, 92, 246, 0.15)' : 'transparent',
            border: activeTab === 'gan' ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid transparent',
            color: activeTab === 'gan' ? '#c4b5fd' : '#94a3b8',
            padding: '10px 20px', borderRadius: '8px', cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 600,
            transition: 'all 0.2s'
          }}
        >
          <Zap size={18} /> GAN (Random Handwriting)
        </button>
      </div>

      {/* LDM ControlNet Tab */}
      {activeTab === 'ldm' && (
        <div className="tab-pane fade-in">
          <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.1)', borderRadius: '12px', padding: '24px', marginBottom: '24px' }}>
            <h3 style={{ margin: '0 0 20px 0', fontSize: '16px', color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={18} style={{ color: '#a78bfa' }} /> Generation Parameters
            </h3>
            
            <div>
              <label style={{ display: 'block', color: '#94a3b8', fontSize: '13px', marginBottom: '8px', fontWeight: 500 }}>
                <Type size={14} style={{ display: 'inline', verticalAlign: '-2px', marginRight: '4px' }} /> Devanagari Text
              </label>
              <input
                type="text"
                value={ldmText}
                onChange={(e) => setLdmText(e.target.value)}
                placeholder="Enter Devanagari text to generate handwriting..."
                style={{ width: '100%', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.2)', borderRadius: '8px', color: '#fff', padding: '12px', fontSize: '18px', fontFamily: 'sans-serif' }}
              />
              <p style={{ margin: '8px 0 0', color: '#64748b', fontSize: '12px' }}>
                Using LDM Checkpoint: <strong style={{ color: '#cbd5e1' }}>{ldmInfo?.checkpoint_name || 'resolving...'}</strong>
              </p>
            </div>

            <div style={{ marginTop: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'between', alignItems: 'center', marginBottom: '8px' }}>
                <label style={{ color: '#94a3b8', fontSize: '13px', fontWeight: 500 }}>
                  ControlNet Conditioning Scale: <strong style={{ color: '#a78bfa' }}>{conditioningScale.toFixed(1)}</strong>
                </label>
                <span style={{ fontSize: '12px', color: '#64748b', marginLeft: 'auto' }}>
                  (Recommended: 1.5 - 1.8 for under-trained checkpoints)
                </span>
              </div>
              <input
                type="range"
                min="0.0"
                max="2.5"
                step="0.1"
                value={conditioningScale}
                onChange={(e) => setConditioningScale(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#8b5cf6', cursor: 'pointer' }}
              />
            </div>

            <div style={{ marginTop: '24px', display: 'flex', alignItems: 'center', gap: '16px' }}>
              <button
                onClick={handleGenerateLdm}
                disabled={loadingLdm || !ldmText}
                style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  background: loadingLdm || !ldmText ? 'rgba(99, 102, 241, 0.3)' : 'linear-gradient(135deg, #7c3aed, #6366f1)',
                  border: 'none', borderRadius: '10px', color: '#fff', padding: '12px 28px',
                  fontSize: '15px', fontWeight: 600, cursor: loadingLdm || !ldmText ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s', boxShadow: '0 4px 12px rgba(99, 102, 241, 0.2)'
                }}
              >
                {loadingLdm ? <RefreshCw size={18} className="spinning" /> : <PenTool size={18} />}
                {loadingLdm ? 'Running LDM Pipeline...' : 'Generate Realistic Handwriting'}
              </button>
            </div>
          </div>

          {errorLdm && (
            <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '10px', padding: '16px', color: '#fca5a5', marginBottom: '24px', fontSize: '14px' }}>
              <strong>Error:</strong> {errorLdm}
            </div>
          )}

          {ldmOutput && !loadingLdm && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
              <div style={{ background: 'rgba(30, 41, 59, 0.3)', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.1)', overflow: 'hidden' }}>
                <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(148, 163, 184, 0.1)', color: '#cbd5e1', fontSize: '14px', fontWeight: 500, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>Final Render ({ldmInfo?.resolution || '256x256'})</span>
                  <button onClick={() => downloadImage(ldmOutput, `ldm_${ldmText}.png`)} style={{ background: 'none', border: 'none', color: '#818cf8', cursor: 'pointer' }}><Download size={16} /></button>
                </div>
                <div style={{ padding: '24px', display: 'flex', justifyContent: 'center' }}>
                  <img src={`data:image/png;base64,${ldmOutput}`} alt="LDM Output" style={{ maxWidth: '100%', borderRadius: '4px', boxShadow: '0 4px 20px rgba(0,0,0,0.3)' }} />
                </div>
              </div>
              
              <div style={{ background: 'rgba(30, 41, 59, 0.3)', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.1)', overflow: 'hidden' }}>
                <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(148, 163, 184, 0.1)', color: '#cbd5e1', fontSize: '14px', fontWeight: 500, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>Control Map (Font Renderer)</span>
                  <button onClick={() => downloadImage(ldmControl!, `ctrl_${ldmText}.png`)} style={{ background: 'none', border: 'none', color: '#818cf8', cursor: 'pointer' }}><Download size={16} /></button>
                </div>
                <div style={{ padding: '24px', display: 'flex', justifyContent: 'center' }}>
                  <img src={`data:image/png;base64,${ldmControl}`} alt="LDM Control" style={{ maxWidth: '100%', borderRadius: '4px' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* GAN Tab */}
      {activeTab === 'gan' && (
        <div className="tab-pane fade-in">
          {ganInfo && (
            <div style={{ background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.1)', borderRadius: '12px', padding: '20px', marginBottom: '24px', display: 'flex', gap: '32px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Info size={16} style={{ color: '#64748b' }} />
                <span style={{ color: '#94a3b8', fontSize: '13px' }}>Architecture:</span>
                <span style={{ color: '#e2e8f0', fontSize: '13px', fontWeight: 600 }}>{ganInfo.architecture}</span>
              </div>
              <div>
                <span style={{ color: '#94a3b8', fontSize: '13px' }}>Resolution: </span>
                <span style={{ color: '#e2e8f0', fontSize: '13px', fontWeight: 600 }}>{ganInfo.resolution}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="dot" style={{ width: '8px', height: '8px', borderRadius: '50%', background: ganInfo.available ? '#22c55e' : '#ef4444', display: 'inline-block' }} />
                <span style={{ color: ganInfo.available ? '#22c55e' : '#ef4444', fontSize: '13px', fontWeight: 600 }}>
                  {ganInfo.available ? 'Model Ready' : 'Model Not Found'}
                </span>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', gap: '16px', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ color: '#94a3b8', fontSize: '14px' }}>Count:</label>
              <select
                value={count}
                onChange={(e) => setCount(Number(e.target.value))}
                style={{ background: 'rgba(30, 41, 59, 0.8)', border: '1px solid rgba(148, 163, 184, 0.2)', borderRadius: '8px', color: '#e2e8f0', padding: '8px 12px', fontSize: '14px' }}
              >
                {[1, 4, 8, 12, 16].map(n => (
                  <option key={n} value={n}>{n} image{n > 1 ? 's' : ''}</option>
                ))}
              </select>
            </div>

            <button
              onClick={handleGenerateGan}
              disabled={loadingGan}
              style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                background: loadingGan ? 'rgba(99, 102, 241, 0.3)' : 'linear-gradient(135deg, #7c3aed, #6366f1)',
                border: 'none', borderRadius: '10px', color: '#fff', padding: '10px 24px',
                fontSize: '14px', fontWeight: 600, cursor: loadingGan ? 'wait' : 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {loadingGan ? <RefreshCw size={16} className="spinning" /> : <Zap size={16} />}
              {loadingGan ? 'Generating...' : 'Generate Random Words'}
            </button>

            {images.length > 0 && (
              <button
                onClick={downloadAllGan}
                style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  background: 'rgba(30, 41, 59, 0.8)', border: '1px solid rgba(148, 163, 184, 0.2)',
                  borderRadius: '10px', color: '#e2e8f0', padding: '10px 20px',
                  fontSize: '14px', cursor: 'pointer',
                }}
              >
                <Download size={16} /> Download All
              </button>
            )}
          </div>

          {errorGan && (
            <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '10px', padding: '12px 16px', color: '#fca5a5', marginBottom: '24px', fontSize: '14px' }}>
              {errorGan}
            </div>
          )}

          {images.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))', gap: '16px' }}>
              {images.map((img, i) => (
                <div
                  key={i}
                  onClick={() => downloadImage(img.image_base64, `devgen_gan_${i}.png`)}
                  style={{
                    background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(148, 163, 184, 0.1)',
                    borderRadius: '12px', overflow: 'hidden', cursor: 'pointer', transition: 'all 0.2s',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.5)'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.1)'; e.currentTarget.style.transform = 'translateY(0)'; }}
                >
                  <div style={{ padding: '12px', display: 'flex', justifyContent: 'center', background: '#fff' }}>
                    <img src={`data:image/png;base64,${img.image_base64}`} alt={`Generated ${i}`} style={{ width: '100%', height: 'auto', imageRendering: 'pixelated' }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {images.length === 0 && !loadingGan && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 20px', background: 'rgba(30, 41, 59, 0.3)', borderRadius: '16px', border: '1px dashed rgba(148, 163, 184, 0.2)' }}>
              <Zap size={48} style={{ color: '#475569', marginBottom: '16px' }} />
              <p style={{ color: '#64748b', fontSize: '16px', margin: 0 }}>Click <strong>Generate Random Words</strong> to test the GAN.</p>
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .spinning { animation: spin 1s linear infinite; }
        .fade-in { animation: fadeIn 0.3s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
