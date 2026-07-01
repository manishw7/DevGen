import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Check,
  Copy,
  Cpu,
  FileCheck,
  Scan,
  Sparkles,
  Timer,
  Upload,
  Waypoints,
} from "lucide-react";

import { fetchModelInfo, getApiBase, recognizeDocument, recognizeFull } from "../api";

interface ModelInfo {
  base_model_name: string;
  adapter_path: string | null;
  adapter_root: string;
  adapter_source: string;
  available_checkpoints: string[];
  using_adapter: boolean;
  device: string;
  loaded: boolean;
  default_max_length: number;
  num_beams: number;
}

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

interface RecognitionResult {
  status: string;
  text: string;
  confidence_scores: number[];
  tokens: string[];
  preprocessed_image_base64: string;
  filename: string;
  average_confidence: number | null;
  inference_ms: number;
  generation_steps: number;
  entities?: Entities;
  entities_summary?: string;
  model_info: {
    base_model_name: string;
    adapter_path: string | null;
    using_adapter: boolean;
    device: string;
    model_used: string;
    force_model_received: string | null;
  };
}

const ENTITY_LABELS: Record<string, string> = {
  dates: "Dates",
  citizenship_numbers: "Citizenship",
  killa_numbers: "Killa",
  paana_numbers: "Paana",
  districts: "Districts",
  provinces: "Provinces",
  wards: "Wards",
  raw_numbers: "Other IDs",
};

export default function Recognize() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragover, setDragover] = useState(false);
  const [includeExtraction, setIncludeExtraction] = useState(true);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [routingMode, setRoutingMode] = useState<string>("Automatic");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadModelInfo();
  }, []);

  const handleFile = useCallback((selectedFile: File) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(selectedFile);
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragover(false);
      const droppedFile = event.dataTransfer.files?.[0];
      if (droppedFile) {
        handleFile(droppedFile);
      }
    },
    [handleFile]
  );

  async function loadModelInfo() {
    try {
      setModelError(null);
      const info = await fetchModelInfo();
      setModelInfo(info);
    } catch (err: any) {
      setModelError(err.message || "Could not load model info");
    }
  }

  async function runRecognition() {
    if (!file) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const forceModel = routingMode === "Automatic" ? undefined : (routingMode === "Word" ? "trocr" : "cnn");
      const response = includeExtraction
        ? await recognizeFull(file, forceModel)
        : await recognizeDocument(file, forceModel);
      setResult(response);
      await loadModelInfo();
    } catch (err: any) {
      setError(err.message || "Recognition failed");
    } finally {
      setLoading(false);
    }
  }

  function clearSelection() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  function getConfidenceClass(score: number) {
    if (score >= 0.85) return "high-conf";
    if (score >= 0.6) return "medium-conf";
    return "low-conf";
  }

  function getConfidenceColor(score: number) {
    if (score >= 0.85) return "var(--conf-high)";
    if (score >= 0.6) return "var(--conf-medium)";
    return "var(--conf-low)";
  }

  function getBarClass(score: number) {
    if (score >= 0.85) return "high";
    if (score >= 0.6) return "medium";
    return "low";
  }

  async function copyRecognizedText() {
    if (!result?.text) {
      return;
    }

    await navigator.clipboard.writeText(result.text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  const adapterName = modelInfo?.adapter_path?.split(/[/\\]/).pop();
  const extractedGroups = result?.entities
    ? Object.entries(result.entities).filter(
        ([key, value]) => !key.startsWith("_") && Array.isArray(value) && value.length > 0
      )
    : [];

  return (
    <div>
      <div className="page-header animate-in">
        <h1>Document Recognition</h1>
        <p>
          Upload a handwritten Devanagari document and run your TrOCR LoRA model
          through a clean OCR + extraction workflow.
        </p>
      </div>

      <div className="stats-grid animate-in animate-in-delay-1">
        <div className="stat-card">
          <div className="stat-icon purple">
            <Cpu size={20} />
          </div>
          <div className="stat-value" style={{ fontSize: "18px" }}>
            {modelInfo?.loaded ? "Loaded" : "Lazy"}
          </div>
          <div className="stat-label">
            {modelInfo?.using_adapter ? adapterName || "LoRA Adapter" : "Base Model Only"}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon blue">
            <Waypoints size={20} />
          </div>
          <div className="stat-value" style={{ fontSize: "18px" }}>
            {modelInfo?.available_checkpoints?.length || 0}
          </div>
          <div className="stat-label">Checkpoints Discovered</div>
        </div>

        <div className="stat-card">
          <div className="stat-icon green">
            <Sparkles size={20} />
          </div>
          <div className="stat-value" style={{ fontSize: "18px" }}>
            {modelInfo?.device || "CPU"}
          </div>
          <div className="stat-label">Inference Device</div>
        </div>

        <div className="stat-card">
          <div className="stat-icon amber">
            <Timer size={20} />
          </div>
          <div className="stat-value" style={{ fontSize: "18px" }}>
            {modelInfo?.num_beams || 4}
          </div>
          <div className="stat-label">Beam Search Width</div>
        </div>
      </div>

      <div className="card animate-in animate-in-delay-2" style={{ marginBottom: "24px" }}>
        <div className="card-header card-header-wrap">
          <div className="card-title">
            <Cpu size={20} />
            Model Integration
          </div>
          <button className="btn btn-secondary" onClick={loadModelInfo}>
            Refresh Status
          </button>
        </div>

        {modelError ? (
          <div className="inline-alert error">
            <AlertCircle size={18} />
            <span>{modelError}</span>
          </div>
        ) : (
          <div className="info-grid">
            <div className="info-chip">
              <span className="info-chip-label">API</span>
              <span className="info-chip-value">{getApiBase()}</span>
            </div>
            <div className="info-chip">
              <span className="info-chip-label">Base model</span>
              <span className="info-chip-value">{modelInfo?.base_model_name || "Loading..."}</span>
            </div>
            <div className="info-chip">
              <span className="info-chip-label">Adapter source</span>
              <span className="info-chip-value">{modelInfo?.adapter_source || "Detecting..."}</span>
            </div>
            <div className="info-chip">
              <span className="info-chip-label">Resolved adapter</span>
              <span className="info-chip-value">{modelInfo?.adapter_path || "No adapter detected"}</span>
            </div>
          </div>
        )}
      </div>

      <div
        className={`upload-zone animate-in animate-in-delay-2 ${dragover ? "dragover" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setDragover(true);
        }}
        onDragLeave={() => setDragover(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => {
            const selectedFile = event.target.files?.[0];
            if (selectedFile) {
              handleFile(selectedFile);
            }
          }}
        />
        <div className="upload-zone-icon">
          <Upload />
        </div>
        <h3>Drop your document here or click to browse</h3>
        <p>
          Supports JPEG, PNG, TIFF. The app will preprocess the image, run TrOCR,
          and optionally extract structured document fields.
        </p>
      </div>

      {preview && (
        <div className="card animate-in" style={{ marginTop: "24px" }}>
          <div className="card-header card-header-wrap">
            <div className="card-title">
              <Scan size={20} />
              {file?.name}
            </div>
              <div className="custom-select-wrapper">
                <select 
                  className="btn btn-secondary" 
                  value={routingMode} 
                  onChange={(e) => setRoutingMode(e.target.value)}
                  disabled={loading}
                  style={{ minWidth: "140px", cursor: "pointer" }}
                >
                  <option value="Automatic">Automatic Mode</option>
                  <option value="Word">Word (TrOCR)</option>
                  <option value="Character">Character (CNN)</option>
                </select>
              </div>
              <button className="btn btn-primary" onClick={runRecognition} disabled={loading}>
                {loading ? "Running OCR..." : includeExtraction ? "Run OCR + NER" : "Run OCR"}
              </button>
            </div>

          <div className="tabs" style={{ marginBottom: "16px" }}>
            <button
              className={`tab ${!includeExtraction ? "active" : ""}`}
              onClick={() => setIncludeExtraction(false)}
            >
              OCR Only
            </button>
            <button
              className={`tab ${includeExtraction ? "active" : ""}`}
              onClick={() => setIncludeExtraction(true)}
            >
              OCR + NER
            </button>
          </div>

          <div className="info-grid" style={{ marginBottom: "18px" }}>
            <div className="info-chip">
              <span className="info-chip-label">File type</span>
              <span className="info-chip-value">{file?.type || "Unknown"}</span>
            </div>
            <div className="info-chip">
              <span className="info-chip-label">File size</span>
              <span className="info-chip-value">
                {file ? `${(file.size / 1024).toFixed(1)} KB` : "—"}
              </span>
            </div>
            <div className="info-chip">
              <span className="info-chip-label">Pipeline</span>
              <span className="info-chip-value">
                {includeExtraction ? "Recognition + extraction" : "Recognition only"}
              </span>
            </div>
          </div>

          <div className="image-preview">
            <img src={preview} alt="Uploaded document" />
          </div>
        </div>
      )}

      {error && (
        <div className="card animate-in" style={{ marginTop: "24px", borderColor: "var(--error)" }}>
          <div className="inline-alert error">
            <AlertCircle size={20} />
            <div>
              <strong>Recognition failed</strong>
              <p style={{ fontSize: "13px", marginTop: "4px", opacity: 0.85 }}>{error}</p>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="animate-in" style={{ marginTop: "24px" }}>
          <div className="stats-grid" style={{ marginBottom: "24px" }}>
            <div className="stat-card">
              <div className="stat-icon purple">
                <FileCheck size={20} />
              </div>
              <div className="stat-value" style={{ fontSize: "18px" }}>
                {result.text?.length || 0}
              </div>
              <div className="stat-label">Recognized Characters</div>
            </div>

            <div className="stat-card">
              <div className="stat-icon green">
                <Sparkles size={20} />
              </div>
              <div className="stat-value" style={{ fontSize: "18px" }}>
                {result.average_confidence !== null
                  ? `${(result.average_confidence * 100).toFixed(1)}%`
                  : "—"}
              </div>
              <div className="stat-label">Average Token Confidence</div>
            </div>

            <div className="stat-card">
              <div className="stat-icon blue">
                <Timer size={20} />
              </div>
              <div className="stat-value" style={{ fontSize: "18px" }}>
                {result.inference_ms ? `${result.inference_ms.toFixed(0)} ms` : "—"}
              </div>
              <div className="stat-label">
                Inference ({result.model_info.model_used === "cnn_classifier" ? "CNN" : "TrOCR"})
                {result.model_info.force_model_received && (
                  <span style={{ opacity: 0.5, fontSize: "10px", marginLeft: "4px" }}>
                    [Forced: {result.model_info.force_model_received}]
                  </span>
                )}
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon amber">
                <Waypoints size={20} />
              </div>
              <div className="stat-value" style={{ fontSize: "18px" }}>
                {result.entities?._field_count ?? result.tokens.length}
              </div>
              <div className="stat-label">
                {result.entities ? "Structured Fields Found" : "Token Steps"}
              </div>
            </div>
          </div>

          <div className="result-container" style={{ gridTemplateColumns: "1fr 1fr 1fr", gap: "24px" }}>
            <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "24px" }}>
              <div className="card-title" style={{ marginBottom: "20px" }}>Overall Confidence</div>
              {result.average_confidence !== null && (
                <div style={{ position: "relative", width: "120px", height: "120px" }}>
                  <svg width="120" height="120" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
                    <circle 
                      cx="50" cy="50" r="45" fill="none" 
                      stroke={getConfidenceColor(result.average_confidence)} 
                      strokeWidth="8" 
                      strokeDasharray="282.7" 
                      strokeDashoffset={282.7 * (1 - result.average_confidence)} 
                      strokeLinecap="round" 
                      style={{ transition: "stroke-dashoffset 0.8s ease-out" }}
                    />
                    <text x="50" y="58" fontFamily="Arial" fontSize="22" fontWeight="bold" fill={getConfidenceColor(result.average_confidence)} textAnchor="middle">
                      {Math.round(result.average_confidence * 100)}%
                    </text>
                  </svg>
                </div>
              )}
              <div 
                className={`badge ${result.average_confidence! > 0.9 ? "success" : result.average_confidence! > 0.7 ? "info" : "error"}`}
                style={{ marginTop: "16px", padding: "6px 12px", fontSize: "14px" }}
              >
                {result.average_confidence! > 0.9 ? "Highly Confident" : result.average_confidence! > 0.7 ? "Likely Correct" : "Manual Review Needed"}
              </div>
            </div>

            <div className="card">
              <div className="card-header card-header-wrap">
                <div className="card-title">
                  <FileCheck size={20} />
                  Recognized Text
                </div>
                <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
                  <span className="badge success">Complete</span>
                  <button className="btn btn-ghost" onClick={copyRecognizedText}>
                    {copied ? <Check size={14} /> : <Copy size={14} />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                </div>
              </div>
              <div className="result-text-area">{result.text || "(no text recognized)"}</div>
            </div>

            <div className="card">
              <div className="card-header card-header-wrap">
                <div className="card-title">
                  <Scan size={20} />
                  Model Output
                </div>
                <span className="badge info">
                  {result.model_info.using_adapter ? "LoRA active" : "Base model"}
                </span>
              </div>
              <div className="image-compare-grid">
                <div>
                  <div className="mini-label">Original upload</div>
                  <div className="image-preview compact">
                    <img src={preview || undefined} alt="Original document" />
                  </div>
                </div>
                <div>
                  <div className="mini-label">Preprocessed for OCR</div>
                  <div className="image-preview compact">
                    <img
                      src={`data:image/png;base64,${result.preprocessed_image_base64}`}
                      alt="Preprocessed document"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {result.confidence_scores.length > 0 && (
            <div className="card" style={{ marginTop: "24px" }}>
              <div className="card-header card-header-wrap">
                <div className="card-title">Confidence Heatmap</div>
                <div style={{ display: "flex", gap: "12px", fontSize: "12px", flexWrap: "wrap" }}>
                  <span style={{ color: "var(--conf-high)" }}>High (greater than 85%)</span>
                  <span style={{ color: "var(--conf-medium)" }}>Medium (60 to 85%)</span>
                  <span style={{ color: "var(--conf-low)" }}>Low (less than 60%)</span>
                </div>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {result.tokens.map((token, index) => {
                  const score = result.confidence_scores[index] ?? 0;
                  return (
                    <div key={`${token}-${index}`} className={`token-chip ${getConfidenceClass(score)}`}>
                      <span className="token-char">{token}</span>
                      <div className="confidence-bar" style={{ width: "70px" }}>
                        <div
                          className={`confidence-bar-fill ${getBarClass(score)}`}
                          style={{ width: `${Math.max(score * 100, 2)}%` }}
                        />
                      </div>
                      <span className="token-score" style={{ color: getConfidenceColor(score) }}>
                        {(score * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {result.entities && (
            <div className="card" style={{ marginTop: "24px" }}>
              <div className="card-header card-header-wrap">
                <div className="card-title">
                  <Sparkles size={20} />
                  Structured Extraction
                </div>
                <span className="badge info">
                  {(result.entities._extraction_confidence * 100).toFixed(0)}% confidence
                </span>
              </div>

              {extractedGroups.length > 0 ? (
                <div className="entity-section">
                  {extractedGroups.map(([key, value]) => (
                    <div key={key} className="entity-group">
                      <div className="mini-label">{ENTITY_LABELS[key] || key}</div>
                      <div className="entity-chip-row">
                        {(value as string[]).map((item) => (
                          <span key={`${key}-${item}`} className="entity-chip">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state" style={{ padding: "24px 12px" }}>
                  <Sparkles size={32} />
                  <h3>No structured fields found</h3>
                  <p>The OCR text was produced successfully, but the NER stage did not match document fields.</p>
                </div>
              )}

              <div className="result-summary">
                {result.entities_summary || "No structured entities found."}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
