import { useEffect, useState } from "react";
import { Cpu, Database, Image, PlugZap, Waypoints } from "lucide-react";

import { fetchDatasetInfo, fetchHealth, fetchRandomSamples } from "../api";

interface DatasetSplit {
  num_samples: number;
  columns: string[];
}

interface Sample {
  text: string;
  image_base64: string;
  index: number;
  split: string;
}

interface HealthData {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
  model: {
    base_model_name: string;
    adapter_path: string | null;
    adapter_source: string;
    device: string;
    using_adapter: boolean;
  };
}

export default function Dashboard() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [info, setInfo] = useState<Record<string, DatasetSplit> | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState(true);
  const [systemError, setSystemError] = useState<string | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    setLoading(true);
    setSystemError(null);
    setDatasetError(null);

    const [healthResult, infoResult, samplesResult] = await Promise.allSettled([
      fetchHealth(),
      fetchDatasetInfo(),
      fetchRandomSamples("train", 8),
    ]);

    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
    } else {
      setHealth(null);
      setSystemError(healthResult.reason?.message || "Failed to connect to backend");
    }

    if (infoResult.status === "fulfilled") {
      setInfo(infoResult.value);
    } else {
      setInfo(null);
      setDatasetError(infoResult.reason?.message || "Dataset metadata is unavailable");
    }

    if (samplesResult.status === "fulfilled") {
      setSamples(samplesResult.value);
    } else {
      setSamples([]);
      setDatasetError(samplesResult.reason?.message || "Dataset samples are unavailable");
    }

    setLoading(false);
  }

  const totalSamples = info
    ? Object.values(info).reduce((sum, split) => sum + (split.num_samples || 0), 0)
    : 0;

  const adapterName = health?.model.adapter_path?.split(/[/\\]/).pop();

  if (loading && !health) {
    return (
      <div>
        <div className="page-header animate-in">
          <h1>Dashboard</h1>
          <p>Overview of the DevGen OCR workspace</p>
        </div>
        <div className="loading-spinner" />
      </div>
    );
  }

  if (systemError && !health) {
    return (
      <div>
        <div className="page-header animate-in">
          <h1>Dashboard</h1>
          <p>Overview of the DevGen OCR workspace</p>
        </div>
        <div className="card" style={{ textAlign: "center", padding: "60px 20px" }}>
          <p style={{ color: "var(--error)", marginBottom: "16px", fontSize: "15px" }}>
            Cannot connect to the backend
          </p>
          <p style={{ color: "var(--text-muted)", fontSize: "13px", marginBottom: "20px" }}>
            Start the API with <code>python -m backend.main</code> from the project root.
          </p>
          <button className="btn btn-primary" onClick={loadData}>
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header animate-in">
        <h1>Dashboard</h1>
        <p>
          Your frontend is now connected to the FastAPI backend and the TrOCR LoRA
          checkpoint inside this project.
        </p>
      </div>

      <div className="stats-grid">
        <div className="stat-card animate-in animate-in-delay-1">
          <div className="stat-icon green">
            <PlugZap size={20} />
          </div>
          <div className="stat-value" style={{ fontSize: "18px" }}>
            {health?.status === "ok" ? "Online" : "Offline"}
          </div>
          <div className="stat-label">Backend API</div>
        </div>

        <div className="stat-card animate-in animate-in-delay-1">
          <div className="stat-icon purple">
            <Cpu size={20} />
          </div>
          <div className="stat-value" style={{ fontSize: "18px" }}>
            {health?.model.using_adapter ? adapterName || "LoRA" : "Base"}
          </div>
          <div className="stat-label">Active OCR Model</div>
        </div>

        <div className="stat-card animate-in animate-in-delay-2">
          <div className="stat-icon blue">
            <Database size={20} />
          </div>
          <div className="stat-value">{totalSamples.toLocaleString()}</div>
          <div className="stat-label">Total Samples</div>
        </div>

        <div className="stat-card animate-in animate-in-delay-3">
          <div className="stat-icon amber">
            <Waypoints size={20} />
          </div>
          <div className="stat-value" style={{ fontSize: "18px" }}>
            {health?.model.device || "—"}
          </div>
          <div className="stat-label">Inference Device</div>
        </div>
      </div>

      <div className="card animate-in animate-in-delay-2" style={{ marginBottom: "24px" }}>
        <div className="card-header card-header-wrap">
          <div className="card-title">
            <Cpu size={20} />
            Runtime Summary
          </div>
          <button className="btn btn-ghost" onClick={loadData}>
            Refresh
          </button>
        </div>
        <div className="info-grid">
          <div className="info-chip">
            <span className="info-chip-label">Service</span>
            <span className="info-chip-value">{health?.service || "DevGen"}</span>
          </div>
          <div className="info-chip">
            <span className="info-chip-label">Version</span>
            <span className="info-chip-value">{health?.version || "0.1.0"}</span>
          </div>
          <div className="info-chip">
            <span className="info-chip-label">Base model</span>
            <span className="info-chip-value">{health?.model.base_model_name || "—"}</span>
          </div>
          <div className="info-chip">
            <span className="info-chip-label">Adapter source</span>
            <span className="info-chip-value">{health?.model.adapter_source || "—"}</span>
          </div>
        </div>
      </div>

      <div className="card animate-in animate-in-delay-3">
        <div className="card-header card-header-wrap">
          <div className="card-title">
            <Image size={20} />
            Random Training Samples
          </div>
        </div>

        {datasetError && (
          <div className="inline-alert warning" style={{ marginBottom: "16px" }}>
            <span>{datasetError}</span>
          </div>
        )}

        {samples.length > 0 ? (
          <div className="dataset-grid">
            {samples.map((sample) => (
              <div key={`${sample.split}-${sample.index}`} className="sample-card">
                <div className="sample-card-image">
                  <img src={`data:image/jpeg;base64,${sample.image_base64}`} alt={sample.text} />
                </div>
                <div className="sample-card-body">
                  <div className="sample-card-label">{sample.text}</div>
                  <div className="sample-card-index">
                    idx: {sample.index} · {sample.split}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <Database size={40} />
            <h3>Dataset preview unavailable</h3>
            <p>The backend is online, but the sample dataset could not be fetched right now.</p>
          </div>
        )}
      </div>
    </div>
  );
}
