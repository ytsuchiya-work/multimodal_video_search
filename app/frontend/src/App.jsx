import React, { useState, useEffect, useRef } from "react";
import SearchBar from "./components/SearchBar.jsx";
import ResultGrid from "./components/ResultGrid.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";

const styles = {
  container: {
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "24px",
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  header: { textAlign: "center", marginBottom: "24px" },
  title: { fontSize: "28px", fontWeight: "700", color: "#1a1a1a", margin: "0 0 8px 0" },
  subtitle: { fontSize: "14px", color: "#666", margin: 0 },
  stats: {
    display: "flex", justifyContent: "center", gap: "24px",
    marginTop: "12px", fontSize: "13px", color: "#888",
  },
  error: {
    background: "#fef2f2", border: "1px solid #fecaca",
    borderRadius: "8px", padding: "12px 16px", color: "#dc2626", marginTop: "16px",
  },
  clusterBar: {
    display: "flex", alignItems: "center", justifyContent: "center",
    gap: "12px", padding: "10px 16px", borderRadius: "8px",
    marginBottom: "16px", fontSize: "13px",
  },
  clusterRunning: { background: "#f0fdf4", border: "1px solid #bbf7d0", color: "#166534" },
  clusterStopped: { background: "#fef2f2", border: "1px solid #fecaca", color: "#991b1b" },
  clusterPending: { background: "#fffbeb", border: "1px solid #fde68a", color: "#92400e" },
  clusterUnknown: { background: "#f9fafb", border: "1px solid #e5e7eb", color: "#6b7280" },
  dot: { width: "8px", height: "8px", borderRadius: "50%", display: "inline-block" },
  tabs: {
    display: "flex", justifyContent: "center", gap: "4px", marginBottom: "24px",
    background: "#f3f4f6", borderRadius: "10px", padding: "4px",
    maxWidth: "400px", margin: "0 auto 24px",
  },
  tab: {
    flex: 1, padding: "10px 16px", fontSize: "13px", fontWeight: "500",
    border: "none", borderRadius: "8px", cursor: "pointer",
    background: "transparent", color: "#666", transition: "all 0.2s",
  },
  tabActive: {
    background: "#fff", color: "#1a1a1a", fontWeight: "600",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
  // Endpoint panel styles
  epToggle: {
    display: "flex", justifyContent: "center", marginBottom: "12px",
  },
  epToggleBtn: {
    padding: "6px 16px", fontSize: "12px", fontWeight: "500",
    background: "transparent", border: "1px solid #d1d5db",
    borderRadius: "20px", cursor: "pointer", color: "#6b7280",
    display: "flex", alignItems: "center", gap: "6px",
  },
  epPanel: {
    background: "#f9fafb", border: "1px solid #e5e7eb",
    borderRadius: "12px", padding: "16px", marginBottom: "20px",
  },
  epPanelTitle: {
    fontSize: "13px", fontWeight: "600", color: "#374151",
    marginBottom: "12px", display: "flex", alignItems: "center", gap: "8px",
  },
  epCards: { display: "flex", flexDirection: "column", gap: "10px" },
  epCard: {
    background: "#fff", border: "1px solid #e5e7eb",
    borderRadius: "8px", padding: "12px 16px",
    display: "flex", alignItems: "flex-start", gap: "12px",
  },
  epCardBody: { flex: 1, minWidth: 0 },
  epCardName: { fontSize: "14px", fontWeight: "600", color: "#111827", marginBottom: "3px" },
  epCardDesc: { fontSize: "12px", color: "#6b7280", lineHeight: "1.5", marginBottom: "6px" },
  epCardMeta: { display: "flex", gap: "12px", fontSize: "11px", color: "#9ca3af" },
  epBadge: {
    display: "inline-flex", alignItems: "center", gap: "4px",
    padding: "2px 8px", borderRadius: "12px", fontSize: "11px", fontWeight: "500",
    whiteSpace: "nowrap", flexShrink: 0,
  },
  epBadgeReady: { background: "#f0fdf4", color: "#166534" },
  epBadgeNotReady: { background: "#fef2f2", color: "#991b1b" },
  epBadgeUnknown: { background: "#f9fafb", color: "#6b7280" },
  epCardActions: {
    display: "flex", flexDirection: "column", alignItems: "flex-end",
    gap: "8px", flexShrink: 0,
  },
  warmupBtn: {
    padding: "5px 12px", fontSize: "11px", fontWeight: "500",
    background: "#eff6ff", color: "#2563eb", border: "1px solid #bfdbfe",
    borderRadius: "6px", cursor: "pointer", whiteSpace: "nowrap",
  },
  warmupBtnLoading: { background: "#f0fdf4", color: "#166534", border: "1px solid #bbf7d0", cursor: "not-allowed" },
};

// ── Endpoints Panel ───────────────────────────────────────────────────────────

function EndpointsPanel() {
  const [endpoints, setEndpoints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [warmingUp, setWarmingUp] = useState({});
  const [warmupDone, setWarmupDone] = useState({});

  const fetchEndpoints = async () => {
    setLoading(true);
    try {
      const data = await fetch("/api/endpoints").then((r) => r.json());
      setEndpoints(data.endpoints || []);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEndpoints();
    const id = setInterval(fetchEndpoints, 20000);
    return () => clearInterval(id);
  }, []);

  const handleWarmup = async (name) => {
    setWarmingUp((w) => ({ ...w, [name]: true }));
    setWarmupDone((d) => ({ ...d, [name]: false }));
    try {
      await fetch(`/api/endpoints/${name}/warmup`, { method: "POST" });
      // Poll until READY or error
      for (let i = 0; i < 90; i++) {
        await new Promise((r) => setTimeout(r, 3000));
        const data = await fetch("/api/endpoints").then((r) => r.json());
        const ep = (data.endpoints || []).find((e) => e.name === name);
        if (ep && ep.ready === "READY") {
          setEndpoints(data.endpoints);
          setWarmupDone((d) => ({ ...d, [name]: true }));
          break;
        }
      }
    } catch {
      // ignore
    } finally {
      setWarmingUp((w) => ({ ...w, [name]: false }));
    }
  };

  const badgeStyle = (ready) => {
    if (ready === "READY") return { ...styles.epBadge, ...styles.epBadgeReady };
    if (ready === "UNKNOWN") return { ...styles.epBadge, ...styles.epBadgeUnknown };
    return { ...styles.epBadge, ...styles.epBadgeNotReady };
  };

  const badgeDot = (ready) => {
    if (ready === "READY") return "#22c55e";
    if (ready === "UNKNOWN") return "#9ca3af";
    return "#ef4444";
  };

  const badgeLabel = (ready, configUpdate) => {
    if (configUpdate === "UPDATE_IN_PROGRESS") return "更新中";
    if (ready === "READY") return "READY";
    if (ready === "NOT_READY") return "起動中";
    return ready || "不明";
  };

  return (
    <div style={styles.epPanel}>
      <div style={styles.epPanelTitle}>
        <span>⚡ サービングエンドポイント</span>
        {loading && <span style={{ fontSize: "11px", color: "#9ca3af" }}>更新中...</span>}
        <button
          onClick={fetchEndpoints}
          style={{ marginLeft: "auto", fontSize: "11px", color: "#6b7280",
            background: "none", border: "none", cursor: "pointer" }}
        >
          ↻ 更新
        </button>
      </div>
      <div style={styles.epCards}>
        {endpoints.map((ep) => (
          <div key={ep.name} style={styles.epCard}>
            <div style={styles.epCardBody}>
              <div style={styles.epCardName}>{ep.display_name}</div>
              <div style={styles.epCardDesc}>{ep.description}</div>
              <div style={styles.epCardMeta}>
                <span>モデル: {ep.model}</span>
                <span>次元: {ep.dimension}</span>
                <span>用途: {ep.usage}</span>
                {ep.version && ep.version !== "?" && <span>v{ep.version}</span>}
              </div>
            </div>
            <div style={styles.epCardActions}>
              <span style={badgeStyle(ep.ready)}>
                <span style={{ ...styles.dot, background: badgeDot(ep.ready) }} />
                {badgeLabel(ep.ready, ep.config_update)}
              </span>
              <button
                style={
                  warmingUp[ep.name]
                    ? { ...styles.warmupBtn, ...styles.warmupBtnLoading }
                    : styles.warmupBtn
                }
                onClick={() => !warmingUp[ep.name] && handleWarmup(ep.name)}
                disabled={warmingUp[ep.name]}
              >
                {warmingUp[ep.name]
                  ? "⏳ 起動中..."
                  : warmupDone[ep.name]
                  ? "✓ 完了"
                  : "▶ ウォームアップ"}
              </button>
            </div>
          </div>
        ))}
        {endpoints.length === 0 && !loading && (
          <p style={{ fontSize: "12px", color: "#9ca3af", textAlign: "center" }}>
            エンドポイント情報を読み込み中...
          </p>
        )}
      </div>
    </div>
  );
}

// ── Cluster status bar ────────────────────────────────────────────────────────

function ClusterStatusBar({ onToggleEndpoints, showEndpoints }) {
  const [clusterState, setClusterState] = useState(null);
  const pollRef = useRef(null);

  const fetchStatus = async () => {
    try {
      const data = await fetch("/api/cluster/status").then((r) => r.json());
      setClusterState(data.state);
      if (data.state === "RUNNING") stopPolling();
    } catch {
      setClusterState("UNKNOWN");
    }
  };

  const startPolling = () => {
    if (pollRef.current) return;
    pollRef.current = setInterval(fetchStatus, 10000);
  };

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  useEffect(() => {
    fetchStatus();
    return () => stopPolling();
  }, []);

  useEffect(() => {
    if (clusterState && clusterState !== "RUNNING") startPolling();
  }, [clusterState]);

  if (!clusterState) return null;

  const isRunning = clusterState === "RUNNING";
  const isPending = clusterState === "STARTING";
  let barStyle = styles.clusterUnknown;
  if (isRunning) barStyle = styles.clusterRunning;
  else if (isPending) barStyle = styles.clusterPending;

  return (
    <div style={{ ...styles.clusterBar, ...barStyle }}>
      <span style={{ ...styles.dot, background: isRunning ? "#22c55e" : isPending ? "#f59e0b" : "#9ca3af" }} />
      {isRunning && <span>GPU Ready — 検索可能です</span>}
      {isPending && <span>エンドポイント確認中...</span>}
      {!isRunning && !isPending && <span>エンドポイント: {clusterState}</span>}
      <button
        onClick={onToggleEndpoints}
        style={{
          marginLeft: "8px", padding: "3px 10px", fontSize: "11px",
          background: "rgba(0,0,0,0.06)", border: "none", borderRadius: "12px",
          cursor: "pointer", color: "inherit",
        }}
      >
        {showEndpoints ? "▲ 閉じる" : "⚡ エンドポイント管理"}
      </button>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 150; // 5 minutes

export default function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("検索中...");
  const [error, setError] = useState(null);
  const [query, setQuery] = useState("");
  const [videoCount, setVideoCount] = useState(0);
  const [searched, setSearched] = useState(false);
  const [searchMode, setSearchMode] = useState("cosmos");
  const [playTarget, setPlayTarget] = useState(null);
  const [showEndpoints, setShowEndpoints] = useState(false);

  useEffect(() => {
    fetch("/api/videos")
      .then((r) => r.json())
      .then((data) => setVideoCount(data.videos?.length || 0))
      .catch(() => {});
  }, []);

  const handleSearch = async (searchQuery) => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setLoadingMsg("検索リクエスト送信中...");
    setError(null);
    setQuery(searchQuery);
    setSearched(true);
    setResults([]);

    const endpoint = searchMode === "cosmos" ? "/api/search" : "/api/search/multimodal";

    try {
      // Submit search task (returns immediately)
      const submitRes = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, num_results: 12 }),
      });
      if (!submitRes.ok) {
        const err = await submitRes.json().catch(() => ({}));
        throw new Error(err.detail || `送信エラー (HTTP ${submitRes.status})`);
      }
      const { task_id } = await submitRes.json();

      // Poll for result
      const startTime = Date.now();
      for (let i = 0; i < MAX_POLL_ATTEMPTS; i++) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        const elapsed = Math.round((Date.now() - startTime) / 1000);

        if (elapsed < 10) setLoadingMsg("エンドポイントに接続中...");
        else if (elapsed < 60) setLoadingMsg(`embedding 計算中... (${elapsed}秒)`);
        else setLoadingMsg(`GPUエンドポイント起動中... (${elapsed}秒 / コールドスタートは2〜3分かかります)`);

        const pollRes = await fetch(`/api/search/result/${task_id}`);
        if (!pollRes.ok) continue;
        const pollData = await pollRes.json();

        if (pollData.status === "done") {
          setResults(pollData.results || []);
          return;
        }
        if (pollData.status === "error") {
          const msg = pollData.error || "検索中にエラーが発生しました";
          if (msg.includes("timed out") || msg.includes("timeout") || msg.includes("Read timed out")) {
            setError("⏱ GPUエンドポイントのコールドスタートがタイムアウトしました。「ウォームアップ」ボタンでエンドポイントを先に起動してから再試行してください。");
          } else {
            setError(msg);
          }
          return;
        }
        // status === "pending" → continue polling
      }
      setError("検索タイムアウト: エンドポイントの起動に時間がかかっています。「⚡ エンドポイント管理」からウォームアップをお試しください。");
    } catch (e) {
      setError(e.message || "ネットワークエラーが発生しました");
    } finally {
      setLoading(false);
    }
  };

  const handleModeChange = (mode) => {
    setSearchMode(mode);
    setResults([]);
    setSearched(false);
    setError(null);
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Video Search</h1>
        <p style={styles.subtitle}>
          {searchMode === "cosmos"
            ? "Cosmos-Embed1 による動画セマンティック検索"
            : "音声文字起こし + 画像フレームによるマルチモーダル検索"}
        </p>
        <div style={styles.stats}>
          <span>登録動画: {videoCount} 本</span>
          <span>
            {searchMode === "cosmos"
              ? "Powered by NVIDIA Cosmos-Embed1-448p"
              : "Powered by Whisper + CLIP + multilingual-e5-large"}
          </span>
        </div>
      </header>

      <ClusterStatusBar
        onToggleEndpoints={() => setShowEndpoints((s) => !s)}
        showEndpoints={showEndpoints}
      />

      {showEndpoints && <EndpointsPanel />}

      <div style={styles.tabs}>
        <button
          style={{ ...styles.tab, ...(searchMode === "cosmos" ? styles.tabActive : {}) }}
          onClick={() => handleModeChange("cosmos")}
        >
          Cosmos検索
        </button>
        <button
          style={{ ...styles.tab, ...(searchMode === "multimodal" ? styles.tabActive : {}) }}
          onClick={() => handleModeChange("multimodal")}
        >
          マルチモーダル検索
        </button>
      </div>

      <SearchBar onSearch={handleSearch} loading={loading} loadingMsg={loadingMsg} />

      {error && <div style={styles.error}>{error}</div>}

      <ResultGrid
        results={results}
        loading={loading}
        query={query}
        searched={searched}
        searchMode={searchMode}
        onPlay={setPlayTarget}
      />

      {playTarget && (
        <VideoPlayer target={playTarget} onClose={() => setPlayTarget(null)} />
      )}
    </div>
  );
}
