import React, { useState, useEffect, useRef } from "react";
import SearchBar from "./components/SearchBar.jsx";
import ResultGrid from "./components/ResultGrid.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";

const styles = {
  container: {
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "24px",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  header: {
    textAlign: "center",
    marginBottom: "32px",
  },
  title: {
    fontSize: "28px",
    fontWeight: "700",
    color: "#1a1a1a",
    margin: "0 0 8px 0",
  },
  subtitle: {
    fontSize: "14px",
    color: "#666",
    margin: 0,
  },
  stats: {
    display: "flex",
    justifyContent: "center",
    gap: "24px",
    marginTop: "16px",
    fontSize: "13px",
    color: "#888",
  },
  error: {
    background: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: "8px",
    padding: "12px 16px",
    color: "#dc2626",
    marginTop: "16px",
  },
  clusterBar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "12px",
    padding: "10px 16px",
    borderRadius: "8px",
    marginBottom: "24px",
    fontSize: "13px",
  },
  clusterRunning: {
    background: "#f0fdf4",
    border: "1px solid #bbf7d0",
    color: "#166534",
  },
  clusterStopped: {
    background: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#991b1b",
  },
  clusterPending: {
    background: "#fffbeb",
    border: "1px solid #fde68a",
    color: "#92400e",
  },
  clusterUnknown: {
    background: "#f9fafb",
    border: "1px solid #e5e7eb",
    color: "#6b7280",
  },
  startButton: {
    padding: "6px 14px",
    fontSize: "12px",
    fontWeight: "600",
    background: "#2563eb",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },
  startButtonDisabled: {
    background: "#93c5fd",
    cursor: "not-allowed",
  },
  dot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    display: "inline-block",
  },
  tabs: {
    display: "flex",
    justifyContent: "center",
    gap: "4px",
    marginBottom: "24px",
    background: "#f3f4f6",
    borderRadius: "10px",
    padding: "4px",
    maxWidth: "400px",
    margin: "0 auto 24px",
  },
  tab: {
    flex: 1,
    padding: "10px 16px",
    fontSize: "13px",
    fontWeight: "500",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    background: "transparent",
    color: "#666",
    transition: "all 0.2s",
  },
  tabActive: {
    background: "#fff",
    color: "#1a1a1a",
    fontWeight: "600",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
};

function ClusterStatusBar() {
  const [clusterState, setClusterState] = useState(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/cluster/status");
      if (res.ok) {
        const data = await res.json();
        setClusterState(data.state);
        if (data.state === "RUNNING") {
          setStarting(false);
          stopPolling();
        }
      }
    } catch {
      setClusterState("UNKNOWN");
    }
  };

  const startPolling = () => {
    if (pollRef.current) return;
    pollRef.current = setInterval(fetchStatus, 10000);
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => {
    fetchStatus();
    return () => stopPolling();
  }, []);

  useEffect(() => {
    if (
      clusterState &&
      clusterState !== "RUNNING" &&
      clusterState !== "TERMINATED"
    ) {
      startPolling();
    }
  }, [clusterState]);

  const handleStart = async () => {
    setStarting(true);
    try {
      const res = await fetch("/api/cluster/start", { method: "POST" });
      if (res.ok) {
        setClusterState("PENDING");
        startPolling();
      } else {
        setStarting(false);
      }
    } catch {
      setStarting(false);
    }
  };

  if (!clusterState) return null;

  const isPending =
    clusterState === "PENDING" ||
    clusterState === "RESTARTING" ||
    clusterState === "RESIZING";
  const isRunning = clusterState === "RUNNING";
  const isStopped = clusterState === "TERMINATED" || clusterState === "TERMINATING";

  let barStyle = styles.clusterUnknown;
  if (isRunning) barStyle = styles.clusterRunning;
  else if (isStopped) barStyle = styles.clusterStopped;
  else if (isPending) barStyle = styles.clusterPending;

  let dotColor = "#9ca3af";
  if (isRunning) dotColor = "#22c55e";
  else if (isStopped) dotColor = "#ef4444";
  else if (isPending) dotColor = "#f59e0b";

  return (
    <div style={{ ...styles.clusterBar, ...barStyle }}>
      <span style={{ ...styles.dot, background: dotColor }} />
      {isRunning && <span>GPU Ready — 検索可能です</span>}
      {isStopped && !starting && (
        <>
          <span>GPUクラスタ停止中 — 検索するにはクラスタの起動が必要です</span>
          <button style={styles.startButton} onClick={handleStart}>
            クラスタを起動
          </button>
        </>
      )}
      {(isPending || starting) && (
        <span>GPUクラスタ起動中... （3〜5分かかります）</span>
      )}
      {!isRunning && !isStopped && !isPending && !starting && (
        <span>GPUクラスタ: {clusterState}</span>
      )}
    </div>
  );
}

export default function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState("");
  const [videoCount, setVideoCount] = useState(0);
  const [searched, setSearched] = useState(false);
  const [searchMode, setSearchMode] = useState("cosmos");
  const [playTarget, setPlayTarget] = useState(null);

  useEffect(() => {
    fetch("/api/videos")
      .then((r) => r.json())
      .then((data) => setVideoCount(data.videos?.length || 0))
      .catch(() => {});
  }, []);

  const handleSearch = async (searchQuery) => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);
    setQuery(searchQuery);
    setSearched(true);

    const endpoint =
      searchMode === "cosmos" ? "/api/search" : "/api/search/multimodal";

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, num_results: 12 }),
        signal: AbortSignal.timeout(310000),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const detail = errData.detail || `検索エラー: ${res.statusText}`;
        if (res.status === 503) {
          throw new Error("🔄 GPUエンドポイントが起動中です（初回リクエスト時は2〜3分かかります）。しばらく待ってから再度お試しください。");
        }
        throw new Error(detail);
      }

      const data = await res.json();
      setResults(data.results || []);
    } catch (e) {
      if (e.name === "TimeoutError" || e.name === "AbortError" || (e.message && e.message.includes("timed out"))) {
        setError("🔄 GPUエンドポイントが起動中です（スケールゼロからの復帰に2〜3分かかります）。しばらく待ってから再度お試しください。");
      } else {
        setError(e.message);
      }
      setResults([]);
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

      <ClusterStatusBar />

      <div style={styles.tabs}>
        <button
          style={{
            ...styles.tab,
            ...(searchMode === "cosmos" ? styles.tabActive : {}),
          }}
          onClick={() => handleModeChange("cosmos")}
        >
          Cosmos検索
        </button>
        <button
          style={{
            ...styles.tab,
            ...(searchMode === "multimodal" ? styles.tabActive : {}),
          }}
          onClick={() => handleModeChange("multimodal")}
        >
          マルチモーダル検索
        </button>
      </div>

      <SearchBar onSearch={handleSearch} loading={loading} />

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
