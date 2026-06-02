import React from "react";
import VideoCard from "./VideoCard.jsx";

const styles = {
  container: {
    marginTop: "24px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "16px",
  },
  queryInfo: {
    fontSize: "14px",
    color: "#666",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
    gap: "16px",
  },
  empty: {
    textAlign: "center",
    padding: "48px",
    color: "#999",
    fontSize: "15px",
  },
  loading: {
    textAlign: "center",
    padding: "48px",
    color: "#666",
  },
  spinner: {
    display: "inline-block",
    width: "24px",
    height: "24px",
    border: "3px solid #e5e7eb",
    borderTopColor: "#2563eb",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
};

export default function ResultGrid({ results, loading, query, searched, searchMode, onPlay }) {
  if (loading) {
    return (
      <div style={styles.loading}>
        <div style={styles.spinner} />
        <p>検索中...</p>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (!searched) {
    return (
      <div style={styles.empty}>
        <p>テキストを入力して動画を検索してください</p>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div style={styles.empty}>
        <p>「{query}」に一致する動画が見つかりませんでした</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.queryInfo}>
          「{query}」の検索結果: {results.length} 件
        </span>
      </div>
      <div style={styles.grid}>
        {results.map((result) => (
          <VideoCard key={result.segment_id} result={result} searchMode={searchMode} onPlay={onPlay} />
        ))}
      </div>
    </div>
  );
}
