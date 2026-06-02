import React, { useState, useRef } from "react";

const styles = {
  overlay: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: "rgba(0,0,0,0.85)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
    padding: "24px",
  },
  modal: {
    background: "#fff",
    borderRadius: "16px",
    width: "100%",
    maxWidth: "900px",
    maxHeight: "90vh",
    overflow: "auto",
    boxShadow: "0 25px 60px rgba(0,0,0,0.5)",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "16px 20px",
    borderBottom: "1px solid #e5e7eb",
  },
  closeBtn: {
    background: "none",
    border: "none",
    fontSize: "24px",
    cursor: "pointer",
    color: "#666",
    padding: "4px 8px",
    borderRadius: "4px",
  },
  titleText: {
    fontSize: "16px",
    fontWeight: "600",
    color: "#1a1a1a",
    margin: 0,
  },
  videoContainer: {
    width: "100%",
    background: "#000",
  },
  video: {
    width: "100%",
    maxHeight: "480px",
    display: "block",
  },
  info: {
    padding: "16px 20px",
    borderBottom: "1px solid #f3f4f6",
  },
  videoTitle: {
    fontSize: "15px",
    fontWeight: "600",
    color: "#1a1a1a",
    margin: "0 0 4px 0",
  },
  channel: {
    fontSize: "13px",
    color: "#666",
    margin: 0,
  },
  clipSection: {
    padding: "20px",
  },
  sectionTitle: {
    fontSize: "14px",
    fontWeight: "600",
    color: "#1a1a1a",
    margin: "0 0 16px 0",
    paddingBottom: "8px",
    borderBottom: "2px solid #7c3aed",
    display: "inline-block",
  },
  timeRow: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    marginBottom: "12px",
  },
  timeLabel: {
    fontSize: "13px",
    fontWeight: "500",
    color: "#444",
    width: "40px",
  },
  timeInput: {
    width: "100px",
    padding: "8px 12px",
    fontSize: "14px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    outline: "none",
  },
  unit: {
    fontSize: "13px",
    color: "#666",
  },
  setTimeBtn: {
    padding: "6px 10px",
    fontSize: "11px",
    fontWeight: "500",
    background: "#f3f4f6",
    border: "1px solid #d1d5db",
    borderRadius: "4px",
    cursor: "pointer",
    color: "#374151",
  },
  checkbox: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "13px",
    color: "#444",
    margin: "16px 0",
  },
  btnCreate: {
    padding: "10px 20px",
    fontSize: "14px",
    fontWeight: "600",
    border: "none",
    borderRadius: "8px",
    background: "#7c3aed",
    color: "#fff",
    cursor: "pointer",
  },
  btnDisabled: {
    background: "#c4b5fd",
    cursor: "not-allowed",
  },
  success: {
    background: "#f0fdf4",
    border: "1px solid #bbf7d0",
    borderRadius: "8px",
    padding: "12px 16px",
    marginTop: "16px",
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  successText: {
    fontSize: "13px",
    color: "#166534",
    fontWeight: "600",
  },
  btnDownload: {
    padding: "8px 16px",
    fontSize: "13px",
    fontWeight: "600",
    background: "#2563eb",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    textDecoration: "none",
  },
  error: {
    fontSize: "13px",
    color: "#dc2626",
    marginBottom: "12px",
  },
};

function extractVideoId(segmentId) {
  return segmentId.replace(/_(seg|mm)\d+$/, "");
}

export default function VideoPlayer({ target, onClose }) {
  const videoRef = useRef(null);
  const [startTime, setStartTime] = useState(target.start_time);
  const [endTime, setEndTime] = useState(target.end_time);
  const [clipName, setClipName] = useState(
    `${target.title}_${Math.floor(target.start_time)}_${Math.floor(target.end_time)}`
  );
  const [saveToVolume, setSaveToVolume] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const videoId = extractVideoId(target.segment_id);
  const streamUrl = `/api/videos/${videoId}/stream#t=${Math.floor(target.start_time)}`;
  const workspaceHost = window.location.origin.replace(
    /video-search-cosmos-(\d+)\.(\d+)\.azure\.databricksapps\.com/,
    "adb-$1.$2.azuredatabricks.net"
  );
  const sourceVolumePath = `/Volumes/ytsuchiya/video_search/videos/${videoId}.mp4`;

  const updateStartTime = (val) => {
    setStartTime(val);
    setClipName(`${target.title}_${Math.floor(val)}_${Math.floor(endTime)}`);
    setResult(null);
    setError(null);
  };

  const updateEndTime = (val) => {
    setEndTime(val);
    setClipName(`${target.title}_${Math.floor(startTime)}_${Math.floor(val)}`);
    setResult(null);
    setError(null);
  };

  const setCurrentAsStart = () => {
    if (videoRef.current) {
      updateStartTime(Math.round(videoRef.current.currentTime * 10) / 10);
    }
  };

  const setCurrentAsEnd = () => {
    if (videoRef.current) {
      updateEndTime(Math.round(videoRef.current.currentTime * 10) / 10);
    }
  };

  const handleCreate = async () => {
    if (endTime <= startTime) {
      setError("終了時間は開始時間より後にしてください");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/clip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: videoId,
          start_time: startTime,
          end_time: endTime,
          save_to_volume: saveToVolume,
          clip_name: clipName.trim() || null,
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `エラー: ${res.statusText}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.titleText}>動画プレーヤー</h2>
          <button style={styles.closeBtn} onClick={onClose}>
            &times;
          </button>
        </div>

        <div style={styles.videoContainer}>
          <video
            ref={videoRef}
            style={styles.video}
            src={streamUrl}
            controls
            autoPlay
          />
        </div>

        <div style={styles.info}>
          <p style={styles.videoTitle}>{target.title}</p>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <p style={styles.channel}>{target.channel_name}</p>
            <a
              href={`${workspaceHost}/explore/data/Volumes/ytsuchiya/video_search/videos`}
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: "12px", color: "#2563eb" }}
            >
              元動画Volume
            </a>
          </div>
        </div>

        <div style={styles.clipSection}>
          <h3 style={styles.sectionTitle}>クリップ保存</h3>

          {error && <p style={styles.error}>{error}</p>}

          <div style={styles.timeRow}>
            <span style={styles.timeLabel}>開始</span>
            <input
              style={styles.timeInput}
              type="number"
              step="0.1"
              min="0"
              value={startTime}
              onChange={(e) => updateStartTime(parseFloat(e.target.value) || 0)}
              disabled={loading}
            />
            <span style={styles.unit}>秒</span>
            <button
              style={styles.setTimeBtn}
              onClick={setCurrentAsStart}
              disabled={loading}
            >
              現在位置を設定
            </button>
          </div>

          <div style={styles.timeRow}>
            <span style={styles.timeLabel}>終了</span>
            <input
              style={styles.timeInput}
              type="number"
              step="0.1"
              min="0"
              value={endTime}
              onChange={(e) => updateEndTime(parseFloat(e.target.value) || 0)}
              disabled={loading}
            />
            <span style={styles.unit}>秒</span>
            <button
              style={styles.setTimeBtn}
              onClick={setCurrentAsEnd}
              disabled={loading}
            >
              現在位置を設定
            </button>
          </div>

          <div style={{ marginBottom: "12px" }}>
            <span style={{ ...styles.timeLabel, width: "auto", marginBottom: "6px", display: "block" }}>
              保存名
            </span>
            <input
              style={{ ...styles.timeInput, width: "100%" }}
              type="text"
              value={clipName}
              onChange={(e) => setClipName(e.target.value)}
              disabled={loading}
              placeholder="クリップのファイル名"
            />
            <span style={{ fontSize: "11px", color: "#999" }}>.mp4</span>
          </div>

          <label style={styles.checkbox}>
            <input
              type="checkbox"
              checked={saveToVolume}
              onChange={(e) => setSaveToVolume(e.target.checked)}
              disabled={loading}
            />
            UC Volumeにも保存する
          </label>

          <button
            style={{ ...styles.btnCreate, ...(loading ? styles.btnDisabled : {}) }}
            onClick={handleCreate}
            disabled={loading}
          >
            {loading ? "作成中..." : "クリップ作成"}
          </button>

          {result && (
            <div style={styles.success}>
              <span style={styles.successText}>
                クリップ作成完了 ({result.duration}秒)
              </span>
              <a href={result.download_url} style={styles.btnDownload} download>
                ダウンロード
              </a>
            </div>
          )}
          {result && result.volume_path && (
            <p style={{ fontSize: "12px", color: "#666", marginTop: "8px" }}>
              Volume保存先:{" "}
              <a
                href={`${workspaceHost}/explore/data/Volumes/ytsuchiya/video_search/clips`}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "#2563eb" }}
              >
                {result.volume_path}
              </a>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
