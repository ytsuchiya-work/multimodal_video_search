import React, { useState, useEffect, useRef } from "react";

const styles = {
  card: {
    border: "1px solid #e5e7eb",
    borderRadius: "12px",
    overflow: "hidden",
    transition: "box-shadow 0.2s, transform 0.2s",
    cursor: "pointer",
    background: "#fff",
  },
  cardHover: {
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    transform: "translateY(-2px)",
  },
  thumbnailContainer: {
    position: "relative",
    width: "100%",
    aspectRatio: "16/9",
    background: "#f3f4f6",
    overflow: "hidden",
  },
  thumbnail: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  thumbnailPlaceholder: {
    width: "100%",
    height: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#999",
    fontSize: "13px",
  },
  skeleton: {
    position: "absolute",
    inset: 0,
    background: "linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)",
    backgroundSize: "200% 100%",
    animation: "shimmer 1.4s infinite",
    transition: "opacity 0.2s",
  },
  timeBadge: {
    position: "absolute",
    bottom: "8px",
    right: "8px",
    background: "rgba(0,0,0,0.75)",
    color: "#fff",
    padding: "2px 6px",
    borderRadius: "4px",
    fontSize: "11px",
    fontWeight: "500",
  },
  scoreBadge: {
    position: "absolute",
    top: "8px",
    left: "8px",
    background: "rgba(37,99,235,0.9)",
    color: "#fff",
    padding: "2px 8px",
    borderRadius: "12px",
    fontSize: "11px",
    fontWeight: "600",
  },
  content: {
    padding: "12px 14px",
  },
  title: {
    fontSize: "14px",
    fontWeight: "600",
    color: "#1a1a1a",
    margin: "0 0 6px 0",
    lineHeight: "1.3",
    display: "-webkit-box",
    WebkitLineClamp: 2,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
  },
  meta: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    fontSize: "12px",
    color: "#888",
  },
  channel: {
    fontSize: "12px",
    color: "#666",
  },
  transcript: {
    fontSize: "12px",
    color: "#555",
    margin: "8px 0 0 0",
    lineHeight: "1.4",
    display: "-webkit-box",
    WebkitLineClamp: 2,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
    background: "#f9fafb",
    padding: "6px 8px",
    borderRadius: "4px",
    borderLeft: "3px solid #8b5cf6",
  },
  scoreBadgeMultimodal: {
    position: "absolute",
    top: "8px",
    left: "8px",
    background: "rgba(139,92,246,0.9)",
    color: "#fff",
    padding: "2px 8px",
    borderRadius: "12px",
    fontSize: "11px",
    fontWeight: "600",
  },
  playIcon: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: "48px",
    height: "48px",
    background: "rgba(0,0,0,0.6)",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    opacity: 0,
    transition: "opacity 0.2s",
  },
  playIconVisible: {
    opacity: 1,
  },
};

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const shimmerCSS = `
@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
`;

export default function VideoCard({ result, searchMode, onPlay }) {
  const [hovered, setHovered] = useState(false);
  const [imgError, setImgError] = useState(false);
  const [imgLoaded, setImgLoaded] = useState(false);

  const scorePercent = Math.round(result.score * 100);
  const isMultimodal = searchMode === "multimodal";

  return (
    <div
      style={{ ...styles.card, ...(hovered ? styles.cardHover : {}) }}
      onClick={() => onPlay(result)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <style>{shimmerCSS}</style>
      <div style={styles.thumbnailContainer}>
        {!imgError ? (
          <>
            <img
              style={styles.thumbnail}
              src={result.thumbnail_url}
              alt={result.title}
              onLoad={() => setImgLoaded(true)}
              onError={() => setImgError(true)}
            />
            {!imgLoaded && <div style={styles.skeleton} />}
          </>
        ) : (
          <div style={styles.thumbnailPlaceholder}>No Thumbnail</div>
        )}
        <div style={{ ...styles.playIcon, ...(hovered ? styles.playIconVisible : {}) }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
        <span style={styles.timeBadge}>
          {formatTime(result.start_time)} - {formatTime(result.end_time)}
        </span>
        {result.score > 0 && (
          <span style={isMultimodal ? styles.scoreBadgeMultimodal : styles.scoreBadge}>
            {scorePercent}%
          </span>
        )}
      </div>
      <div style={styles.content}>
        <h3 style={styles.title}>{result.title}</h3>
        <div style={styles.meta}>
          <span style={styles.channel}>{result.channel_name}</span>
        </div>
        {isMultimodal && result.transcript && (
          <p style={styles.transcript}>{result.transcript}</p>
        )}
      </div>
    </div>
  );
}
