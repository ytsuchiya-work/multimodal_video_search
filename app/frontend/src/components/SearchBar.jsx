import React, { useState } from "react";

const styles = {
  container: {
    display: "flex",
    gap: "8px",
    maxWidth: "640px",
    margin: "0 auto 32px",
  },
  input: {
    flex: 1,
    padding: "12px 16px",
    fontSize: "15px",
    border: "2px solid #e5e7eb",
    borderRadius: "10px",
    outline: "none",
    transition: "border-color 0.2s",
  },
  button: {
    padding: "12px 24px",
    fontSize: "15px",
    fontWeight: "600",
    background: "#2563eb",
    color: "#fff",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    transition: "background 0.2s",
    whiteSpace: "nowrap",
  },
  buttonDisabled: {
    background: "#93c5fd",
    cursor: "not-allowed",
  },
  suggestions: {
    display: "flex",
    flexWrap: "wrap",
    gap: "8px",
    justifyContent: "center",
    marginTop: "12px",
  },
  chip: {
    padding: "6px 12px",
    fontSize: "12px",
    background: "#f3f4f6",
    border: "1px solid #e5e7eb",
    borderRadius: "16px",
    cursor: "pointer",
    transition: "all 0.2s",
  },
};

const SUGGESTIONS = [
  "データレイクハウスの説明",
  "機械学習のデモ",
  "SQLの実行",
  "ダッシュボード作成",
  "Unity Catalogの紹介",
];

export default function SearchBar({ onSearch, loading }) {
  const [value, setValue] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (value.trim() && !loading) {
      onSearch(value.trim());
    }
  };

  const handleSuggestion = (text) => {
    setValue(text);
    onSearch(text);
  };

  return (
    <div>
      <form onSubmit={handleSubmit} style={styles.container}>
        <input
          style={styles.input}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="検索したい動画の内容を入力..."
          disabled={loading}
        />
        <button
          style={{ ...styles.button, ...(loading ? styles.buttonDisabled : {}) }}
          type="submit"
          disabled={loading}
        >
          {loading ? "検索中..." : "検索"}
        </button>
      </form>
      <div style={styles.suggestions}>
        {SUGGESTIONS.map((s) => (
          <span
            key={s}
            style={styles.chip}
            onClick={() => handleSuggestion(s)}
          >
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}
