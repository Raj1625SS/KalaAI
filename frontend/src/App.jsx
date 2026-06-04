import { useState, useRef } from "react";
import axios from "axios";

const CLASS_INFO = {
  mughal: {
    emoji: "🌸",
    origin: "Mughal Empire, 16th–19th century",
    description: "Ornate floral and geometric patterns inspired by Mughal court art. Rich in symmetry and intricate detail.",
    color: "#C8860A",
    bg: "#FDF3DC",
  },
  sanganeri: {
    emoji: "🌼",
    origin: "Sanganer, Rajasthan",
    description: "Delicate floral motifs on white or pastel backgrounds. Known for fine lines and soft color palettes.",
    color: "#2E7D5E",
    bg: "#E8F5EE",
  },
  kalamkari: {
    emoji: "🦚",
    origin: "Andhra Pradesh & Telangana",
    description: "Hand-painted or block-printed narrative scenes from Hindu epics. Uses natural dyes on cotton.",
    color: "#7B3FA0",
    bg: "#F3E8FB",
  },
  gamthi: {
    emoji: "🪬",
    origin: "Gujarat, Western India",
    description: "Bold folk patterns with geometric motifs. Reflects Gujarat's vibrant tribal textile traditions.",
    color: "#C0392B",
    bg: "#FDECEA",
  },
};

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setPreview(URL.createObjectURL(f));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  };

  const predict = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", formData);
      setResult(res.data);
    } catch {
      alert("Prediction failed. Make sure the FastAPI backend is running.");
    }
    setLoading(false);
  };

  const info = result ? CLASS_INFO[result.pattern.toLowerCase()] : null;

  return (
    <div style={styles.page}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}>🪡</span>
          <div>
            <div style={styles.logoName}>KalaAI</div>
            <div style={styles.logoSub}>Textile Heritage Intelligence</div>
          </div>
        </div>
        <div style={styles.badge}>AI · Culture · Heritage</div>
      </header>

      {/* Hero */}
      <section style={styles.hero}>
        <h1 style={styles.heroTitle}>
          Identify Indian Block Print Traditions
        </h1>
        <p style={styles.heroSub}>
          Upload a textile image — KalaAI uses a Vision Transformer (DeiT) to
          classify the print tradition and surface its cultural story.
        </p>
      </section>

      {/* Main card */}
      <main style={styles.main}>
        {/* Upload zone */}
        <div
          style={{
            ...styles.dropzone,
            borderColor: dragging ? "#C8860A" : "#D4C5A9",
            background: dragging ? "#FDF3DC" : preview ? "#FAFAF8" : "#FAFAF8",
          }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {preview ? (
            <img src={preview} alt="preview" style={styles.preview} />
          ) : (
            <div style={styles.uploadPrompt}>
              <div style={styles.uploadIcon}>🖼️</div>
              <div style={styles.uploadText}>Drop a textile image here</div>
              <div style={styles.uploadHint}>or click to browse — JPG, PNG, WEBP</div>
            </div>
          )}
        </div>

        {/* Predict button */}
        <button
          style={{
            ...styles.btn,
            opacity: file && !loading ? 1 : 0.5,
            cursor: file && !loading ? "pointer" : "not-allowed",
          }}
          onClick={predict}
          disabled={!file || loading}
        >
          {loading ? (
            <span style={styles.btnInner}>
              <span style={styles.spinner} /> Analysing...
            </span>
          ) : (
            <span style={styles.btnInner}>✦ Identify Textile</span>
          )}
        </button>

        {/* Result */}
        {result && info && (
          <div style={{ ...styles.resultCard, borderColor: info.color }}>
            {/* Pattern name */}
            <div style={{ ...styles.resultHeader, background: info.bg }}>
              <span style={styles.resultEmoji}>{info.emoji}</span>
              <div>
                <div style={{ ...styles.resultPattern, color: info.color }}>
                  {result.pattern}
                </div>
                <div style={styles.resultOrigin}>{info.origin}</div>
              </div>
              <div style={{ ...styles.confidenceBadge, background: info.color }}>
                {(result.confidence * 100).toFixed(1)}%
              </div>
            </div>

            {/* Description */}
            <p style={styles.resultDesc}>{info.description}</p>

            {/* Confidence bars */}
            <div style={styles.barsLabel}>Class Probabilities</div>
            <div style={styles.bars}>
              {Object.entries(result.class_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, prob]) => {
                  const c = CLASS_INFO[cls];
                  return (
                    <div key={cls} style={styles.barRow}>
                      <div style={styles.barName}>
                        {c.emoji} {cls.charAt(0).toUpperCase() + cls.slice(1)}
                      </div>
                      <div style={styles.barTrack}>
                        <div
                          style={{
                            ...styles.barFill,
                            width: `${(prob * 100).toFixed(1)}%`,
                            background: c.color,
                          }}
                        />
                      </div>
                      <div style={styles.barPct}>
                        {(prob * 100).toFixed(1)}%
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer style={styles.footer}>
        KalaAI · Tradition Hacks 2026 · Preserving Indian textile heritage through Explainable AI
      </footer>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#F7F3ED",
    fontFamily: "'Georgia', serif",
    color: "#2C2416",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "20px 40px",
    borderBottom: "1px solid #E0D5C5",
    background: "#FFFDF9",
  },
  logo: { display: "flex", alignItems: "center", gap: 12 },
  logoIcon: { fontSize: 32 },
  logoName: { fontSize: 22, fontWeight: "bold", letterSpacing: 1 },
  logoSub: { fontSize: 11, color: "#8B7355", letterSpacing: 2, textTransform: "uppercase" },
  badge: {
    fontSize: 11,
    letterSpacing: 2,
    textTransform: "uppercase",
    color: "#8B7355",
    border: "1px solid #D4C5A9",
    borderRadius: 20,
    padding: "5px 14px",
  },
  hero: {
    textAlign: "center",
    padding: "48px 20px 24px",
    maxWidth: 640,
    margin: "0 auto",
  },
  heroTitle: {
    fontSize: 30,
    fontWeight: "bold",
    margin: "0 0 12px",
    lineHeight: 1.3,
  },
  heroSub: {
    fontSize: 15,
    color: "#6B5A3E",
    lineHeight: 1.7,
    margin: 0,
  },
  main: {
    maxWidth: 600,
    margin: "0 auto",
    padding: "24px 20px 60px",
    display: "flex",
    flexDirection: "column",
    gap: 16,
  },
  dropzone: {
    border: "2px dashed",
    borderRadius: 16,
    minHeight: 220,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    transition: "all 0.2s",
    overflow: "hidden",
  },
  uploadPrompt: { textAlign: "center", padding: 32 },
  uploadIcon: { fontSize: 48, marginBottom: 12 },
  uploadText: { fontSize: 16, fontWeight: "bold", marginBottom: 6 },
  uploadHint: { fontSize: 13, color: "#8B7355" },
  preview: { width: "100%", maxHeight: 320, objectFit: "contain", borderRadius: 14 },
  btn: {
    width: "100%",
    padding: "14px 0",
    background: "#2C2416",
    color: "#F7F3ED",
    border: "none",
    borderRadius: 12,
    fontSize: 15,
    fontFamily: "'Georgia', serif",
    letterSpacing: 1,
    transition: "opacity 0.2s",
  },
  btnInner: { display: "flex", alignItems: "center", justifyContent: "center", gap: 8 },
  spinner: {
    display: "inline-block",
    width: 14,
    height: 14,
    border: "2px solid #F7F3ED",
    borderTopColor: "transparent",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  resultCard: {
    border: "1.5px solid",
    borderRadius: 16,
    overflow: "hidden",
    background: "#FFFDF9",
  },
  resultHeader: {
    display: "flex",
    alignItems: "center",
    gap: 14,
    padding: "18px 20px",
  },
  resultEmoji: { fontSize: 36 },
  resultPattern: { fontSize: 22, fontWeight: "bold" },
  resultOrigin: { fontSize: 12, color: "#6B5A3E", marginTop: 2 },
  confidenceBadge: {
    marginLeft: "auto",
    color: "white",
    borderRadius: 20,
    padding: "5px 14px",
    fontSize: 14,
    fontWeight: "bold",
  },
  resultDesc: {
    padding: "0 20px 16px",
    fontSize: 14,
    color: "#4A3C2A",
    lineHeight: 1.7,
    borderBottom: "1px solid #EDE5D8",
    margin: 0,
  },
  barsLabel: {
    padding: "14px 20px 8px",
    fontSize: 11,
    letterSpacing: 2,
    textTransform: "uppercase",
    color: "#8B7355",
  },
  bars: { padding: "0 20px 20px", display: "flex", flexDirection: "column", gap: 10 },
  barRow: { display: "flex", alignItems: "center", gap: 10 },
  barName: { width: 110, fontSize: 13 },
  barTrack: { flex: 1, height: 8, background: "#EDE5D8", borderRadius: 4, overflow: "hidden" },
  barFill: { height: "100%", borderRadius: 4, transition: "width 0.6s ease" },
  barPct: { width: 42, fontSize: 12, color: "#6B5A3E", textAlign: "right" },
  footer: {
    textAlign: "center",
    padding: "20px",
    fontSize: 12,
    color: "#8B7355",
    borderTop: "1px solid #E0D5C5",
    letterSpacing: 0.5,
  },
};
