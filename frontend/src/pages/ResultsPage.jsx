import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { usePatient } from "../context/usePatient";

function ResultsPage() {
  const navigate = useNavigate();
  const { predictionResults } = usePatient();

  if (!predictionResults || (!predictionResults.prediction && !predictionResults.predictions)) {
    return (
      <div className="page-wrapper">
        <Navbar />
        <main className="review-container" style={{ textAlign: "center", paddingTop: "4rem" }}>
          <h2>No Prediction Results Available</h2>
          <p>Please complete a patient assessment and click 'Analyze Patient' first.</p>
          <button className="primary-button" onClick={() => navigate("/patient-assessment")}>
            Go to Patient Assessment
          </button>
        </main>
      </div>
    );
  }

  const {
    prediction = "Normal",
    probability = 0.0,
    confidence = (probability * 100).toFixed(1),
    risk_level = "Low Risk",
    model = "PTB-XL Multimodal GNN",
    clinical_explanation = [],
    graph_explanation = null,
    disclaimer = "This AI-generated result is for research/educational purposes and is not a medical diagnosis."
  } = predictionResults;

  const isAbnormal = prediction.toUpperCase() === "ABNORMAL";

  return (
    <div className="page-wrapper">
      <Navbar />

      <main className="review-container">
        <header className="page-header">
          <div className="header-badge">Phase 5 Inference Complete</div>
          <h1>AI Assessment Result</h1>
          <p className="page-subtitle">
            Multimodal Graph Neural Network evaluation of patient clinical profile and ECG representation.
          </p>
        </header>

        {/* Primary AI Assessment Banner Card */}
        <div
          className="summary-card"
          style={{
            borderLeft: `5px solid ${isAbnormal ? "#ef4444" : "#10b981"}`,
            background: isAbnormal ? "rgba(239, 68, 68, 0.04)" : "rgba(16, 185, 129, 0.04)",
            marginBottom: "1.5rem"
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
            <div>
              <span style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#64748b", fontWeight: "600" }}>
                Prediction Outcome
              </span>
              <div
                style={{
                  fontSize: "2.4rem",
                  fontWeight: "800",
                  letterSpacing: "-0.02em",
                  color: isAbnormal ? "#dc2626" : "#059669",
                  marginTop: "0.25rem"
                }}
              >
                {prediction.toUpperCase()}
              </div>
            </div>

            <div style={{ textAlign: "right" }}>
              <span style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#64748b", fontWeight: "600" }}>
                Model Confidence
              </span>
              <div
                style={{
                  fontSize: "2.4rem",
                  fontWeight: "800",
                  color: "var(--accent-primary, #2563eb)",
                  marginTop: "0.25rem"
                }}
              >
                {typeof confidence === "number" ? `${confidence.toFixed(1)}%` : `${confidence}%`}
              </div>
            </div>
          </div>

          <div
            style={{
              marginTop: "1.25rem",
              paddingTop: "1rem",
              borderTop: "1px solid rgba(0,0,0,0.08)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "0.75rem"
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontSize: "0.9rem", color: "#64748b" }}>Evaluated By:</span>
              <span
                style={{
                  fontSize: "0.85rem",
                  fontWeight: "700",
                  padding: "0.25rem 0.6rem",
                  borderRadius: "6px",
                  background: "rgba(37, 99, 235, 0.1)",
                  color: "#2563eb"
                }}
              >
                {model}
              </span>
            </div>

            <div>
              <span
                style={{
                  display: "inline-block",
                  padding: "0.3rem 0.85rem",
                  borderRadius: "9999px",
                  fontSize: "0.85rem",
                  fontWeight: "700",
                  backgroundColor: isAbnormal ? "#fee2e2" : "#dcfce7",
                  color: isAbnormal ? "#991b1b" : "#166534"
                }}
              >
                {risk_level} (Raw Probability: {(probability * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>

        {/* Clinical Feature Attributions */}
        {clinical_explanation && clinical_explanation.length > 0 && (
          <div className="summary-card" style={{ marginTop: "1.5rem" }}>
            <div className="summary-card-header">
              <span className="summary-icon">🔍</span>
              <h3>Clinical Feature Attributions</h3>
            </div>
            <p style={{ color: "#64748b", fontSize: "0.9rem", marginBottom: "1rem" }}>
              Key patient clinical measurements driving the GNN neural risk evaluation:
            </p>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {clinical_explanation.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "0.75rem 1rem",
                    background: "rgba(255,255,255,0.6)",
                    borderRadius: "8px",
                    border: "1px solid rgba(0,0,0,0.06)"
                  }}
                >
                  <div>
                    <strong style={{ textTransform: "capitalize" }}>{item.feature.replace("_", " ")}</strong>
                    <span style={{ fontSize: "0.85rem", color: "#64748b", marginLeft: "0.75rem" }}>
                      (Value: {item.value})
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: "0.85rem",
                      fontWeight: "600",
                      color: item.impact.includes("elevates") ? "#dc2626" : "#059669"
                    }}
                  >
                    {item.impact}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Patient Graph Notice */}
        {graph_explanation && (
          <div
            className="summary-card"
            style={{
              marginTop: "1.5rem",
              background: "rgba(37, 99, 235, 0.04)",
              borderColor: "rgba(37, 99, 235, 0.2)"
            }}
          >
            <div className="summary-card-header">
              <span className="summary-icon">🕸️</span>
              <h3>Multimodal Patient Graph Notice</h3>
            </div>
            <p style={{ fontSize: "0.9rem", color: "#475569", lineHeight: "1.5" }}>
              {graph_explanation.message || "Evaluated against PTB-XL multimodal GNN cosine similarity representation."}
            </p>
          </div>
        )}

        {/* Disclaimer */}
        <div
          style={{
            marginTop: "2rem",
            padding: "1rem 1.25rem",
            borderRadius: "8px",
            background: "#fffbebe6",
            border: "1px solid #fde68a",
            color: "#92400e",
            fontSize: "0.85rem",
            textAlign: "center",
            lineHeight: "1.5"
          }}
        >
          ⚠️ <strong>Medical Disclaimer:</strong> {disclaimer}
        </div>

        {/* Back and Retest Actions */}
        <div className="review-actions" style={{ marginTop: "2rem" }}>
          <button type="button" className="secondary-button" onClick={() => navigate("/patient-review")}>
            ← Back to Patient Summary
          </button>
          <button type="button" className="primary-button" onClick={() => navigate("/patient-assessment")}>
            New Assessment ↻
          </button>
        </div>
      </main>
    </div>
  );
}

export default ResultsPage;
