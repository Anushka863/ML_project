import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { usePatient } from "../context/usePatient";

function ResultsPage() {
  const navigate = useNavigate();
  const { patientData, predictionResults } = usePatient();

  if (!predictionResults || !predictionResults.predictions) {
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

  const { predictions, clinical_explanation, graph_explanation, disclaimer } = predictionResults;

  const getRiskBadgeClass = (score) => {
    if (score >= 0.65) return "badge-high-risk";
    if (score >= 0.40) return "badge-med-risk";
    return "badge-low-risk";
  };

  return (
    <div className="page-wrapper">
      <Navbar />

      <main className="review-container">
        <header className="page-header">
          <div className="header-badge">GNN Analysis Complete</div>
          <h1>Multi-Disease Clinical Decision Report</h1>
          <p className="page-subtitle">
            Graph Neural Network (GNN) risk predictions and Explainable AI (XAI) feature attributions.
          </p>
        </header>

        {/* Risk Scores Grid */}
        <div className="summary-cards" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
          {Object.entries(predictions).map(([key, val]) => (
            <div key={key} className="summary-card">
              <div className="summary-card-header">
                <span className="summary-icon">
                  {key === "diabetes" ? "🩸" : key === "heart_disease" ? "❤️" : "🫘"}
                </span>
                <h3>{key.replace("_", " ").toUpperCase()}</h3>
              </div>
              <div style={{ marginTop: "1rem" }}>
                <div style={{ fontSize: "2.2rem", fontWeight: "700", color: "var(--accent-primary, #2563eb)" }}>
                  {(val.risk_score * 100).toFixed(1)}%
                </div>
                <div style={{ display: "inline-block", marginTop: "0.5rem", padding: "0.25rem 0.75rem", borderRadius: "9999px", fontSize: "0.85rem", fontWeight: "600", backgroundColor: val.risk_score >= 0.65 ? "#fee2e2" : val.risk_score >= 0.4 ? "#fef3c7" : "#dcfce7", color: val.risk_score >= 0.65 ? "#991b1b" : val.risk_score >= 0.4 ? "#92400e" : "#166534" }}>
                  {val.risk_level}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Clinical XAI Explanation Card */}
        {clinical_explanation && clinical_explanation.length > 0 && (
          <div className="summary-card" style={{ marginTop: "1.5rem" }}>
            <div className="summary-card-header">
              <span className="summary-icon">🔍</span>
              <h3>Clinical Feature Attributions (Why?)</h3>
            </div>
            <p style={{ color: "#64748b", fontSize: "0.9rem", marginBottom: "1rem" }}>
              Key patient clinical measurements driving the GNN neural risk prediction:
            </p>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {clinical_explanation.map((item, idx) => (
                <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem 1rem", background: "rgba(255,255,255,0.05)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.1)" }}>
                  <div>
                    <strong style={{ textTransform: "capitalize" }}>{item.feature.replace("_", " ")}</strong>
                    <span style={{ fontSize: "0.85rem", color: "#94a3b8", marginLeft: "0.75rem" }}>
                      (Value: {item.value})
                    </span>
                  </div>
                  <div style={{ fontSize: "0.85rem", fontWeight: "600", color: item.impact.includes("increases") ? "#ef4444" : "#10b981" }}>
                    {item.impact} (score: {item.importance > 0 ? `+${item.importance}` : item.importance})
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Graph Explanation Notice */}
        {graph_explanation && (
          <div className="summary-card" style={{ marginTop: "1.5rem", background: "rgba(37, 99, 235, 0.05)", borderColor: "rgba(37, 99, 235, 0.2)" }}>
            <div className="summary-card-header">
              <span className="summary-icon">🕸️</span>
              <h3>Patient Graph Representation Notice</h3>
            </div>
            <p style={{ fontSize: "0.9rem", color: "#64748b", lineHeight: "1.5" }}>
              {graph_explanation.message || "Evaluated against GNN patient similarity graph nodes."}
            </p>
          </div>
        )}

        {/* Disclaimer */}
        <div style={{ marginTop: "2rem", padding: "1rem", borderRadius: "8px", background: "#fffbebe6", border: "1px solid #fde68a", color: "#92400e", fontSize: "0.85rem", textAlign: "center" }}>
          ⚠️ <strong>Medical Disclaimer:</strong> {disclaimer}
        </div>

        {/* Back Button */}
        <div className="review-actions" style={{ marginTop: "2rem" }}>
          <button type="button" className="secondary-button" onClick={() => navigate("/patient-review")}>
            ← Back to Patient Summary
          </button>
        </div>
      </main>
    </div>
  );
}

export default ResultsPage;
