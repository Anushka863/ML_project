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
    ecg_explanation = null,
    graph_explanation = null,
    additional_clinical_info = [],
    xai = null,
    disclaimer = "This AI-generated result is for research/educational purposes and is not a medical diagnosis."
  } = predictionResults;

  const isAbnormal = prediction.toUpperCase() === "ABNORMAL";
  const ecgData = ecg_explanation || (xai && xai.ecg) || null;
  const leadAttrs = ecgData?.lead_attributions || {};
  const temporalAttrs = ecgData?.temporal_attributions || [];
  const topLeads = ecgData?.top_leads || [];

  return (
    <div className="page-wrapper">
      <Navbar />

      <main className="review-container">
        <header className="page-header">
          <div className="header-badge" style={{ background: "#ede9fe", color: "#6d28d9", borderColor: "#c4b5fd" }}>
            Phase 2 — Model-Derived XAI Active
          </div>
          <h1>AI Assessment Result</h1>
          <p className="page-subtitle">
            Multimodal Graph Neural Network evaluation with genuine Integrated Gradients explainability.
          </p>
        </header>

        {/* Primary AI Assessment Banner Card */}
        <div
          className="summary-card"
          style={{
            borderLeft: `6px solid ${isAbnormal ? "#dc2626" : "#059669"}`,
            background: isAbnormal ? "rgba(239, 68, 68, 0.04)" : "rgba(16, 185, 129, 0.04)",
            marginBottom: "1.75rem",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)"
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
            <div>
              <span style={{ fontSize: "0.8rem", textTransform: "uppercase", letterSpacing: "0.06em", color: "#64748b", fontWeight: "700" }}>
                Predicted ECG Status
              </span>
              <div
                style={{
                  fontSize: "2.4rem",
                  fontWeight: "800",
                  letterSpacing: "-0.02em",
                  color: isAbnormal ? "#dc2626" : "#059669",
                  marginTop: "0.2rem"
                }}
              >
                {prediction.toUpperCase()}
              </div>
            </div>

            <div style={{ textAlign: "right" }}>
              <span style={{ fontSize: "0.8rem", textTransform: "uppercase", letterSpacing: "0.06em", color: "#64748b", fontWeight: "700" }}>
                Model Abnormality Probability
              </span>
              <div
                style={{
                  fontSize: "2.4rem",
                  fontWeight: "800",
                  color: "var(--accent-primary, #2563eb)",
                  marginTop: "0.2rem"
                }}
              >
                {(probability * 100).toFixed(1)}%
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
              <span style={{ fontSize: "0.85rem", color: "#64748b" }}>Model Architecture:</span>
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

            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontSize: "0.85rem", color: "#64748b" }}>Confidence: {confidence}%</span>
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
                {risk_level}
              </span>
            </div>
          </div>
        </div>

        {/* SECTION 1: MODEL-DERIVED CLINICAL FEATURE ATTRIBUTIONS */}
        <div className="summary-card" style={{ marginBottom: "1.5rem" }}>
          <div className="summary-card-header">
            <span className="summary-icon">🧬</span>
            <h3>Model-Derived Clinical Feature Attributions (Integrated Gradients)</h3>
          </div>
          <p style={{ color: "#64748b", fontSize: "0.88rem", marginBottom: "1rem", lineHeight: "1.5" }}>
            Attributions computed directly from the trained neural network checkpoint via Captum Integrated Gradients
            for features consumed by the clinical encoder (<code>Age</code>, <code>Sex</code>, <code>Height</code>, <code>Weight</code>).
          </p>

          <div style={{ display: "grid", gap: "0.75rem" }}>
            {clinical_explanation.map((item, idx) => {
              const isTowardAbnormal = item.direction === "toward_abnormal" || (item.attribution && item.attribution > 0);
              const attrVal = item.attribution !== undefined ? item.attribution : item.importance;
              return (
                <div
                  key={idx}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "0.85rem 1.1rem",
                    background: "rgba(255,255,255,0.7)",
                    borderRadius: "8px",
                    border: "1px solid rgba(0,0,0,0.08)",
                    borderLeft: `4px solid ${isTowardAbnormal ? "#ef4444" : "#2563eb"}`
                  }}
                >
                  <div>
                    <strong style={{ textTransform: "capitalize", fontSize: "0.95rem" }}>{item.feature}</strong>
                    <span style={{ fontSize: "0.85rem", color: "#64748b", marginLeft: "0.75rem" }}>
                      Patient Input: <strong>{item.value}</strong>
                    </span>
                  </div>

                  <div style={{ textAlign: "right" }}>
                    <div
                      style={{
                        fontSize: "0.85rem",
                        fontWeight: "700",
                        color: isTowardAbnormal ? "#dc2626" : "#2563eb"
                      }}
                    >
                      {isTowardAbnormal ? "Pushes Toward Abnormal" : "Pushes Toward Normal"}
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "#64748b", marginTop: "0.1rem" }}>
                      Attribution: {attrVal > 0 ? `+${attrVal.toFixed(4)}` : attrVal.toFixed(4)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* SECTION 2: ECG LEAD ATTRIBUTIONS */}
        {ecgData && Object.keys(leadAttrs).length > 0 && (
          <div className="summary-card" style={{ marginBottom: "1.5rem" }}>
            <div className="summary-card-header">
              <span className="summary-icon">📈</span>
              <h3>ECG 12-Lead Model Attribution Summary</h3>
            </div>
            <p style={{ color: "#64748b", fontSize: "0.88rem", marginBottom: "1rem", lineHeight: "1.5" }}>
              Attribution magnitude across 12 ECG leads generated by the 1D CNN waveform encoder.
              Higher magnitude indicates leads that contributed most strongly to the model output.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
              {Object.entries(leadAttrs).map(([lead, val]) => (
                <div
                  key={lead}
                  style={{
                    padding: "0.6rem 0.85rem",
                    borderRadius: "6px",
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", fontWeight: "600" }}>
                    <span>Lead {lead}</span>
                    <span style={{ color: "#2563eb" }}>{val.toFixed(4)}</span>
                  </div>
                  <div style={{ height: "6px", background: "#e2e8f0", borderRadius: "3px", marginTop: "0.4rem", overflow: "hidden" }}>
                    <div
                      style={{
                        height: "100%",
                        width: `${Math.min(100, (val / (Math.max(...Object.values(leadAttrs)) + 1e-6)) * 100)}%`,
                        background: "linear-gradient(90deg, #3b82f6, #ef4444)"
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {temporalAttrs.length > 0 && (
              <div style={{ marginTop: "1rem", paddingTop: "0.75rem", borderTop: "1px solid #f1f5f9" }}>
                <span style={{ fontSize: "0.82rem", fontWeight: "700", color: "#475569", textTransform: "uppercase" }}>
                  Temporal Attribution Windows (10-Second Signal)
                </span>
                <div style={{ display: "flex", gap: "0.35rem", marginTop: "0.5rem" }}>
                  {temporalAttrs.map((seg, i) => (
                    <div
                      key={i}
                      title={`Window ${seg.time_window_seconds}: attribution ${seg.mean_attribution}`}
                      style={{
                        flex: 1,
                        height: "28px",
                        background: `rgba(239, 68, 68, ${Math.max(0.15, Math.min(1.0, seg.mean_attribution * 15))})`,
                        borderRadius: "4px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: "0.7rem",
                        fontWeight: "600",
                        color: "#0f172a"
                      }}
                    >
                      {i + 1}s
                    </div>
                  ))}
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#94a3b8", marginTop: "0.25rem" }}>
                  <span>0.0s (Start)</span>
                  <span>10.0s (End)</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* SECTION 3: PATIENT GRAPH INFERENCE STATE */}
        <div
          className="summary-card"
          style={{
            marginBottom: "1.5rem",
            background: "rgba(37, 99, 235, 0.03)",
            borderColor: "rgba(37, 99, 235, 0.2)"
          }}
        >
          <div className="summary-card-header">
            <span className="summary-icon">🕸️</span>
            <h3>Inference Graph Architecture State</h3>
          </div>
          <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
            <div>
              <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase" }}>Active Nodes</span>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#0f172a" }}>
                {graph_explanation?.nodes ?? 1} (Isolated Target)
              </div>
            </div>
            <div>
              <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase" }}>Active Edges</span>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#0f172a" }}>
                {graph_explanation?.edges ?? 0}
              </div>
            </div>
            <div>
              <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase" }}>Message Passing</span>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: graph_explanation?.message_passing ? "#059669" : "#64748b" }}>
                {graph_explanation?.message_passing ? "Active" : "None (Single Patient)"}
              </div>
            </div>
          </div>
          <p style={{ fontSize: "0.85rem", color: "#475569", lineHeight: "1.5", margin: 0 }}>
            {graph_explanation?.explanation ||
              "Single-patient inference: graph contains 1 node and 0 edges. Message passing between patients did not occur; the GNN layer acts as a direct feature projection without neighbor aggregation."}
          </p>
        </div>

        {/* SECTION 4: ADDITIONAL CLINICAL INFORMATION (NON-MODEL CONTEXT) */}
        {additional_clinical_info && additional_clinical_info.length > 0 && (
          <div className="summary-card" style={{ marginBottom: "1.5rem" }}>
            <div className="summary-card-header">
              <span className="summary-icon">📋</span>
              <h3>Additional Clinical Information (General Context)</h3>
            </div>
            <div
              style={{
                padding: "0.6rem 0.85rem",
                borderRadius: "6px",
                background: "#f0fdf4",
                border: "1px solid #bbf7d0",
                color: "#166534",
                fontSize: "0.82rem",
                marginBottom: "0.85rem"
              }}
            >
              ℹ️ <strong>Transparency Notice:</strong> The markers below are general metabolic and vital measurements from the patient record.
              They are <strong>NOT</strong> inputs to the trained PTB-XL ECG GNN neural network and did not influence the model score.
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "0.75rem" }}>
              {additional_clinical_info.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: "0.75rem 0.9rem",
                    borderRadius: "6px",
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0"
                  }}
                >
                  <div style={{ fontSize: "0.85rem", fontWeight: "600", color: "#1e293b" }}>{item.marker}</div>
                  <div style={{ fontSize: "1.1rem", fontWeight: "700", color: "#0f172a", marginTop: "0.15rem" }}>
                    {item.value} <span style={{ fontSize: "0.8rem", color: "#64748b", fontWeight: "400" }}>{item.unit}</span>
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: "0.2rem" }}>
                    Ref: {item.reference_range} · <span style={{ fontWeight: "600", color: item.clinical_status === "Optimal" || item.clinical_status === "Normal" || item.clinical_status === "Desirable" ? "#166534" : "#b45309" }}>{item.clinical_status}</span>
                  </div>
                </div>
              ))}
            </div>
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
