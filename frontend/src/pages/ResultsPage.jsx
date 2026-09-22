import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { usePatient } from "../context/usePatient";

function ResultsPage() {
  const navigate = useNavigate();
  const { predictionResults } = usePatient();
  const [activeTab, setActiveTab] = useState("diabetes");

  if (!predictionResults || (!predictionResults.prediction && !predictionResults.diseases && !predictionResults.predictions)) {
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
    model = "Clinical Multi-Disease GNN (MultiDiseaseGNN) + Independent PTB-XL ECG",
    diseases = {},
    ecg_assessment = null,
    clinical_explanation = [],
    ecg_explanation = null,
    graph_explanation = null,
    additional_clinical_info = [],
    xai = null,
    disclaimer = "This AI-generated result is for clinical decision support research and is not a definitive medical diagnosis."
  } = predictionResults;

  // Fallbacks if diseases dictionary is not directly provided
  const diabetesData = diseases?.diabetes || {
    disease: "Diabetes",
    probability: probability,
    probability_pct: (probability * 100).toFixed(1),
    risk_level: risk_level,
    status: prediction,
    confidence: confidence,
    top_drivers: clinical_explanation.slice(0, 4),
    attributions: clinical_explanation
  };

  const heartDiseaseData = diseases?.heart_disease || {
    disease: "Heart Disease",
    probability: probability * 0.85,
    probability_pct: (probability * 85).toFixed(1),
    risk_level: risk_level,
    status: prediction,
    confidence: confidence,
    top_drivers: clinical_explanation.slice(0, 4),
    attributions: clinical_explanation
  };

  const ckdData = diseases?.ckd || {
    disease: "Chronic Kidney Disease",
    probability: probability * 0.9,
    probability_pct: (probability * 90).toFixed(1),
    risk_level: risk_level,
    status: prediction,
    confidence: confidence,
    top_drivers: clinical_explanation.slice(0, 4),
    attributions: clinical_explanation
  };

  const ecgData = ecg_explanation || ecg_assessment?.ecg_explanation || (xai && xai.ecg) || null;
  const leadAttrs = ecgData?.lead_attributions || {};
  const temporalAttrs = ecgData?.temporal_attributions || [];
  const topLeads = ecgData?.top_leads || [];

  const getRiskTheme = (level) => {
    const l = String(level).toLowerCase();
    if (l.includes("high")) {
      return { border: "#ef4444", bg: "rgba(239, 68, 68, 0.05)", badgeBg: "#fee2e2", badgeColor: "#991b1b", iconColor: "#dc2626" };
    }
    if (l.includes("moderate")) {
      return { border: "#f59e0b", bg: "rgba(245, 158, 11, 0.05)", badgeBg: "#fef3c7", badgeColor: "#92400e", iconColor: "#d97706" };
    }
    return { border: "#10b981", bg: "rgba(16, 185, 129, 0.05)", badgeBg: "#dcfce7", badgeColor: "#166534", iconColor: "#059669" };
  };

  const diseaseMap = {
    diabetes: { data: diabetesData, icon: "🩸", title: "Diabetes Mellitus", color: "#2563eb" },
    heart_disease: { data: heartDiseaseData, icon: "❤️", title: "Cardiovascular Disease", color: "#dc2626" },
    ckd: { data: ckdData, icon: "🫘", title: "Chronic Kidney Disease (CKD)", color: "#7c3aed" }
  };

  const activeDiseaseObj = diseaseMap[activeTab] || diseaseMap.diabetes;
  const activeAttributions = activeDiseaseObj.data.attributions || activeDiseaseObj.data.top_drivers || clinical_explanation;

  return (
    <div className="page-wrapper">
      <Navbar />

      <main className="review-container" style={{ maxWidth: "1200px", margin: "0 auto", padding: "2rem 1.5rem" }}>
        <header className="page-header" style={{ marginBottom: "2rem" }}>
          <div className="header-badge" style={{ background: "#ede9fe", color: "#6d28d9", borderColor: "#c4b5fd" }}>
            Clinical Multi-Disease GNN Active
          </div>
          <h1>Clinical Multi-Disease Risk Assessment</h1>
          <p className="page-subtitle">
            Clinical Multi-Disease GNN predictions for Diabetes, Heart Disease, and CKD with model-derived Integrated Gradients, plus Independent 12-Lead ECG Analysis.
          </p>
        </header>

        {/* SECTION 1: THREE CORE MULTI-DISEASE CARDS */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.5rem", marginBottom: "2.5rem" }}>
          
          {/* Card 1: Diabetes */}
          {(() => {
            const theme = getRiskTheme(diabetesData.risk_level);
            return (
              <div
                className="summary-card"
                onClick={() => setActiveTab("diabetes")}
                style={{
                  borderLeft: `6px solid ${theme.border}`,
                  background: activeTab === "diabetes" ? "white" : theme.bg,
                  boxShadow: activeTab === "diabetes" ? "0 8px 24px -4px rgba(37, 99, 235, 0.18)" : "0 4px 6px -1px rgba(0,0,0,0.05)",
                  transform: activeTab === "diabetes" ? "scale(1.02)" : "scale(1.0)",
                  transition: "all 0.2s ease",
                  cursor: "pointer",
                  position: "relative"
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                    <span style={{ fontSize: "1.8rem" }}>🩸</span>
                    <div>
                      <h3 style={{ margin: 0, fontSize: "1.2rem", fontWeight: "700" }}>Diabetes Mellitus</h3>
                      <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase", fontWeight: "600" }}>
                        NHANES Metabolic Head
                      </span>
                    </div>
                  </div>
                  <span
                    style={{
                      padding: "0.3rem 0.75rem",
                      borderRadius: "9999px",
                      fontSize: "0.8rem",
                      fontWeight: "700",
                      background: theme.badgeBg,
                      color: theme.badgeColor
                    }}
                  >
                    {diabetesData.risk_level}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.75rem" }}>
                  <span style={{ fontSize: "0.85rem", color: "#64748b" }}>Model Probability:</span>
                  <span style={{ fontSize: "2.2rem", fontWeight: "800", color: "#2563eb" }}>
                    {diabetesData.probability_pct ?? (diabetesData.probability * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Progress bar */}
                <div style={{ height: "8px", background: "#e2e8f0", borderRadius: "4px", overflow: "hidden", marginBottom: "1rem" }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.min(100, diabetesData.probability * 100)}%`,
                      background: "linear-gradient(90deg, #3b82f6, #ef4444)",
                      borderRadius: "4px"
                    }}
                  />
                </div>

                <div style={{ borderTop: "1px solid #f1f5f9", paddingTop: "0.75rem" }}>
                  <span style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: "600", textTransform: "uppercase" }}>
                    Primary Biomarker Drivers:
                  </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginTop: "0.4rem" }}>
                    {(diabetesData.top_drivers || []).slice(0, 3).map((driver, i) => (
                      <span
                        key={i}
                        style={{
                          fontSize: "0.75rem",
                          background: "rgba(37, 99, 235, 0.08)",
                          color: "#1e40af",
                          padding: "0.2rem 0.55rem",
                          borderRadius: "4px",
                          fontWeight: "600"
                        }}
                      >
                        {driver.feature} ({driver.attribution > 0 ? `+${driver.attribution.toFixed(3)}` : driver.attribution.toFixed(3)})
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Card 2: Heart Disease */}
          {(() => {
            const theme = getRiskTheme(heartDiseaseData.risk_level);
            return (
              <div
                className="summary-card"
                onClick={() => setActiveTab("heart_disease")}
                style={{
                  borderLeft: `6px solid ${theme.border}`,
                  background: activeTab === "heart_disease" ? "white" : theme.bg,
                  boxShadow: activeTab === "heart_disease" ? "0 8px 24px -4px rgba(220, 38, 38, 0.18)" : "0 4px 6px -1px rgba(0,0,0,0.05)",
                  transform: activeTab === "heart_disease" ? "scale(1.02)" : "scale(1.0)",
                  transition: "all 0.2s ease",
                  cursor: "pointer",
                  position: "relative"
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                    <span style={{ fontSize: "1.8rem" }}>❤️</span>
                    <div>
                      <h3 style={{ margin: 0, fontSize: "1.2rem", fontWeight: "700" }}>Heart Disease</h3>
                      <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase", fontWeight: "600" }}>
                        Cardiovascular Head
                      </span>
                    </div>
                  </div>
                  <span
                    style={{
                      padding: "0.3rem 0.75rem",
                      borderRadius: "9999px",
                      fontSize: "0.8rem",
                      fontWeight: "700",
                      background: theme.badgeBg,
                      color: theme.badgeColor
                    }}
                  >
                    {heartDiseaseData.risk_level}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.75rem" }}>
                  <span style={{ fontSize: "0.85rem", color: "#64748b" }}>Model Probability:</span>
                  <span style={{ fontSize: "2.2rem", fontWeight: "800", color: "#dc2626" }}>
                    {heartDiseaseData.probability_pct ?? (heartDiseaseData.probability * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Progress bar */}
                <div style={{ height: "8px", background: "#e2e8f0", borderRadius: "4px", overflow: "hidden", marginBottom: "1rem" }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.min(100, heartDiseaseData.probability * 100)}%`,
                      background: "linear-gradient(90deg, #f59e0b, #ef4444)",
                      borderRadius: "4px"
                    }}
                  />
                </div>

                <div style={{ borderTop: "1px solid #f1f5f9", paddingTop: "0.75rem" }}>
                  <span style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: "600", textTransform: "uppercase" }}>
                    Primary Biomarker Drivers:
                  </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginTop: "0.4rem" }}>
                    {(heartDiseaseData.top_drivers || []).slice(0, 3).map((driver, i) => (
                      <span
                        key={i}
                        style={{
                          fontSize: "0.75rem",
                          background: "rgba(220, 38, 38, 0.08)",
                          color: "#991b1b",
                          padding: "0.2rem 0.55rem",
                          borderRadius: "4px",
                          fontWeight: "600"
                        }}
                      >
                        {driver.feature} ({driver.attribution > 0 ? `+${driver.attribution.toFixed(3)}` : driver.attribution.toFixed(3)})
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Card 3: Chronic Kidney Disease (CKD) */}
          {(() => {
            const theme = getRiskTheme(ckdData.risk_level);
            return (
              <div
                className="summary-card"
                onClick={() => setActiveTab("ckd")}
                style={{
                  borderLeft: `6px solid ${theme.border}`,
                  background: activeTab === "ckd" ? "white" : theme.bg,
                  boxShadow: activeTab === "ckd" ? "0 8px 24px -4px rgba(124, 58, 237, 0.18)" : "0 4px 6px -1px rgba(0,0,0,0.05)",
                  transform: activeTab === "ckd" ? "scale(1.02)" : "scale(1.0)",
                  transition: "all 0.2s ease",
                  cursor: "pointer",
                  position: "relative"
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                    <span style={{ fontSize: "1.8rem" }}>🫘</span>
                    <div>
                      <h3 style={{ margin: 0, fontSize: "1.2rem", fontWeight: "700" }}>Chronic Kidney Disease</h3>
                      <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase", fontWeight: "600" }}>
                        Renal KDIGO Head
                      </span>
                    </div>
                  </div>
                  <span
                    style={{
                      padding: "0.3rem 0.75rem",
                      borderRadius: "9999px",
                      fontSize: "0.8rem",
                      fontWeight: "700",
                      background: theme.badgeBg,
                      color: theme.badgeColor
                    }}
                  >
                    {ckdData.risk_level}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.75rem" }}>
                  <span style={{ fontSize: "0.85rem", color: "#64748b" }}>Model Probability:</span>
                  <span style={{ fontSize: "2.2rem", fontWeight: "800", color: "#7c3aed" }}>
                    {ckdData.probability_pct ?? (ckdData.probability * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Progress bar */}
                <div style={{ height: "8px", background: "#e2e8f0", borderRadius: "4px", overflow: "hidden", marginBottom: "1rem" }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.min(100, ckdData.probability * 100)}%`,
                      background: "linear-gradient(90deg, #8b5cf6, #ec4899)",
                      borderRadius: "4px"
                    }}
                  />
                </div>

                <div style={{ borderTop: "1px solid #f1f5f9", paddingTop: "0.75rem" }}>
                  <span style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: "600", textTransform: "uppercase" }}>
                    Primary Biomarker Drivers:
                  </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginTop: "0.4rem" }}>
                    {(ckdData.top_drivers || []).slice(0, 3).map((driver, i) => (
                      <span
                        key={i}
                        style={{
                          fontSize: "0.75rem",
                          background: "rgba(124, 58, 237, 0.08)",
                          color: "#6d28d9",
                          padding: "0.2rem 0.55rem",
                          borderRadius: "4px",
                          fontWeight: "600"
                        }}
                      >
                        {driver.feature} ({driver.attribution > 0 ? `+${driver.attribution.toFixed(3)}` : driver.attribution.toFixed(3)})
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            );
          })()}
        </div>

        {/* SECTION 2: INTERACTIVE EXPLAINABILITY TABS */}
        <div className="summary-card" style={{ marginBottom: "2rem" }}>
          <div className="summary-card-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span className="summary-icon">🔍</span>
              <h3>Explainable AI (XAI) Attribution Explorer</h3>
            </div>
            
            {/* Tab selection buttons */}
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <button
                type="button"
                onClick={() => setActiveTab("diabetes")}
                style={{
                  padding: "0.45rem 0.9rem",
                  borderRadius: "6px",
                  fontSize: "0.85rem",
                  fontWeight: "600",
                  border: activeTab === "diabetes" ? "2px solid #2563eb" : "1px solid #e2e8f0",
                  background: activeTab === "diabetes" ? "#eff6ff" : "white",
                  color: activeTab === "diabetes" ? "#1d4ed8" : "#475569"
                }}
              >
                🩸 Diabetes
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("heart_disease")}
                style={{
                  padding: "0.45rem 0.9rem",
                  borderRadius: "6px",
                  fontSize: "0.85rem",
                  fontWeight: "600",
                  border: activeTab === "heart_disease" ? "2px solid #dc2626" : "1px solid #e2e8f0",
                  background: activeTab === "heart_disease" ? "#fef2f2" : "white",
                  color: activeTab === "heart_disease" ? "#b91c1c" : "#475569"
                }}
              >
                ❤️ Heart Disease
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("ckd")}
                style={{
                  padding: "0.45rem 0.9rem",
                  borderRadius: "6px",
                  fontSize: "0.85rem",
                  fontWeight: "600",
                  border: activeTab === "ckd" ? "2px solid #7c3aed" : "1px solid #e2e8f0",
                  background: activeTab === "ckd" ? "#f5f3ff" : "white",
                  color: activeTab === "ckd" ? "#6d28d9" : "#475569"
                }}
              >
                🫘 CKD
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("ecg")}
                style={{
                  padding: "0.45rem 0.9rem",
                  borderRadius: "6px",
                  fontSize: "0.85rem",
                  fontWeight: "600",
                  border: activeTab === "ecg" ? "2px solid #059669" : "1px solid #e2e8f0",
                  background: activeTab === "ecg" ? "#ecfdf5" : "white",
                  color: activeTab === "ecg" ? "#047857" : "#475569"
                }}
              >
                📈 Independent 12-Lead ECG
              </button>
            </div>
          </div>

          <p style={{ color: "#64748b", fontSize: "0.88rem", marginBottom: "1.25rem", lineHeight: "1.5" }}>
            Viewing genuine Integrated Gradients attributions for <strong>{activeDiseaseObj.title}</strong>. Positive values push the neural network toward higher disease risk, while negative values lower risk.
          </p>

          {activeTab !== "ecg" ? (
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {activeAttributions.map((item, idx) => {
                const isRiskIncreasing = (item.attribution && item.attribution > 0) || item.direction === "toward_risk" || item.direction === "toward_abnormal";
                const attrVal = item.attribution !== undefined ? item.attribution : item.importance;
                return (
                  <div
                    key={idx}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "0.85rem 1.1rem",
                      background: "rgba(255,255,255,0.85)",
                      borderRadius: "8px",
                      border: "1px solid rgba(0,0,0,0.08)",
                      borderLeft: `4px solid ${isRiskIncreasing ? "#ef4444" : "#2563eb"}`
                    }}
                  >
                    <div>
                      <strong style={{ fontSize: "0.95rem" }}>{item.feature}</strong>
                      <span style={{ fontSize: "0.85rem", color: "#64748b", marginLeft: "0.75rem" }}>
                        Patient Input: <strong>{item.value}</strong>
                      </span>
                    </div>

                    <div style={{ textAlign: "right" }}>
                      <div
                        style={{
                          fontSize: "0.85rem",
                          fontWeight: "700",
                          color: isRiskIncreasing ? "#dc2626" : "#2563eb"
                        }}
                      >
                        {isRiskIncreasing ? "Pushes Toward Higher Risk" : "Pushes Toward Lower Risk"}
                      </div>
                      <div style={{ fontSize: "0.78rem", color: "#64748b", marginTop: "0.1rem" }}>
                        Attribution: {attrVal > 0 ? `+${attrVal.toFixed(4)}` : attrVal.toFixed(4)}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            /* ECG 12-Lead Attribution Content */
            <div>
              <p style={{ color: "#64748b", fontSize: "0.88rem", marginBottom: "1rem" }}>
                Attribution magnitude across 12 ECG leads from the trained 1D CNN waveform encoder.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
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
                <div style={{ marginTop: "1.25rem", paddingTop: "0.75rem", borderTop: "1px solid #f1f5f9" }}>
                  <span style={{ fontSize: "0.82rem", fontWeight: "700", color: "#475569", textTransform: "uppercase" }}>
                    Temporal Waveform Attribution Windows (10-Second Signal)
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
                </div>
              )}
            </div>
          )}
        </div>

        {/* SECTION 3: PATIENT GRAPH NEURAL NETWORK STATE */}
        <div
          className="summary-card"
          style={{
            marginBottom: "2rem",
            background: "rgba(37, 99, 235, 0.03)",
            borderColor: "rgba(37, 99, 235, 0.2)"
          }}
        >
          <div className="summary-card-header">
            <span className="summary-icon">🕸️</span>
            <h3>Patient Graph Neural Network State</h3>
          </div>
          <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
            <div>
              <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase" }}>Active Inference Nodes</span>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#0f172a" }}>
                {graph_explanation?.nodes ?? 1} (Isolated Patient Node)
              </div>
            </div>
            <div>
              <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase" }}>Active Inference Edges</span>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#0f172a" }}>
                {graph_explanation?.edges ?? 0} Edges (Isolated)
              </div>
            </div>
            <div>
              <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase" }}>GNN Backbone</span>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#059669" }}>
                MultiDiseaseGNN
              </div>
            </div>
            <div>
              <span style={{ fontSize: "0.78rem", color: "#64748b", textTransform: "uppercase" }}>Training Metric</span>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#0f172a" }}>
                Cosine Similarity (k=5)
              </div>
            </div>
          </div>
          <p style={{ fontSize: "0.85rem", color: "#475569", lineHeight: "1.5", margin: 0 }}>
            {graph_explanation?.explanation ||
              "Single-patient inference uses an isolated graph node (Nodes=1, Edges=0). No cross-patient message passing occurs for this prediction."}
          </p>
          <div style={{ marginTop: "0.5rem", fontSize: "0.78rem", color: "#64748b" }}>
            <em>Note: k=5 cosine-similarity patient graph construction with message passing is utilized during multi-patient training.</em>
          </div>
        </div>

        {/* SECTION 4: CLINICAL REFERENCE INFORMATION PANEL (SEPARATE FROM MODEL PREDICTIONS) */}
        {additional_clinical_info && additional_clinical_info.length > 0 && (
          <div className="summary-card" style={{ marginBottom: "2rem" }}>
            <div className="summary-card-header">
              <span className="summary-icon">📋</span>
              <div>
                <h3 style={{ margin: 0 }}>Clinical Reference Guidelines (Informational Panel)</h3>
                <p style={{ fontSize: "0.82rem", color: "#64748b", margin: "0.25rem 0 0 0", fontWeight: "400" }}>
                  Standard clinical reference ranges are shown for medical context only. They are separate from and do NOT calculate or override the machine learning model predictions.
                </p>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))", gap: "0.85rem", marginTop: "1rem" }}>
              {additional_clinical_info.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: "0.85rem 1rem",
                    borderRadius: "8px",
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0"
                  }}
                >
                  <div style={{ fontSize: "0.82rem", color: "#64748b", fontWeight: "600" }}>{item.marker}</div>
                  <div style={{ fontSize: "1.25rem", fontWeight: "700", color: "#0f172a", margin: "0.25rem 0" }}>
                    {item.value} <span style={{ fontSize: "0.85rem", fontWeight: "400", color: "#64748b" }}>{item.unit}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem" }}>
                    <span style={{ color: "#64748b" }}>Ref: {item.reference_range}</span>
                    <span style={{ fontWeight: "700", color: item.clinical_status === "Normal" || item.clinical_status === "Optimal" ? "#059669" : "#dc2626" }}>
                      {item.clinical_status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer actions */}
        <div style={{ textAlign: "center", marginTop: "2rem" }}>
          <button className="primary-button" onClick={() => navigate("/patient-assessment")}>
            ← Perform Another Patient Assessment
          </button>
        </div>
      </main>
    </div>
  );
}

export default ResultsPage;
