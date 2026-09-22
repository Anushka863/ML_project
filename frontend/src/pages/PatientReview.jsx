import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { usePatient } from "../context/usePatient";
import { predictPatientRisk } from "../services/api";

function PatientReview() {
  const navigate = useNavigate();
  const context = usePatient() || {};
  const patientData = context.patientData || {};
  const setPatientData = context.setPatientData || (() => {});
  const setPredictionResults = context.setPredictionResults || (() => {});
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const hasAnyData = Boolean(patientData.age || patientData.gender || patientData.systolic || patientData.glucose);

  const fillSampleDataAndAnalyze = () => {
    const sample = {
      age: "58",
      gender: "Male",
      height: "175",
      weight: "82",
      bmi: "26.8",
      systolic: "138",
      diastolic: "88",
      glucose: "126",
      hba1c: "6.8",
      hdl: "42",
      totalCholesterol: "215",
      creatinine: "1.2",
      bun: "18",
      waist: "96"
    };
    setPatientData(sample);
  };

  const handleAnalyzeClick = async () => {
    if (loading) return; // Prevent duplicate submissions
    setLoading(true);
    setErrorMsg("");

    try {
      // Execute prediction via clean centralized API service
      const data = await predictPatientRisk(patientData);
      setPredictionResults(data);
      setLoading(false);
      navigate("/results");
    } catch (err) {
      console.error("Prediction API failed:", err);
      setErrorMsg(err.message || "Failed to connect to backend prediction endpoint.");
      setLoading(false);
    }
  };

  return (
    <div className="page-wrapper">
      <Navbar />

      <main className="review-container">
        {/* Header */}
        <header className="page-header">
          <div className="header-badge">Clinical Review</div>
          <h1>Patient Summary</h1>
          <p className="page-subtitle">
            Verify patient clinical parameters before running Clinical Multi-Disease GNN risk assessment.
          </p>
        </header>

        {!hasAnyData && (
          <div className="validation-alert" style={{ marginBottom: "2rem", background: "rgba(37, 99, 235, 0.05)", borderColor: "rgba(37, 99, 235, 0.2)", color: "#1e40af" }}>
            <span className="alert-icon">ℹ️</span>
            <div>
              <strong>No Patient Parameters Entered Yet</strong>
              <p>You opened the review page directly. Complete the clinical assessment or click below to load sample patient data.</p>
              <div style={{ marginTop: "1rem", display: "flex", gap: "1rem" }}>
                <button type="button" className="secondary-button" onClick={() => navigate("/patient-assessment")}>
                  ← Go to Assessment Form
                </button>
                <button type="button" className="primary-button" onClick={fillSampleDataAndAnalyze}>
                  ⚡ Load Sample Patient Data
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Summary Content Cards */}
        <div className="summary-cards">
          {/* Basic Information Card */}
          <div className="summary-card">
            <div className="summary-card-header">
              <span className="summary-icon">👤</span>
              <h3>Basic Information</h3>
            </div>
            <div className="summary-details-grid">
              <div className="summary-item">
                <span className="summary-label">Age</span>
                <span className="summary-value">{patientData.age ? `${patientData.age} yrs` : "—"}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Gender</span>
                <span className="summary-value">{patientData.gender || "—"}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Height</span>
                <span className="summary-value">{patientData.height ? `${patientData.height} cm` : "—"}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Weight</span>
                <span className="summary-value">{patientData.weight ? `${patientData.weight} kg` : "—"}</span>
              </div>
              <div className="summary-item highlight-item">
                <span className="summary-label">BMI</span>
                <span className="summary-value badge-value">
                  {patientData.bmi ? `${patientData.bmi} kg/m²` : "—"}
                </span>
              </div>
              {patientData.waist && (
                <div className="summary-item">
                  <span className="summary-label">Waist Circumference</span>
                  <span className="summary-value">{patientData.waist} cm</span>
                </div>
              )}
            </div>
          </div>

          {/* Vital Signs Card */}
          <div className="summary-card">
            <div className="summary-card-header">
              <span className="summary-icon">🫀</span>
              <h3>Vital Signs</h3>
            </div>
            <div className="summary-details-grid">
              <div className="summary-item highlight-item full-width-sm">
                <span className="summary-label">Blood Pressure</span>
                <span className="summary-value">
                  {patientData.systolic && patientData.diastolic
                    ? `${patientData.systolic} / ${patientData.diastolic} mmHg`
                    : "—"}
                </span>
              </div>
            </div>
          </div>

          {/* Laboratory Information Card */}
          <div className="summary-card">
            <div className="summary-card-header">
              <span className="summary-icon">🧪</span>
              <h3>Laboratory Information</h3>
            </div>
            <div className="summary-details-grid">
              <div className="summary-item">
                <span className="summary-label">Glucose</span>
                <span className="summary-value">{patientData.glucose ? `${patientData.glucose} mg/dL` : "—"}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">HbA1c</span>
                <span className="summary-value">{patientData.hba1c ? `${patientData.hba1c} %` : "—"}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">HDL Cholesterol</span>
                <span className="summary-value">{patientData.hdl ? `${patientData.hdl} mg/dL` : "—"}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Total Cholesterol</span>
                <span className="summary-value">{patientData.totalCholesterol ? `${patientData.totalCholesterol} mg/dL` : "—"}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Creatinine</span>
                <span className="summary-value">{patientData.creatinine ? `${patientData.creatinine} mg/dL` : "—"}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">BUN</span>
                <span className="summary-value">{patientData.bun ? `${patientData.bun} mg/dL` : "—"}</span>
              </div>
            </div>
          </div>
        </div>

        {errorMsg && (
          <div className="validation-alert" style={{ marginBottom: "1.5rem", background: "#fef2f2", borderColor: "#fca5a5", color: "#991b1b" }}>
            <span className="alert-icon">⚠️</span>
            <div>
              <strong>Inference Error</strong>
              <p>{errorMsg}</p>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="review-actions">
          <button
            type="button"
            className="secondary-button back-btn"
            onClick={() => navigate("/patient-assessment")}
            disabled={loading}
          >
            ← Edit Information
          </button>
          <button
            type="button"
            className="primary-button analyze-btn"
            onClick={handleAnalyzeClick}
            disabled={loading}
          >
            {loading ? "Analyzing Patient GNN..." : "Analyze Patient →"}
          </button>
        </div>
      </main>
    </div>
  );
}

export default PatientReview;
