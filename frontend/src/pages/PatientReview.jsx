import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { usePatient } from "../context/usePatient";

function PatientReview() {
  const navigate = useNavigate();
  const { patientData } = usePatient();
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);

  const handleAnalyzeClick = () => {
    setShowAnalysisModal(true);
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
            Verify patient clinical parameters before running risk prediction analysis.
          </p>
        </header>

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

        {/* Action Buttons */}
        <div className="review-actions">
          <button
            type="button"
            className="secondary-button back-btn"
            onClick={() => navigate("/patient-assessment")}
          >
            ← Edit Information
          </button>
          <button
            type="button"
            className="primary-button analyze-btn"
            onClick={handleAnalyzeClick}
          >
            Analyze Patient →
          </button>
        </div>

        {/* Analysis Modal Placeholder */}
        {showAnalysisModal && (
          <div className="modal-backdrop" onClick={() => setShowAnalysisModal(false)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-icon">🕸️</div>
              <h3>GNN Multi-Disease Risk Prediction</h3>
              <p>
                Patient clinical profile is validated and ready for backend inference.
              </p>
              <div className="modal-info-pill">
                Multi-disease targets: Diabetes • Heart Disease • Chronic Kidney Disease
              </div>
              <p className="modal-subtext">
                (Backend model endpoint integration point)
              </p>
              <button
                className="primary-button modal-close-btn"
                onClick={() => setShowAnalysisModal(false)}
              >
                Close Summary
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default PatientReview;
