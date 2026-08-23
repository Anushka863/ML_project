import React from "react";

function LabInfoSection({ patientData, updateField, errors }) {
  return (
    <div className="assessment-card">
      <div className="card-header">
        <div className="card-badge">Section 3</div>
        <h3>Blood / Laboratory Information</h3>
        <p className="card-subtitle">Blood glucose, lipid profile, and renal markers</p>
      </div>

      <div className="form-grid">
        {/* Glucose */}
        <div className={`form-group ${errors.glucose ? "has-error" : ""}`}>
          <label htmlFor="glucose">
            Glucose <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="glucose"
              type="number"
              min="0"
              step="0.1"
              placeholder="e.g. 180"
              value={patientData.glucose}
              onChange={(e) => updateField("glucose", e.target.value)}
            />
            <span className="unit-badge">mg/dL</span>
          </div>
          {errors.glucose && <span className="error-message">{errors.glucose}</span>}
        </div>

        {/* HbA1c */}
        <div className={`form-group ${errors.hba1c ? "has-error" : ""}`}>
          <label htmlFor="hba1c">
            HbA1c <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="hba1c"
              type="number"
              min="0"
              max="25"
              step="0.1"
              placeholder="e.g. 7.2"
              value={patientData.hba1c}
              onChange={(e) => updateField("hba1c", e.target.value)}
            />
            <span className="unit-badge">%</span>
          </div>
          {errors.hba1c && <span className="error-message">{errors.hba1c}</span>}
        </div>

        {/* HDL Cholesterol */}
        <div className={`form-group ${errors.hdl ? "has-error" : ""}`}>
          <label htmlFor="hdl">
            HDL Cholesterol <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="hdl"
              type="number"
              min="0"
              step="0.1"
              placeholder="e.g. 45"
              value={patientData.hdl}
              onChange={(e) => updateField("hdl", e.target.value)}
            />
            <span className="unit-badge">mg/dL</span>
          </div>
          {errors.hdl && <span className="error-message">{errors.hdl}</span>}
        </div>

        {/* Total Cholesterol */}
        <div className={`form-group ${errors.totalCholesterol ? "has-error" : ""}`}>
          <label htmlFor="totalCholesterol">
            Total Cholesterol <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="totalCholesterol"
              type="number"
              min="0"
              step="0.1"
              placeholder="e.g. 220"
              value={patientData.totalCholesterol}
              onChange={(e) => updateField("totalCholesterol", e.target.value)}
            />
            <span className="unit-badge">mg/dL</span>
          </div>
          {errors.totalCholesterol && <span className="error-message">{errors.totalCholesterol}</span>}
        </div>

        {/* Creatinine */}
        <div className={`form-group ${errors.creatinine ? "has-error" : ""}`}>
          <label htmlFor="creatinine">
            Creatinine <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="creatinine"
              type="number"
              min="0"
              step="0.01"
              placeholder="e.g. 1.3"
              value={patientData.creatinine}
              onChange={(e) => updateField("creatinine", e.target.value)}
            />
            <span className="unit-badge">mg/dL</span>
          </div>
          {errors.creatinine && <span className="error-message">{errors.creatinine}</span>}
        </div>

        {/* BUN */}
        <div className={`form-group ${errors.bun ? "has-error" : ""}`}>
          <label htmlFor="bun">
            Blood Urea Nitrogen (BUN) <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="bun"
              type="number"
              min="0"
              step="0.1"
              placeholder="e.g. 18"
              value={patientData.bun}
              onChange={(e) => updateField("bun", e.target.value)}
            />
            <span className="unit-badge">mg/dL</span>
          </div>
          {errors.bun && <span className="error-message">{errors.bun}</span>}
        </div>
      </div>
    </div>
  );
}

export default LabInfoSection;
