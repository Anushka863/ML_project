import React from "react";

function AdditionalInfoSection({ patientData, updateField, errors }) {
  return (
    <div className="assessment-card">
      <div className="card-header">
        <div className="card-badge">Section 4</div>
        <h3>Additional Information</h3>
        <p className="card-subtitle">Additional dataset-supported clinical measurements</p>
      </div>

      <div className="form-grid">
        {/* Waist Circumference */}
        <div className={`form-group ${errors.waist ? "has-error" : ""}`}>
          <label htmlFor="waist">
            Waist Circumference
          </label>
          <div className="input-wrapper">
            <input
              id="waist"
              type="number"
              min="0"
              step="0.1"
              placeholder="e.g. 92"
              value={patientData.waist}
              onChange={(e) => updateField("waist", e.target.value)}
            />
            <span className="unit-badge">cm</span>
          </div>
          {errors.waist ? (
            <span className="error-message">{errors.waist}</span>
          ) : (
            <span className="field-hint">Optional abdominal measurement</span>
          )}
        </div>

        {/* eGFR Note Card */}
        <div className="info-box-card">
          <div className="info-box-icon">ℹ️</div>
          <div className="info-box-content">
            <h4>Estimated GFR (eGFR)</h4>
            <p>
              eGFR is calculated directly by the Graph Neural Network pipeline using creatinine, age, and sex metrics.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AdditionalInfoSection;
