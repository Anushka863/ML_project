import React from "react";

function VitalSignsSection({ patientData, updateField, errors }) {
  return (
    <div className="assessment-card">
      <div className="card-header">
        <div className="card-badge">Section 2</div>
        <h3>Vital Signs</h3>
        <p className="card-subtitle">Blood pressure and cardiovascular parameters</p>
      </div>

      <div className="form-grid">
        {/* Systolic BP */}
        <div className={`form-group ${errors.systolic ? "has-error" : ""}`}>
          <label htmlFor="systolic">
            Systolic Blood Pressure <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="systolic"
              type="number"
              min="1"
              max="300"
              placeholder="e.g. 150"
              value={patientData.systolic}
              onChange={(e) => updateField("systolic", e.target.value)}
            />
            <span className="unit-badge">mmHg</span>
          </div>
          {errors.systolic ? (
            <span className="error-message">{errors.systolic}</span>
          ) : (
            <span className="field-hint">Systolic BP (mmHg)</span>
          )}
        </div>

        {/* Diastolic BP */}
        <div className={`form-group ${errors.diastolic ? "has-error" : ""}`}>
          <label htmlFor="diastolic">
            Diastolic Blood Pressure <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="diastolic"
              type="number"
              min="1"
              max="200"
              placeholder="e.g. 95"
              value={patientData.diastolic}
              onChange={(e) => updateField("diastolic", e.target.value)}
            />
            <span className="unit-badge">mmHg</span>
          </div>
          {errors.diastolic ? (
            <span className="error-message">{errors.diastolic}</span>
          ) : (
            <span className="field-hint">Diastolic BP (mmHg)</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default VitalSignsSection;
