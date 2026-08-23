import React from "react";

function BasicInfoSection({ patientData, updateField, errors }) {
  return (
    <div className="assessment-card">
      <div className="card-header">
        <div className="card-badge">Section 1</div>
        <h3>Basic Information</h3>
        <p className="card-subtitle">Demographics and body measurement metrics</p>
      </div>

      <div className="form-grid">
        {/* Age */}
        <div className={`form-group ${errors.age ? "has-error" : ""}`}>
          <label htmlFor="age">
            Age <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="age"
              type="number"
              min="1"
              max="120"
              placeholder="e.g. 52"
              value={patientData.age}
              onChange={(e) => updateField("age", e.target.value)}
            />
            <span className="unit-badge">yrs</span>
          </div>
          {errors.age && <span className="error-message">{errors.age}</span>}
        </div>

        {/* Gender */}
        <div className={`form-group ${errors.gender ? "has-error" : ""}`}>
          <label htmlFor="gender">
            Gender <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <select
              id="gender"
              value={patientData.gender}
              onChange={(e) => updateField("gender", e.target.value)}
            >
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          {errors.gender && <span className="error-message">{errors.gender}</span>}
        </div>

        {/* Height */}
        <div className={`form-group ${errors.height ? "has-error" : ""}`}>
          <label htmlFor="height">
            Height <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="height"
              type="number"
              min="1"
              step="0.1"
              placeholder="e.g. 170"
              value={patientData.height}
              onChange={(e) => updateField("height", e.target.value)}
            />
            <span className="unit-badge">cm</span>
          </div>
          {errors.height && <span className="error-message">{errors.height}</span>}
        </div>

        {/* Weight */}
        <div className={`form-group ${errors.weight ? "has-error" : ""}`}>
          <label htmlFor="weight">
            Weight <span className="required-star">*</span>
          </label>
          <div className="input-wrapper">
            <input
              id="weight"
              type="number"
              min="1"
              step="0.1"
              placeholder="e.g. 82"
              value={patientData.weight}
              onChange={(e) => updateField("weight", e.target.value)}
            />
            <span className="unit-badge">kg</span>
          </div>
          {errors.weight && <span className="error-message">{errors.weight}</span>}
        </div>

        {/* BMI - Read Only Calculated */}
        <div className="form-group full-width-sm">
          <label htmlFor="bmi">
            Body Mass Index (BMI) <span className="readonly-tag">Calculated</span>
          </label>
          <div className="input-wrapper readonly-wrapper">
            <input
              id="bmi"
              type="text"
              readOnly
              value={patientData.bmi ? `${patientData.bmi} kg/m²` : "Auto-calculated from Height & Weight"}
              className="readonly-input"
            />
          </div>
          <span className="field-hint">Calculated automatically using weight (kg) / height (m)²</span>
        </div>
      </div>
    </div>
  );
}

export default BasicInfoSection;
