import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { usePatient } from "../context/usePatient";
import BasicInfoSection from "../components/assessment/BasicInfoSection";
import VitalSignsSection from "../components/assessment/VitalSignsSection";
import LabInfoSection from "../components/assessment/LabInfoSection";
import AdditionalInfoSection from "../components/assessment/AdditionalInfoSection";

function PatientAssessment() {
  const navigate = useNavigate();
  const { patientData, updateField } = usePatient();
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};

    // Section 1: Basic Information
    if (!patientData.age || String(patientData.age).trim() === "") {
      newErrors.age = "Age is required.";
    } else if (parseFloat(patientData.age) <= 0) {
      newErrors.age = "Age must be a positive number.";
    }

    if (!patientData.gender) {
      newErrors.gender = "Gender is required.";
    }

    if (!patientData.height || String(patientData.height).trim() === "") {
      newErrors.height = "Height is required.";
    } else if (parseFloat(patientData.height) <= 0) {
      newErrors.height = "Height must be a positive number.";
    }

    if (!patientData.weight || String(patientData.weight).trim() === "") {
      newErrors.weight = "Weight is required.";
    } else if (parseFloat(patientData.weight) <= 0) {
      newErrors.weight = "Weight must be a positive number.";
    }

    // Section 2: Vital Signs
    if (!patientData.systolic || String(patientData.systolic).trim() === "") {
      newErrors.systolic = "Systolic BP is required.";
    } else if (parseFloat(patientData.systolic) <= 0) {
      newErrors.systolic = "Systolic BP must be a positive number.";
    }

    if (!patientData.diastolic || String(patientData.diastolic).trim() === "") {
      newErrors.diastolic = "Diastolic BP is required.";
    } else if (parseFloat(patientData.diastolic) <= 0) {
      newErrors.diastolic = "Diastolic BP must be a positive number.";
    }

    // Section 3: Laboratory Information
    if (!patientData.glucose || String(patientData.glucose).trim() === "") {
      newErrors.glucose = "Glucose is required.";
    } else if (parseFloat(patientData.glucose) < 0) {
      newErrors.glucose = "Glucose cannot be negative.";
    }

    if (!patientData.hba1c || String(patientData.hba1c).trim() === "") {
      newErrors.hba1c = "HbA1c is required.";
    } else if (parseFloat(patientData.hba1c) < 0) {
      newErrors.hba1c = "HbA1c cannot be negative.";
    }

    if (!patientData.hdl || String(patientData.hdl).trim() === "") {
      newErrors.hdl = "HDL Cholesterol is required.";
    } else if (parseFloat(patientData.hdl) < 0) {
      newErrors.hdl = "HDL Cholesterol cannot be negative.";
    }

    if (!patientData.totalCholesterol || String(patientData.totalCholesterol).trim() === "") {
      newErrors.totalCholesterol = "Total Cholesterol is required.";
    } else if (parseFloat(patientData.totalCholesterol) < 0) {
      newErrors.totalCholesterol = "Total Cholesterol cannot be negative.";
    }

    if (!patientData.creatinine || String(patientData.creatinine).trim() === "") {
      newErrors.creatinine = "Creatinine is required.";
    } else if (parseFloat(patientData.creatinine) < 0) {
      newErrors.creatinine = "Creatinine cannot be negative.";
    }

    if (!patientData.bun || String(patientData.bun).trim() === "") {
      newErrors.bun = "BUN is required.";
    } else if (parseFloat(patientData.bun) < 0) {
      newErrors.bun = "BUN cannot be negative.";
    }

    // Section 4: Optional waist validation if entered
    if (patientData.waist && parseFloat(patientData.waist) < 0) {
      newErrors.waist = "Waist measurement cannot be negative.";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      navigate("/patient-review");
    } else {
      window.scrollTo({ top: 100, behavior: "smooth" });
    }
  };

  return (
    <div className="page-wrapper">
      <Navbar />

      <main className="assessment-container">
        {/* Header Section */}
        <header className="page-header">
          <div className="header-badge">Clinical Intake</div>
          <h1>Patient Assessment</h1>
          <p className="page-subtitle">
            Enter the patient's clinical information to assess their health risk.
          </p>
        </header>

        {Object.keys(errors).length > 0 && (
          <div className="validation-alert" role="alert">
            <span className="alert-icon">⚠️</span>
            <div>
              <strong>Please check the form for errors</strong>
              <p>Ensure all required fields are filled out with valid numeric values before continuing.</p>
            </div>
          </div>
        )}

        {/* Assessment Form */}
        <form onSubmit={handleSubmit} className="assessment-form" noValidate>
          <BasicInfoSection
            patientData={patientData}
            updateField={updateField}
            errors={errors}
          />

          <VitalSignsSection
            patientData={patientData}
            updateField={updateField}
            errors={errors}
          />

          <LabInfoSection
            patientData={patientData}
            updateField={updateField}
            errors={errors}
          />

          <AdditionalInfoSection
            patientData={patientData}
            updateField={updateField}
            errors={errors}
          />

          {/* Bottom Action Footer */}
          <div className="form-action-footer">
            <button type="submit" className="primary-button submit-btn">
              Continue to Review →
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}

export default PatientAssessment;
