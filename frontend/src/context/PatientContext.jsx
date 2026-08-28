import React, { useState } from "react";
import { PatientContext } from "./PatientContextInstance";

const initialFormData = {
  // Section 1: Basic Information
  age: "",
  gender: "",
  height: "",
  weight: "",
  bmi: "",
  
  // Section 2: Vital Signs
  systolic: "",
  diastolic: "",
  
  // Section 3: Laboratory Information
  glucose: "",
  hba1c: "",
  hdl: "",
  totalCholesterol: "",
  creatinine: "",
  bun: "",

  // Section 4: Additional Information
  waist: "",
};

export function PatientProvider({ children }) {
  const [patientData, setPatientData] = useState(initialFormData);
  const [predictionResults, setPredictionResults] = useState(null);

  const updateField = (field, value) => {
    setPatientData((prev) => {
      const next = { ...prev, [field]: value };
      
      // Auto-calculate BMI if height and weight are provided
      if (field === "height" || field === "weight") {
        const heightNum = parseFloat(field === "height" ? value : prev.height);
        const weightNum = parseFloat(field === "weight" ? value : prev.weight);
        
        if (heightNum > 0 && weightNum > 0) {
          const heightMeters = heightNum / 100;
          const calculatedBmi = (weightNum / (heightMeters * heightMeters)).toFixed(1);
          next.bmi = calculatedBmi;
        } else {
          next.bmi = "";
        }
      }
      
      return next;
    });
  };

  const resetForm = () => {
    setPatientData(initialFormData);
    setPredictionResults(null);
  };

  return (
    <PatientContext.Provider value={{
      patientData, setPatientData, updateField, resetForm,
      predictionResults, setPredictionResults
    }}>
      {children}
    </PatientContext.Provider>
  );
}

export default PatientProvider;

