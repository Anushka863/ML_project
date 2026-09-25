/**
 * API Client Service for Clinical Decision Support System.
 * Connects frontend clinical intake forms to FastAPI backend /predict endpoint.
 */

const API_BASE_URL = (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE_URL) || "http://127.0.0.1:8000";

/**
 * Format raw form data into strict PatientAssessmentRequest schema.
 */
export function formatPatientPayload(patientData) {
  const age = parseFloat(patientData.age);
  const systolic = parseFloat(patientData.systolic);
  const diastolic = parseFloat(patientData.diastolic);
  const glucose = parseFloat(patientData.glucose);
  const hba1c = parseFloat(patientData.hba1c);
  const hdl = parseFloat(patientData.hdl);
  const totalCholesterol = parseFloat(patientData.totalCholesterol);
  const creatinine = parseFloat(patientData.creatinine);
  const bun = parseFloat(patientData.bun);
  
  // Calculate BMI if missing
  let bmi = parseFloat(patientData.bmi);
  if (isNaN(bmi) || bmi <= 0) {
    const heightM = parseFloat(patientData.height) / 100;
    const weightKg = parseFloat(patientData.weight);
    if (heightM > 0 && weightKg > 0) {
      bmi = parseFloat((weightKg / (heightM * heightM)).toFixed(1));
    } else {
      bmi = 24.2; // default healthy baseline
    }
  }

  return {
    age: isNaN(age) ? 50 : age,
    gender: patientData.gender || "Male",
    height: patientData.height ? parseFloat(patientData.height) : 170.0,
    weight: patientData.weight ? parseFloat(patientData.weight) : 70.0,
    bmi: bmi,
    systolic: isNaN(systolic) ? 120 : systolic,
    diastolic: isNaN(diastolic) ? 80 : diastolic,
    glucose: isNaN(glucose) ? 95 : glucose,
    hba1c: isNaN(hba1c) ? 5.4 : hba1c,
    hdl: isNaN(hdl) ? 50 : hdl,
    totalCholesterol: isNaN(totalCholesterol) ? 190 : totalCholesterol,
    creatinine: isNaN(creatinine) ? 0.9 : creatinine,
    bun: isNaN(bun) ? 14 : bun,
    waist: patientData.waist ? parseFloat(patientData.waist) : 85.0
  };
}

/**
 * Executes patient risk prediction against Clinical Multi-Disease GNN and Independent PTB-XL ECG backend.
 * @param {Object} patientData - Form state from patient intake.
 * @returns {Promise<Object>} PredictionResponse data.
 */
export async function predictPatientRisk(patientData) {
  const payload = formatPatientPayload(patientData);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      let errorMessage = `Server responded with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail.map((err) => `${err.loc?.join(".") || "field"}: ${err.msg}`).join(", ");
          } else {
            errorMessage = errorData.detail;
          }
        }
      } catch {
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Prediction API request failed:", error);
    if (error.message && error.message.includes("Failed to fetch")) {
      throw new Error("Cannot connect to backend server. Make sure FastAPI server is running on " + API_BASE_URL);
    }
    throw error;
  }
}

/**
 * Executes ODIR-5K Ophthalmic Multimodal prediction.
 * @param {Object|FormData} payload - Form JSON or FormData with images.
 * @returns {Promise<Object>} ODIRPredictionResponse data.
 */
export async function predictODIRRisk(payload) {
  try {
    const isFormData = payload instanceof FormData;
    const url = isFormData ? `${API_BASE_URL}/predict-odir/upload` : `${API_BASE_URL}/predict-odir`;
    
    const options = {
      method: "POST",
      body: isFormData ? payload : JSON.stringify(payload)
    };

    if (!isFormData) {
      options.headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
      };
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      let errorMessage = `Server responded with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail.map((err) => `${err.loc?.join(".") || "field"}: ${err.msg}`).join(", ");
          } else {
            errorMessage = errorData.detail;
          }
        }
      } catch {
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("ODIR Prediction API request failed:", error);
    if (error.message && error.message.includes("Failed to fetch")) {
      throw new Error("Cannot connect to backend server. Make sure FastAPI server is running on " + API_BASE_URL);
    }
    throw error;
  }
}

/**
 * Health check endpoint test.
 */
export async function checkBackendHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
}

/**
 * Model info endpoint query.
 */
export async function getModelInfo() {
  const response = await fetch(`${API_BASE_URL}/model-info`);
  return response.json();
}

