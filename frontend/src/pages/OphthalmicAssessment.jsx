import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { predictODIRRisk } from "../services/api";

const ODIR_DISEASE_COLORS = {
  N: "#10b981", // Green
  D: "#ef4444", // Red
  G: "#f59e0b", // Amber
  C: "#6366f1", // Indigo
  A: "#ec4899", // Pink
  H: "#f97316", // Orange
  M: "#8b5cf6", // Purple
  O: "#64748b"  // Slate
};

export default function OphthalmicAssessment() {
  const navigate = useNavigate();

  // Form State
  const [age, setAge] = useState("58");
  const [sex, setSex] = useState("Male");
  const [leftImageFile, setLeftImageFile] = useState(null);
  const [rightImageFile, setRightImageFile] = useState(null);
  const [leftPreview, setLeftPreview] = useState(null);
  const [rightPreview, setRightPreview] = useState(null);

  // Status & Results
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  // File Handlers
  const handleLeftImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Left eye file must be a valid image (.jpg, .png, .jpeg).");
        return;
      }
      setLeftImageFile(file);
      setLeftPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleRightImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Right eye file must be a valid image (.jpg, .png, .jpeg).");
        return;
      }
      setRightImageFile(file);
      setRightPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  // Submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!leftImageFile && !rightImageFile) {
      setError("Please upload both the left-eye and right-eye fundus images.");
      return;
    }
    if (!leftImageFile) {
      setError("Please upload the left-eye fundus image.");
      return;
    }
    if (!rightImageFile) {
      setError("Please upload the right-eye fundus image.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("age", age);
      formData.append("sex", sex);
      formData.append("left_image", leftImageFile);
      formData.append("right_image", rightImageFile);

      const response = await predictODIRRisk(formData);
      setResults(response);
    } catch (err) {
      setError(err.message || "Ophthalmic assessment failed. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const resetAssessment = () => {
    setResults(null);
    setLeftImageFile(null);
    setRightImageFile(null);
    setLeftPreview(null);
    setRightPreview(null);
    setError(null);
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return "0 KB";
    const kb = bytes / 1024;
    return kb < 1000 ? `${kb.toFixed(1)} KB` : `${(kb / 1024).toFixed(2)} MB`;
  };

  return (
    <div className="assessment-page">
      <Navbar />

      <main className="assessment-container" style={{ maxWidth: "1100px", margin: "0 auto", padding: "2rem 1.5rem" }}>
        {/* Page Header */}
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <div style={{ display: "inline-block", background: "#e0f2fe", color: "#0369a1", padding: "4px 12px", borderRadius: "9999px", fontSize: "0.85rem", fontWeight: "600", marginBottom: "0.5rem" }}>
            BRANCH 3 — INDEPENDENT OPHTHALMIC MULTIMODAL MODEL
          </div>
          <h1 style={{ fontSize: "2.2rem", fontWeight: "800", color: "#0f172a", margin: "0.5rem 0" }}>
            ODIR-5K Retinal Fundus Multi-Disease Assessment
          </h1>
          <p style={{ color: "#64748b", maxWidth: "780px", margin: "0 auto", fontSize: "0.95rem", lineHeight: "1.5" }}>
            Deep multimodal analysis combining bilateral fundus photography (Left + Right eyes) with patient demographics. The GNN is trained using a patient similarity graph. During single-patient web inference, the submitted patient is processed as an isolated graph node.
          </p>
        </div>

        {/* Input Form Card */}
        {!results && (
          <form onSubmit={handleSubmit} style={{ background: "#ffffff", borderRadius: "1rem", padding: "2rem", boxShadow: "0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.01)", border: "1px solid #e2e8f0" }}>
            {error && (
              <div style={{ background: "#fee2e2", border: "1px solid #f87171", color: "#991b1b", padding: "1rem", borderRadius: "0.5rem", marginBottom: "1.5rem", fontSize: "0.9rem" }}>
                ⚠️ {error}
              </div>
            )}

            {/* Demographics Section */}
            <div style={{ marginBottom: "2rem" }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: "700", color: "#1e293b", marginBottom: "1rem", borderBottom: "1px solid #f1f5f9", paddingBottom: "0.5rem" }}>
                1. Patient Demographics
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
                <div>
                  <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "600", color: "#475569", marginBottom: "0.5rem" }}>
                    Patient Age (Years) *
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="120"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    required
                    style={{ width: "100%", padding: "0.65rem 0.85rem", border: "1px solid #cbd5e1", borderRadius: "0.5rem", fontSize: "0.95rem" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "600", color: "#475569", marginBottom: "0.5rem" }}>
                    Patient Biological Sex *
                  </label>
                  <select
                    value={sex}
                    onChange={(e) => setSex(e.target.value)}
                    style={{ width: "100%", padding: "0.65rem 0.85rem", border: "1px solid #cbd5e1", borderRadius: "0.5rem", fontSize: "0.95rem", background: "#fff" }}
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Bilateral Fundus Images Section */}
            <div style={{ marginBottom: "2rem" }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: "700", color: "#1e293b", marginBottom: "0.5rem", borderBottom: "1px solid #f1f5f9", paddingBottom: "0.5rem" }}>
                2. Bilateral Retinal Fundus Images (Both Eyes Required)
              </h3>
              <p style={{ fontSize: "0.85rem", color: "#64748b", marginBottom: "1.25rem" }}>
                Upload the left and right retinal fundus photographs used for this patient assessment.
              </p>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
                {/* Left Eye Upload */}
                <div style={{ border: leftImageFile ? "2px solid #0284c7" : "2px dashed #cbd5e1", borderRadius: "0.75rem", padding: "1.5rem", textAlign: "center", background: "#f8fafc" }}>
                  <h4 style={{ margin: "0 0 0.5rem 0", color: "#334155" }}>👁️ Left Eye Fundus (OS) *</h4>
                  {leftPreview && leftImageFile ? (
                    <div>
                      <img src={leftPreview} alt="Left Fundus Preview" style={{ width: "160px", height: "160px", objectFit: "cover", borderRadius: "0.5rem", margin: "0.5rem auto", boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)" }} />
                      <div style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "#1e293b", fontWeight: "600" }}>
                        📄 {leftImageFile.name}
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "#64748b", marginBottom: "0.5rem" }}>
                        Size: {formatFileSize(leftImageFile.size)}
                      </div>
                      <div>
                        <button type="button" onClick={() => { setLeftImageFile(null); setLeftPreview(null); }} style={{ fontSize: "0.8rem", color: "#ef4444", background: "none", border: "none", cursor: "pointer", textDecoration: "underline" }}>
                          Remove Image
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div style={{ color: "#94a3b8", fontSize: "0.85rem", fontWeight: "500", margin: "0.5rem 0" }}>
                        No left-eye image selected
                      </div>
                      <p style={{ fontSize: "0.85rem", color: "#64748b", margin: "0.25rem 0 1rem 0" }}>Upload Left Retinal Image (.jpg, .png)</p>
                      <input type="file" accept="image/*" onChange={handleLeftImageChange} style={{ fontSize: "0.85rem" }} />
                    </div>
                  )}
                </div>

                {/* Right Eye Upload */}
                <div style={{ border: rightImageFile ? "2px solid #0284c7" : "2px dashed #cbd5e1", borderRadius: "0.75rem", padding: "1.5rem", textAlign: "center", background: "#f8fafc" }}>
                  <h4 style={{ margin: "0 0 0.5rem 0", color: "#334155" }}>👁️ Right Eye Fundus (OD) *</h4>
                  {rightPreview && rightImageFile ? (
                    <div>
                      <img src={rightPreview} alt="Right Fundus Preview" style={{ width: "160px", height: "160px", objectFit: "cover", borderRadius: "0.5rem", margin: "0.5rem auto", boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)" }} />
                      <div style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "#1e293b", fontWeight: "600" }}>
                        📄 {rightImageFile.name}
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "#64748b", marginBottom: "0.5rem" }}>
                        Size: {formatFileSize(rightImageFile.size)}
                      </div>
                      <div>
                        <button type="button" onClick={() => { setRightImageFile(null); setRightPreview(null); }} style={{ fontSize: "0.8rem", color: "#ef4444", background: "none", border: "none", cursor: "pointer", textDecoration: "underline" }}>
                          Remove Image
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div style={{ color: "#94a3b8", fontSize: "0.85rem", fontWeight: "500", margin: "0.5rem 0" }}>
                        No right-eye image selected
                      </div>
                      <p style={{ fontSize: "0.85rem", color: "#64748b", margin: "0.25rem 0 1rem 0" }}>Upload Right Retinal Image (.jpg, .png)</p>
                      <input type="file" accept="image/*" onChange={handleRightImageChange} style={{ fontSize: "0.85rem" }} />
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ textAlign: "center", marginTop: "2rem" }}>
              <button
                type="submit"
                disabled={loading || !leftImageFile || !rightImageFile}
                style={{
                  background: loading || !leftImageFile || !rightImageFile ? "#94a3b8" : "linear-gradient(135deg, #0284c7, #0369a1)",
                  color: "#ffffff",
                  padding: "0.85rem 2.5rem",
                  fontSize: "1.05rem",
                  fontWeight: "700",
                  border: "none",
                  borderRadius: "0.5rem",
                  cursor: loading || !leftImageFile || !rightImageFile ? "not-allowed" : "pointer",
                  boxShadow: "0 4px 12px rgba(2, 132, 199, 0.25)"
                }}
              >
                {loading ? "Analyzing Bilateral Fundus GNN..." : "Run Ophthalmic Assessment"}
              </button>
            </div>
          </form>
        )}

        {/* Results View */}
        {results && (
          <div>
            {/* Primary Finding Banner */}
            <div style={{ background: "linear-gradient(135deg, #0f172a, #1e293b)", color: "#ffffff", borderRadius: "1rem", padding: "1.75rem 2rem", marginBottom: "2rem", boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1)" }}>
              <div style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#38bdf8", fontWeight: "700", marginBottom: "0.35rem" }}>
                {results.branch}
              </div>
              <h2 style={{ fontSize: "1.6rem", fontWeight: "800", margin: "0 0 0.5rem 0" }}>
                {results.prediction}
              </h2>
              <p style={{ fontSize: "0.9rem", color: "#cbd5e1", margin: 0 }}>
                Model: <code>{results.model_name}</code>
              </p>
            </div>

            {/* 8 ODIR Multi-Label Targets Grid */}
            <h3 style={{ fontSize: "1.3rem", fontWeight: "800", color: "#0f172a", marginBottom: "1rem" }}>
              ODIR Ophthalmic Model — 8 Multi-Disease Probabilities
            </h3>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1.25rem", marginBottom: "2.5rem" }}>
              {Object.values(results.diseases).map((disease) => {
                const color = ODIR_DISEASE_COLORS[disease.code] || "#0284c7";
                const isHigh = disease.risk_level.includes("High");

                return (
                  <div
                    key={disease.code}
                    style={{
                      background: "#ffffff",
                      borderRadius: "0.75rem",
                      padding: "1.25rem",
                      border: isHigh ? `2px solid ${color}` : "1px solid #e2e8f0",
                      boxShadow: "0 4px 6px -1px rgba(0,0,0,0.05)"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                      <span style={{ fontSize: "0.85rem", fontWeight: "700", color: color, background: `${color}15`, padding: "2px 8px", borderRadius: "4px" }}>
                        [{disease.code}]
                      </span>
                      <span style={{ fontSize: "0.75rem", fontWeight: "600", color: isHigh ? "#ef4444" : "#64748b" }}>
                        {disease.risk_level}
                      </span>
                    </div>

                    <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", color: "#1e293b", fontWeight: "700" }}>
                      {disease.name}
                    </h4>

                    {/* Percentage Display */}
                    <div style={{ fontSize: "0.85rem", color: "#475569", marginBottom: "0.25rem" }}>
                      Model Probability: <strong style={{ fontSize: "1.25rem", color: color }}>{disease.probability_pct}%</strong>
                    </div>

                    {/* Progress Bar */}
                    <div style={{ width: "100%", height: "8px", background: "#f1f5f9", borderRadius: "4px", overflow: "hidden", marginBottom: "0.75rem" }}>
                      <div
                        style={{
                          width: `${disease.probability_pct}%`,
                          height: "100%",
                          background: color,
                          borderRadius: "4px",
                          transition: "width 0.5s ease"
                        }}
                      />
                    </div>

                    <div style={{ fontSize: "0.8rem", color: "#64748b" }}>
                      Status: <strong>{disease.status}</strong>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Explainability Section (Grad-CAM & Demographics) */}
            {results.xai && (
              <div style={{ background: "#ffffff", borderRadius: "1rem", padding: "2rem", border: "1px solid #e2e8f0", marginBottom: "2rem" }}>
                <h3 style={{ fontSize: "1.2rem", fontWeight: "800", color: "#0f172a", marginBottom: "0.5rem" }}>
                  Model Explainability & Visual Saliency
                </h3>
                <p style={{ fontSize: "0.85rem", color: "#64748b", marginBottom: "1.5rem" }}>
                  {results.xai.disclaimer}
                </p>

                {/* Demographic Attributions */}
                {results.xai.demographic_attributions && (
                  <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{ fontSize: "0.95rem", fontWeight: "700", color: "#334155", marginBottom: "0.75rem" }}>
                      Demographic Risk Sensitivity
                    </h4>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                      {results.xai.demographic_attributions.map((attr, idx) => (
                        <div key={idx} style={{ background: "#f8fafc", padding: "1rem", borderRadius: "0.5rem", border: "1px solid #e2e8f0" }}>
                          <div style={{ fontSize: "0.85rem", fontWeight: "600", color: "#475569" }}>{attr.feature}</div>
                          <div style={{ fontSize: "1.1rem", fontWeight: "700", color: "#0f172a", margin: "0.25rem 0" }}>
                            Gradient Attribution: {attr.attribution}
                          </div>
                          <div style={{ fontSize: "0.8rem", color: "#64748b" }}>{attr.impact}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Reset Button */}
            <div style={{ textAlign: "center", marginTop: "2rem" }}>
              <button
                onClick={resetAssessment}
                style={{
                  background: "#334155",
                  color: "#ffffff",
                  padding: "0.75rem 2rem",
                  fontSize: "0.95rem",
                  fontWeight: "600",
                  border: "none",
                  borderRadius: "0.5rem",
                  cursor: "pointer"
                }}
              >
                ← Perform Another Ophthalmic Assessment
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
