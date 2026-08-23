import { BrowserRouter, Routes, Route } from "react-router-dom";
import Landing from "./pages/landing";
import HowItWorks from "./pages/HowItWorks";
import PatientAssessment from "./pages/PatientAssessment";
import PatientReview from "./pages/PatientReview";
import { PatientProvider } from "./context/PatientContext";

function App() {
  return (
    <PatientProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="/patient-assessment" element={<PatientAssessment />} />
          <Route path="/patient-review" element={<PatientReview />} />
        </Routes>
      </BrowserRouter>
    </PatientProvider>
  );
}

export default App;