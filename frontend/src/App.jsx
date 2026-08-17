import { BrowserRouter, Routes, Route } from "react-router-dom";
import Landing from "./pages/landing";
import HowItWorks from "./pages/HowItWorks";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/how-it-works" element={<HowItWorks />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;