import { Link, useNavigate } from "react-router-dom";

function Navbar() {
  const navigate = useNavigate();

  return (
    <nav className="navbar">
      <Link to="/" className="logo" style={{ textDecoration: "none", color: "inherit" }}>
        🏥 ClinAI
      </Link>

      <div className="nav-links">
        <Link to="/">Home</Link>
        <Link to="/patient-assessment">Clinical Assessment</Link>
        <Link to="/ophthalmic-assessment">Ophthalmic (ODIR)</Link>
        <Link to="/how-it-works">How It Works</Link>
        <a href="/#diseases">Diseases</a>
        <a href="/#about">About</a>
      </div>

      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button className="nav-button" onClick={() => navigate("/patient-assessment")}>
          Clinical CDS
        </button>
        <button className="nav-button" style={{ background: "linear-gradient(135deg, #0369a1, #0284c7)" }} onClick={() => navigate("/ophthalmic-assessment")}>
          Ophthalmic CDS
        </button>
      </div>
    </nav>
  );
}

export default Navbar;