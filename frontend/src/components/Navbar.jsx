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
        <Link to="/patient-assessment">Assessment</Link>
        <Link to="/how-it-works">How It Works</Link>
        <a href="/#diseases">Diseases</a>
        <a href="/#about">About</a>
      </div>

      <button className="nav-button" onClick={() => navigate("/patient-assessment")}>
        Start Assessment
      </button>
    </nav>
  );
}

export default Navbar;