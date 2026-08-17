function Navbar() {
  return (
    <nav className="navbar">
      <div className="logo">
        🏥 ClinAI
      </div>

      <div className="nav-links">
        <a href="#home">Home</a>
        <a href="#how-it-works">How It Works</a>
        <a href="#diseases">Diseases</a>
        <a href="#about">About</a>
      </div>

      <button className="nav-button">
        Start 
      </button>
    </nav>
  );
}

export default Navbar;