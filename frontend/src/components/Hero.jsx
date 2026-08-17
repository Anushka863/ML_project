import { useNavigate } from "react-router-dom";

function Hero() {
  const navigate = useNavigate();

  return (
    <section className="hero" id="home">

      <div className="hero-content">

        <p className="hero-label">
          AI-POWERED CLINICAL DECISION SUPPORT
        </p>

        <h1>
          Understand Risk.
          <br />
          Understand <span>Why.</span>
        </h1>

        <p className="hero-description">
          An explainable multi-disease clinical decision support
          system powered by Graph Neural Networks and Explainable AI.
        </p>

        <div className="hero-buttons">

          <button className="primary-button" onClick={() => navigate("/how-it-works")}>
            Start  →
          </button>

          <button className="secondary-button" onClick={() => navigate("/how-it-works")}>
            Learn How It Works
          </button>

        </div>

      </div>

    </section>
  );
}

export default Hero;