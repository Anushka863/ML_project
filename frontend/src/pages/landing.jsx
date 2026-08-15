import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import DiseaseCard from "../components/DiseaseCard";
import FeatureCard from "../components/FeatureCard";

function Landing() {
  return (
    <div>

      <Navbar />

      <Hero />

      {/* Diseases */}

      <section className="section" id="diseases">

        <div className="section-heading">

          <p className="section-label">
            MULTI-DISEASE ANALYSIS
          </p>

          <h2>
            One Assessment.
            <br />
            Multiple Disease Risks.
          </h2>

        </div>

        <div className="disease-grid">

          <DiseaseCard
            icon="🩸"
            title="Diabetes"
            description="Estimate diabetes risk using relevant clinical features."
          />

          <DiseaseCard
            icon="❤️"
            title="Heart Disease"
            description="Analyze cardiovascular risk factors using clinical data."
          />

          <DiseaseCard
            icon="🫘"
            title="Chronic Kidney Disease"
            description="Estimate CKD risk using relevant laboratory and clinical features."
          />

        </div>

      </section>


      {/* Features */}

      <section className="section features-section" id="how-it-works">

        <div className="section-heading">

          <p className="section-label">
            WHY THIS SYSTEM?
          </p>

          <h2>
            More Than Just a Prediction
          </h2>

        </div>

        <div className="feature-grid">

          <FeatureCard
            icon="🧠"
            title="Graph Neural Network"
            description="Model relationships between patient features and diseases using graph-based learning."
          />

          <FeatureCard
            icon="🔍"
            title="Explainable AI"
            description="Understand which clinical factors contributed to the model's prediction."
          />

          <FeatureCard
            icon="🕸️"
            title="Clinical Knowledge Graph"
            description="Explore relationships between symptoms, risk factors and diseases."
          />

        </div>

      </section>


      {/* How it works */}

      <section className="workflow-section">

        <p className="section-label">
          SIMPLE WORKFLOW
        </p>

        <h2>How It Works</h2>

        <div className="workflow">

          <div>
            <span>01</span>
            <h3>Enter Data</h3>
            <p>Provide clinical information.</p>
          </div>

          <div>
            <span>02</span>
            <h3>GNN Analysis</h3>
            <p>The model analyzes relationships.</p>
          </div>

          <div>
            <span>03</span>
            <h3>Risk Prediction</h3>
            <p>Receive multi-disease risk estimates.</p>
          </div>

          <div>
            <span>04</span>
            <h3>Understand Why</h3>
            <p>Explore the model explanation.</p>
          </div>

        </div>

      </section>


      {/* Disclaimer */}

      <section className="disclaimer" id="about">

        <h3>⚠️ Research & Educational Prototype</h3>

        <p>
          This system provides model-based risk estimates for
          research and educational purposes. It does not replace
          professional medical diagnosis, treatment, or clinical judgment.
        </p>

      </section>


      <footer>
      </footer>

    </div>
  );
}

export default Landing;