import { useEffect, useRef, useState } from "react";
import Navbar from "../components/Navbar";
import { useNavigate } from "react-router-dom";

/* ─────────────────────────────────────────────
   Scroll-reveal hook
   ───────────────────────────────────────────── */
function useScrollReveal() {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { threshold: 0.12 }
    );
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return [ref, visible];
}

/* ─────────────────────────────────────────────
   Workflow steps data
   ───────────────────────────────────────────── */
const WORKFLOW_STEPS = [
  { num: "01", icon: "🧬", title: "Patient Data", desc: "Clinical information such as age, BMI, blood pressure, glucose, and other relevant patient features." },
  { num: "02", icon: "⚙️", title: "Data Preprocessing", desc: "Clinical data is cleaned, normalized, and prepared for analysis to ensure consistency." },
  { num: "03", icon: "🕸️", title: "Clinical Graph", desc: "Patient features and their relationships are represented as a graph of connected clinical information." },
  { num: "04", icon: "🧠", title: "Graph Neural Network", desc: "The GNN learns relationships between connected clinical features and disease-related information." },
  { num: "05", icon: "📊", title: "Multi-Disease Prediction", desc: "The system estimates risk across multiple diseases simultaneously." },
  { num: "06", icon: "🔍", title: "Explainable AI", desc: "The system identifies important features and relationships contributing to the prediction." },
  { num: "07", icon: "🏥", title: "Decision Support", desc: "Results are presented in an understandable form to support further clinical evaluation." },
];

/* ─────────────────────────────────────────────
   XAI contribution data (UI placeholders)
   ───────────────────────────────────────────── */
const XAI_FACTORS = [
  { label: "Glucose", level: "High contribution", width: 88, color: "#167d9a" },
  { label: "BMI", level: "Moderate contribution", width: 62, color: "#2a9d8f" },
  { label: "Age", level: "Moderate contribution", width: 54, color: "#2a9d8f" },
  { label: "Blood Pressure", level: "Lower contribution", width: 34, color: "#8ecbda" },
];

/* ─────────────────────────────────────────────
   Interactive GNN Diagram
   ───────────────────────────────────────────── */
const GNN_NODES = {
  patient:   { cx: 300, cy: 200, r: 38, label: "Patient",        type: "center",  color: "#167d9a" },
  glucose:   { cx: 130, cy: 110, r: 28, label: "Glucose",        type: "feature", color: "#2a9d8f" },
  bmi:       { cx: 130, cy: 200, r: 28, label: "BMI",            type: "feature", color: "#2a9d8f" },
  bp:        { cx: 130, cy: 290, r: 28, label: "Blood Pressure", type: "feature", color: "#2a9d8f" },
  age:       { cx: 300, cy: 345, r: 28, label: "Age",            type: "feature", color: "#2a9d8f" },
  diabetes:  { cx: 480, cy: 110, r: 28, label: "Diabetes Risk",  type: "output",  color: "#e07b39" },
  heart:     { cx: 480, cy: 200, r: 28, label: "Heart Disease Risk", type: "output", color: "#e07b39" },
  hyper:     { cx: 480, cy: 290, r: 28, label: "Hypertension",   type: "output",  color: "#e07b39" },
};

const GNN_EDGES = [
  ["glucose", "patient"], ["bmi", "patient"], ["bp", "patient"], ["age", "patient"],
  ["patient", "diabetes"], ["patient", "heart"], ["patient", "hyper"],
  ["glucose", "diabetes"], ["bmi", "heart"], ["bp", "hyper"],
];

function GNNDiagram() {
  const [hovered, setHovered] = useState(null);

  const connectedTo = (nodeKey) => {
    if (!nodeKey) return new Set();
    const connected = new Set([nodeKey]);
    GNN_EDGES.forEach(([a, b]) => {
      if (a === nodeKey) connected.add(b);
      if (b === nodeKey) connected.add(a);
    });
    return connected;
  };

  const activeSet = hovered ? connectedTo(hovered) : null;

  const edgeHighlighted = (a, b) => {
    if (!hovered) return false;
    return (a === hovered || b === hovered);
  };

  return (
    <div className="gnn-diagram-wrap">
      <svg viewBox="0 0 610 460" className="gnn-svg" aria-label="Clinical Graph Neural Network Diagram">
        {/* Defs */}
        <defs>
          <marker id="arrow-teal" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#167d9a" />
          </marker>
          <marker id="arrow-dim" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#cdd5de" />
          </marker>
          <marker id="arrow-highlight" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#e07b39" />
          </marker>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Column labels */}
        <text x="130" y="42" textAnchor="middle" fontSize="11" fill="#627d98" fontWeight="600" letterSpacing="1">CLINICAL FEATURES</text>
        <text x="300" y="42" textAnchor="middle" fontSize="11" fill="#627d98" fontWeight="600" letterSpacing="1">PATIENT</text>
        <text x="480" y="42" textAnchor="middle" fontSize="11" fill="#627d98" fontWeight="600" letterSpacing="1">DISEASE RISK</text>

        {/* Edges */}
        {GNN_EDGES.map(([a, b], i) => {
          const na = GNN_NODES[a], nb = GNN_NODES[b];
          const highlighted = edgeHighlighted(a, b);
          const dimmed = hovered && !highlighted;
          const strokeColor = dimmed ? "#e5e7eb" : highlighted ? "#e07b39" : "#b8c7d1";
          const markerId = dimmed ? "arrow-dim" : highlighted ? "arrow-highlight" : "arrow-teal";
          // shorten line to not overlap circle
          const dx = nb.cx - na.cx, dy = nb.cy - na.cy;
          const len = Math.sqrt(dx * dx + dy * dy);
          const x1 = na.cx + (dx / len) * (na.r + 3);
          const y1 = na.cy + (dy / len) * (na.r + 3);
          const x2 = nb.cx - (dx / len) * (nb.r + 8);
          const y2 = nb.cy - (dy / len) * (nb.r + 8);
          return (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={strokeColor}
              strokeWidth={highlighted ? 2.5 : 1.5}
              strokeDasharray={highlighted ? "none" : "none"}
              markerEnd={`url(#${markerId})`}
              style={{ transition: "stroke 0.25s, stroke-width 0.25s" }}
            />
          );
        })}

        {/* Nodes */}
        {Object.entries(GNN_NODES).map(([key, n]) => {
          const dimmed = activeSet && !activeSet.has(key);
          const isActive = key === hovered;
          return (
            <g key={key}
              style={{ cursor: "pointer", transition: "opacity 0.25s" }}
              opacity={dimmed ? 0.28 : 1}
              onMouseEnter={() => setHovered(key)}
              onMouseLeave={() => setHovered(null)}
            >
              <circle
                cx={n.cx} cy={n.cy} r={isActive ? n.r + 5 : n.r}
                fill={isActive ? n.color : "white"}
                stroke={n.color}
                strokeWidth={isActive ? 3 : 2}
                filter={isActive ? "url(#glow)" : "none"}
                style={{ transition: "r 0.2s, fill 0.2s, stroke-width 0.2s" }}
              />
              <text
                x={n.cx} y={n.cy}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={n.type === "center" ? "11" : "9.5"}
                fontWeight="700"
                fill={isActive ? "white" : n.color}
                style={{ transition: "fill 0.2s", pointerEvents: "none", userSelect: "none" }}
              >
                {n.label.split(" ").map((word, wi) => (
                  <tspan key={wi} x={n.cx} dy={wi === 0 ? (n.label.includes(" ") ? "-0.5em" : "0") : "1.2em"}>{word}</tspan>
                ))}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform="translate(20, 415)">
          <circle cx="8" cy="8" r="7" fill="white" stroke="#167d9a" strokeWidth="2" />
          <text x="20" y="12" fontSize="10" fill="#627d98">Patient</text>
          <circle cx="80" cy="8" r="7" fill="white" stroke="#2a9d8f" strokeWidth="2" />
          <text x="92" y="12" fontSize="10" fill="#627d98">Clinical Features</text>
          <circle cx="198" cy="8" r="7" fill="white" stroke="#e07b39" strokeWidth="2" />
          <text x="210" y="12" fontSize="10" fill="#627d98">Disease Risk</text>
        </g>
      </svg>

      <p className="gnn-hint">Hover a node to highlight its connections</p>
    </div>
  );
}

/* ─────────────────────────────────────────────
   XAI Graph Explanation Diagram
   ───────────────────────────────────────────── */
function XAIGraph() {
  return (
    <div className="xai-graph-wrap">
      <svg viewBox="0 0 520 260" className="xai-svg" aria-label="XAI Graph Explanation">
        <defs>
          <marker id="xai-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#167d9a" />
          </marker>
          <marker id="xai-arrow-out" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#e07b39" />
          </marker>
        </defs>

        {/* Edges in */}
        {[[80,65,220,130],[80,130,220,130],[80,195,220,130]].map(([x1,y1,x2,y2],i)=>(
          <line key={i} x1={x1+28} y1={y1} x2={x2-32} y2={y2}
            stroke={i===0||i===1?"#167d9a":"#b8c7d1"} strokeWidth={i===0||i===1?2.5:1.5}
            markerEnd="url(#xai-arrow)" />
        ))}
        {/* Edge out */}
        <line x1={252} y1={130} x2={370} y2={130} stroke="#e07b39" strokeWidth={2.5} markerEnd="url(#xai-arrow-out)" />

        {/* Input nodes */}
        {[
          {cx:80,cy:65,label:"Glucose",highlight:true},
          {cx:80,cy:130,label:"BMI",highlight:true},
          {cx:80,cy:195,label:"Blood\nPressure",highlight:false},
        ].map((n,i)=>(
          <g key={i}>
            <circle cx={n.cx} cy={n.cy} r={26} fill={n.highlight?"#e8f7fa":"white"}
              stroke={n.highlight?"#167d9a":"#b8c7d1"} strokeWidth={n.highlight?2.5:1.5} />
            <text x={n.cx} y={n.cy} textAnchor="middle" dominantBaseline="middle"
              fontSize="9.5" fontWeight="700" fill={n.highlight?"#167d9a":"#8a9db5"}>
              {n.label.split("\n").map((t,j)=>(
                <tspan key={j} x={n.cx} dy={j===0?(n.label.includes("\n")?"-0.5em":"0"):"1.3em"}>{t}</tspan>
              ))}
            </text>
          </g>
        ))}

        {/* Patient node */}
        <circle cx={220} cy={130} r={32} fill="#167d9a" stroke="#167d9a" strokeWidth={2} />
        <text x={220} y={130} textAnchor="middle" dominantBaseline="middle" fontSize="11" fontWeight="700" fill="white">Patient</text>

        {/* Output node */}
        <rect x={372} y={103} width={108} height={54} rx={10} fill="#fff4ee" stroke="#e07b39" strokeWidth={2} />
        <text x={426} y={126} textAnchor="middle" fontSize="9.5" fontWeight="700" fill="#e07b39">Diabetes</text>
        <text x={426} y={140} textAnchor="middle" fontSize="9.5" fontWeight="700" fill="#e07b39">Risk ↑</text>

        {/* Highlight annotation on glucose */}
        <g>
          <rect x={2} y={40} width={24} height={14} rx={4} fill="#167d9a" />
          <text x={14} y={50} textAnchor="middle" fontSize="8" fill="white" fontWeight="700">key</text>
        </g>
      </svg>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Main Page
   ───────────────────────────────────────────── */
function HowItWorks() {
  const navigate = useNavigate();
  const [heroRef, heroVisible] = useScrollReveal();
  const [workflowRef, workflowVisible] = useScrollReveal();
  const [gnnRef, gnnVisible] = useScrollReveal();
  const [xaiRef, xaiVisible] = useScrollReveal();
  const [xaiGraphRef, xaiGraphVisible] = useScrollReveal();
  const [dualRef, dualVisible] = useScrollReveal();
  const [ctaRef, ctaVisible] = useScrollReveal();

  return (
    <div className="hiw-page">
      <Navbar />

      {/* ── SECTION 1: Page Intro ── */}
      <section
        className={`hiw-hero reveal-section ${heroVisible ? "revealed" : ""}`}
        ref={heroRef}
      >
        <div className="hiw-hero-inner">
          <span className="hiw-badge">AI-POWERED CLINICAL DECISION SUPPORT</span>
          <h1 className="hiw-hero-h1">How It Works</h1>
          <p className="hiw-hero-sub">
            Understand how patient clinical information moves through our
            explainable multi-disease clinical decision support system.
          </p>
        </div>
        <div className="hiw-hero-bg" aria-hidden="true">
          {[...Array(6)].map((_, i) => (
            <div key={i} className={`hiw-orb hiw-orb-${i + 1}`} />
          ))}
        </div>
      </section>

      {/* ── SECTION 2: System Workflow ── */}
      <section
        className={`hiw-section reveal-section ${workflowVisible ? "revealed" : ""}`}
        ref={workflowRef}
        id="workflow"
      >
        <div className="hiw-section-header">
          <p className="section-label">SYSTEM PIPELINE</p>
          <h2 className="hiw-section-h2">Complete System Workflow</h2>
          <p className="hiw-section-sub">
            Seven steps from raw clinical data to an explainable decision support output.
          </p>
        </div>

        <div className="hiw-workflow-grid">
          {WORKFLOW_STEPS.map((step, i) => (
            <div
              key={step.num}
              className="hiw-workflow-card"
              style={{ animationDelay: workflowVisible ? `${i * 0.1}s` : "0s" }}
            >
              <div className="hiw-workflow-connector" aria-hidden="true">
                {i < WORKFLOW_STEPS.length - 1 && <span className="hiw-connector-arrow">↓</span>}
              </div>
              <div className="hiw-step-num">{step.num}</div>
              <div className="hiw-step-icon">{step.icon}</div>
              <h3 className="hiw-step-title">{step.title}</h3>
              <p className="hiw-step-desc">{step.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── SECTION 3: GNN Explanation ── */}
      <section
        className={`hiw-section hiw-section-alt reveal-section ${gnnVisible ? "revealed" : ""}`}
        ref={gnnRef}
        id="gnn"
      >
        <div className="hiw-section-header">
          <p className="section-label">GRAPH NEURAL NETWORKS</p>
          <h2 className="hiw-section-h2">Understanding Graph Neural Networks</h2>
          <p className="hiw-section-sub">
            Instead of treating every clinical feature independently, a Graph Neural Network
            can learn from relationships between connected clinical information.
          </p>
        </div>

        <GNNDiagram />

        <div className="hiw-gnn-legend-row">
          {[
            { icon: "🔗", title: "Nodes", desc: "Each clinical feature (age, glucose, BMI…) becomes a node in the graph." },
            { icon: "↔️", title: "Edges", desc: "Connections between features capture clinical relationships." },
            { icon: "🧠", title: "Message Passing", desc: "The GNN propagates information across edges to learn richer representations." },
          ].map((item) => (
            <div className="hiw-legend-card" key={item.title}>
              <span className="hiw-legend-icon">{item.icon}</span>
              <h4>{item.title}</h4>
              <p>{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── SECTION 4: XAI Contributions ── */}
      <section
        className={`hiw-section reveal-section ${xaiVisible ? "revealed" : ""}`}
        ref={xaiRef}
        id="xai"
      >
        <div className="hiw-section-header">
          <p className="section-label">EXPLAINABLE AI</p>
          <h2 className="hiw-section-h2">Why This Prediction?</h2>
          <p className="hiw-section-sub">
            The system does not only provide a prediction. Explainable AI helps identify
            the clinical factors and graph relationships that contributed to that prediction.
          </p>
        </div>

        <div className="hiw-xai-container">
          <div className="hiw-xai-left">
            <div className="hiw-xai-disclaimer">
              <span>🔬</span>
              <p>
                <strong>Illustrative UI example.</strong> These contribution values are
                placeholders. When the model is connected, actual model-generated importance
                values will replace these bars.
              </p>
            </div>

            <div className="hiw-xai-disease-label">
              <span className="hiw-disease-tag">Diabetes Risk — Model Prediction Example</span>
            </div>

            <div className="hiw-contribution-bars">
              {XAI_FACTORS.map((f, i) => (
                <div key={f.label} className="hiw-bar-row" style={{ animationDelay: xaiVisible ? `${i * 0.12}s` : "0s" }}>
                  <div className="hiw-bar-label-row">
                    <span className="hiw-bar-feature">{f.label}</span>
                    <span className="hiw-bar-level" style={{ color: f.color }}>{f.level}</span>
                  </div>
                  <div className="hiw-bar-track">
                    <div
                      className="hiw-bar-fill"
                      style={{
                        width: xaiVisible ? `${f.width}%` : "0%",
                        background: `linear-gradient(90deg, ${f.color}, ${f.color}cc)`,
                        transitionDelay: `${i * 0.12}s`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="hiw-xai-right">
            <div className="hiw-xai-info-card">
              <h4>🔑 What does this mean?</h4>
              <ul>
                <li>Each bar shows the <strong>relative contribution</strong> of a clinical feature to the model's prediction.</li>
                <li>A higher bar indicates the feature had <strong>greater influence</strong> on the estimated risk.</li>
                <li>This helps clinicians understand <strong>which factors</strong> drove the model's output.</li>
                <li>These are <strong>model explanations</strong>, not clinical diagnoses.</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ── SECTION 5: XAI Graph Explanation ── */}
      <section
        className={`hiw-section hiw-section-alt reveal-section ${xaiGraphVisible ? "revealed" : ""}`}
        ref={xaiGraphRef}
      >
        <div className="hiw-section-header">
          <p className="section-label">GRAPH EXPLANATION</p>
          <h2 className="hiw-section-h2">Explained Through the Graph</h2>
          <p className="hiw-section-sub">
            Explainable AI can highlight which nodes and edges in the clinical graph
            contributed most to the predicted risk.
          </p>
        </div>

        <div className="hiw-xai-graph-container">
          <XAIGraph />

          <div className="hiw-model-explanation-card">
            <div className="hiw-explain-badge">💡 Model Explanation</div>
            <p className="hiw-explain-text">
              "The model identified <strong>glucose</strong> and <strong>BMI</strong> as
              important contributing factors for the predicted diabetes risk, based on their
              graph connectivity to the patient node."
            </p>
            <p className="hiw-explain-note">
              This is an illustrative UI example. The actual explanation will be generated
              by the connected XAI method (e.g., GNNExplainer).
            </p>
          </div>
        </div>
      </section>

      {/* ── SECTION 6: GNN + XAI Together ── */}
      <section
        className={`hiw-section reveal-section ${dualVisible ? "revealed" : ""}`}
        ref={dualRef}
      >
        <div className="hiw-section-header">
          <p className="section-label">COMBINED APPROACH</p>
          <h2 className="hiw-section-h2">GNN + XAI Working Together</h2>
          <p className="hiw-section-sub">
            The two core technologies combine to deliver both accurate predictions and
            transparent explanations.
          </p>
        </div>

        <div className="hiw-dual-cards">
          <div className="hiw-dual-card hiw-dual-gnn">
            <div className="hiw-dual-icon">🧠</div>
            <h3>Graph Neural Network</h3>
            <p>Learns relationships between connected clinical information to produce multi-disease risk estimates.</p>
            <ul className="hiw-dual-bullets">
              <li>Captures feature interdependencies</li>
              <li>Works on graph-structured data</li>
              <li>Produces latent patient representations</li>
            </ul>
          </div>

          <div className="hiw-dual-connector" aria-hidden="true">
            <div className="hiw-connector-pipeline">
              <span className="hiw-pipeline-label">Relationships</span>
              <span className="hiw-pipeline-arrow">→</span>
              <span className="hiw-pipeline-label">Prediction</span>
              <span className="hiw-pipeline-arrow">→</span>
              <span className="hiw-pipeline-label">Explanation</span>
            </div>
          </div>

          <div className="hiw-dual-card hiw-dual-xai">
            <div className="hiw-dual-icon">🔍</div>
            <h3>Explainable AI</h3>
            <p>Helps identify the features and relationships contributing to the model's prediction.</p>
            <ul className="hiw-dual-bullets">
              <li>Feature importance scores</li>
              <li>Graph-level explanations</li>
              <li>Human-interpretable output</li>
            </ul>
          </div>
        </div>
      </section>

      {/* ── SECTION 7: CTA ── */}
      <section
        className={`hiw-cta reveal-section ${ctaVisible ? "revealed" : ""}`}
        ref={ctaRef}
      >
        <div className="hiw-cta-inner">
          <p className="section-label" style={{ color: "#8ecbda" }}>GET STARTED</p>
          <h2 className="hiw-cta-h2">Ready to analyze clinical risk?</h2>
          <p className="hiw-cta-sub">
            Continue to the patient assessment to provide clinical information
            and explore the system workflow.
          </p>
          <button
            className="primary-button hiw-cta-btn"
            onClick={() => navigate("/assessment")}
          >
            Start Assessment →
          </button>
        </div>
      </section>

      <footer className="hiw-footer">
        <p>© 2026 ClinAI — Research & Educational Prototype | Not for clinical use</p>
      </footer>
    </div>
  );
}

export default HowItWorks;
