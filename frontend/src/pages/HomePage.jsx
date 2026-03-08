import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function HomePage() {
  const { isLoggedIn, user, request } = useAuth();
  const dashboardLink = user?.role === "doctor" ? "/doctor" : "/patient";
  const [insights, setInsights] = useState([]);

  useEffect(() => {
    const load = async () => {
      try {
        const result = await request("/explainability/summary");
        setInsights(result.top_features || []);
      } catch {
        setInsights([]);
      }
    };
    load();
  }, [request]);

  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <span className="pill">Welcome to CardioShield</span>
          <h1>For private clinics and modern medical centers</h1>
          <p>
            AI-powered cardiovascular risk stratification with transparent explanations, role-based access, and
            structured follow-up for patient and doctor workflows.
          </p>
          <div className="hero-actions">
            {isLoggedIn ? (
              <Link className="btn" to={dashboardLink}>
                Open Dashboard
              </Link>
            ) : (
              <Link className="btn" to="/auth">
                Get Started
              </Link>
            )}
            <a className="btn ghost" href="#services">
              Learn More
            </a>
          </div>
        </div>
        <div className="hero-visual">
          <div className="doctor-card">
            <div className="doctor-avatar">DR</div>
            <div>
              <strong>Clinical Decision Desk</strong>
              <p>Real-time CVD risk scoring</p>
            </div>
          </div>
          <div className="stat-float">
            <p>Target Recall</p>
            <strong>85%</strong>
          </div>
          <div className="stat-float alt">
            <p>Model ROC-AUC</p>
            <strong>0.80+</strong>
          </div>
        </div>
      </section>

      <section className="service-strip" id="services">
        <article>
          <h4>Risk Prediction</h4>
          <p>Probability, category, calibrated threshold, and confidence interval outputs.</p>
        </article>
        <article>
          <h4>Explainability</h4>
          <p>Top SHAP-based risk drivers with human-readable feature labels for clinicians.</p>
        </article>
        <article>
          <h4>Role-Based Workflow</h4>
          <p>Patients track their screenings, doctors review population-level and patient-level history.</p>
        </article>
      </section>

      <section className="card">
        <h3>Model Insight Summary (SHAP)</h3>
        <p className="muted">
          These are the global factors that most influence risk predictions across the dataset.
        </p>
        <div className="insight-grid">
          {insights.length === 0 && <p className="muted">SHAP summary not available yet.</p>}
          {insights.map((item, idx) => (
            <article className="metric" key={`${item.feature}-${idx}`}>
              <span>{item.feature_label}</span>
              <strong>{Number(item.impact_score || 0).toFixed(3)}</strong>
              <small>{item.plain_reason}</small>
            </article>
          ))}
        </div>
      </section>
    </>
  );
}
