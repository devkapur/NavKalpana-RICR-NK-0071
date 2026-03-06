import { useState } from "react";
import { useAuth } from "../context/AuthContext";

const initialClinicalForm = {
  age: 18393,
  gender: 2,
  height: 168,
  weight: 62,
  ap_hi: 110,
  ap_lo: 80,
  cholesterol: 1,
  gluc: 1,
  smoke: 0,
  alco: 0,
  active: 1,
};

export default function PatientDashboardPage() {
  const { authHeaders, request, user } = useAuth();
  const [clinicalForm, setClinicalForm] = useState(initialClinicalForm);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const result = await request("/screenings", {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify(clinicalForm),
      });
      setPrediction(result.prediction_payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    setLoading(true);
    setError("");
    try {
      const result = await request("/screenings/me", { headers: authHeaders });
      setHistory(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="stack">
      <section className="card">
        <h2>Patient Dashboard</h2>
        <p className="muted">Welcome {user?.full_name}. Submit your vitals to run a new risk screening.</p>
      </section>

      <section className="card">
        <h3>New Screening</h3>
        <form className="form-grid cols-3" onSubmit={handlePredict}>
          {Object.entries(clinicalForm).map(([key, value]) => (
            <label key={key}>
              {key}
              <input
                type="number"
                step={Number.isInteger(value) ? "1" : "0.01"}
                value={value}
                onChange={(e) =>
                  setClinicalForm((prev) => ({
                    ...prev,
                    [key]: Number(e.target.value),
                  }))
                }
                required
              />
            </label>
          ))}
          <button className="btn" type="submit" disabled={loading}>
            {loading ? "Running..." : "Save Screening + Predict"}
          </button>
        </form>
      </section>

      {prediction && (
        <section className="card">
          <h3>Prediction Result</h3>
          <div className="metric-grid">
            <article className="metric">
              <span>Risk Category</span>
              <strong>{prediction.risk_category}</strong>
              <small>{Number(prediction.risk_probability).toFixed(4)} probability</small>
            </article>
            <article className="metric">
              <span>Predicted Class</span>
              <strong>{prediction.predicted_class}</strong>
              <small>Decision threshold {prediction.decision_threshold}</small>
            </article>
            <article className="metric">
              <span>Recommendation</span>
              <strong>{prediction.recommendations?.urgency?.toUpperCase()}</strong>
              <small>{prediction.recommendations?.summary}</small>
            </article>
          </div>
          <h4>Why This Result?</h4>
          <p className="muted">{prediction.why_this_result?.summary}</p>
          <ul className="driver-list">
            {(prediction.why_this_result?.main_factors || []).map((factor, index) => (
              <li key={`${factor.label}-${index}`}>
                <b>{factor.label}</b>
                <span>{factor.reason}</span>
              </li>
            ))}
          </ul>
          <h4>Top Risk Drivers</h4>
          <ul className="driver-list">
            {(prediction.top_shap_drivers || []).map((driver, index) => (
              <li key={`${driver.feature}-${index}`}>
                <b>{driver.feature_label || driver.feature}</b>
                <span>{driver.plain_reason}</span>
              </li>
            ))}
          </ul>

          <h4>Recommended Next Steps</h4>
          <p className="muted">{prediction.recommendations?.summary}</p>
          <ul className="action-list">
            {(prediction.recommendations?.actions || []).map((action, index) => (
              <li key={`${action}-${index}`}>{action}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="card">
        <div className="row-between">
          <h3>My Screening History</h3>
          <button className="btn ghost" onClick={loadHistory}>
            Refresh
          </button>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Risk Category</th>
                <th>Probability</th>
                <th>Class</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 && (
                <tr>
                  <td colSpan="5">No screenings found.</td>
                </tr>
              )}
              {history.map((row) => (
                <tr key={row.id}>
                  <td>{row.id}</td>
                  <td>{row.risk_category}</td>
                  <td>{Number(row.risk_probability).toFixed(4)}</td>
                  <td>{row.predicted_class}</td>
                  <td>{new Date(row.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {error && <div className="error-box">{error}</div>}
    </div>
  );
}
