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

export default function DoctorDashboardPage() {
  const { authHeaders, request, user } = useAuth();
  const [clinicalForm, setClinicalForm] = useState(initialClinicalForm);
  const [prediction, setPrediction] = useState(null);
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState("");
  const [patientHistory, setPatientHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const result = await request("/predict", {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify(clinicalForm),
      });
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadPatients = async () => {
    setLoading(true);
    setError("");
    try {
      const result = await request("/doctor/patients", { headers: authHeaders });
      setPatients(result);
      if (result.length > 0) setSelectedPatientId(String(result[0].id));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadPatientHistory = async () => {
    if (!selectedPatientId) return;
    setLoading(true);
    setError("");
    try {
      const result = await request(`/doctor/patients/${selectedPatientId}/screenings`, { headers: authHeaders });
      setPatientHistory(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="stack">
      <section className="card">
        <h2>Doctor Dashboard</h2>
        <p className="muted">Welcome Dr. {user?.full_name}. Review patient history and run quick risk checks.</p>
      </section>

      <section className="card">
        <h3>Quick Risk Check</h3>
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
            {loading ? "Running..." : "Predict"}
          </button>
        </form>
      </section>

      {prediction && (
        <section className="card">
          <h3>Prediction Summary</h3>
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
          <h3>Patient Records</h3>
          <button className="btn ghost" onClick={loadPatients}>
            Load Patients
          </button>
        </div>
        <div className="form-grid cols-2 inline-fields">
          <label>
            Select Patient
            <select value={selectedPatientId} onChange={(e) => setSelectedPatientId(e.target.value)}>
              <option value="">-- Select --</option>
              {patients.map((patient) => (
                <option key={patient.id} value={patient.id}>
                  {patient.full_name} ({patient.email})
                </option>
              ))}
            </select>
          </label>
          <button className="btn" onClick={loadPatientHistory}>
            Load Screening History
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
              {patientHistory.length === 0 && (
                <tr>
                  <td colSpan="5">No screening records found.</td>
                </tr>
              )}
              {patientHistory.map((row) => (
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
