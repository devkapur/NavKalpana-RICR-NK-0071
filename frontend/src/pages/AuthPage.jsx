import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function AuthPage() {
  const navigate = useNavigate();
  const { login, register } = useAuth();
  const [mode, setMode] = useState("login");
  const [role, setRole] = useState("patient");
  const [form, setForm] = useState({ full_name: "", email: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      if (mode === "register") {
        const result = await register({ ...form, role });
        navigate(result.user.role === "doctor" ? "/doctor" : "/patient");
      } else {
        const result = await login({ email: form.email, password: form.password });
        navigate(result.user.role === "doctor" ? "/doctor" : "/patient");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card auth-card">
      <h2>{mode === "login" ? "Welcome Back" : "Create Account"}</h2>
      <p className="muted">Use your role-specific account to continue.</p>
      <div className="mode-switch">
        <button className={`btn ${mode === "login" ? "" : "ghost"}`} onClick={() => setMode("login")}>
          Login
        </button>
        <button className={`btn ${mode === "register" ? "" : "ghost"}`} onClick={() => setMode("register")}>
          Register
        </button>
      </div>

      {mode === "register" && (
        <div className="mode-switch">
          <button className={`btn ${role === "patient" ? "" : "ghost"}`} onClick={() => setRole("patient")}>
            Patient
          </button>
          <button className={`btn ${role === "doctor" ? "" : "ghost"}`} onClick={() => setRole("doctor")}>
            Doctor
          </button>
        </div>
      )}

      <form className="form-grid" onSubmit={onSubmit}>
        {mode === "register" && (
          <label>
            Full Name
            <input
              value={form.full_name}
              onChange={(e) => setForm((prev) => ({ ...prev, full_name: e.target.value }))}
              required
            />
          </label>
        )}
        <label>
          Email
          <input
            type="email"
            value={form.email}
            onChange={(e) => setForm((prev) => ({ ...prev, email: e.target.value }))}
            required
          />
        </label>
        <label>
          Password
          <input
            type="password"
            value={form.password}
            onChange={(e) => setForm((prev) => ({ ...prev, password: e.target.value }))}
            required
          />
        </label>
        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Please wait..." : mode === "login" ? "Login" : "Create Account"}
        </button>
      </form>
      {error && <div className="error-box">{error}</div>}
    </section>
  );
}
