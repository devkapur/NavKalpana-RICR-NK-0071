import { createContext, useContext, useMemo, useState } from "react";

const AuthContext = createContext(null);
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function parseJsonSafe(value, fallback = null) {
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [user, setUser] = useState(parseJsonSafe(localStorage.getItem("user"), null));

  const isLoggedIn = Boolean(token && user);

  const request = async (path, options = {}) => {
    const response = await fetch(`${API_BASE}${path}`, options);
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      const detail = typeof body === "object" && body?.detail ? body.detail : String(body);
      throw new Error(detail || `Request failed: ${response.status}`);
    }
    return body;
  };

  const authHeaders = useMemo(() => {
    if (!token) return { "Content-Type": "application/json" };
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    };
  }, [token]);

  const persistAuth = (authResponse) => {
    setToken(authResponse.access_token);
    setUser(authResponse.user);
    localStorage.setItem("token", authResponse.access_token);
    localStorage.setItem("user", JSON.stringify(authResponse.user));
  };

  const register = async ({ full_name, email, password, role }) => {
    const endpoint = role === "doctor" ? "/auth/doctor/register" : "/auth/patient/register";
    const result = await request(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ full_name, email, password }),
    });
    persistAuth(result);
    return result;
  };

  const login = async ({ email, password }) => {
    const result = await request("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    persistAuth(result);
    return result;
  };

  const logout = () => {
    setToken("");
    setUser(null);
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  };

  return (
    <AuthContext.Provider value={{ API_BASE, token, user, isLoggedIn, request, authHeaders, register, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
