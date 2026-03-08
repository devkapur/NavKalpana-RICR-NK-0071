import { Link, NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function MainLayout() {
  const { isLoggedIn, user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  return (
    <div className="page-shell">
      <header className="topbar">
        <Link to="/" className="logo">
          CardioShield AI
        </Link>
        <nav className="nav">
          <NavLink to="/">Home</NavLink>
          {!isLoggedIn && <NavLink to="/auth">Login</NavLink>}
          {isLoggedIn && user?.role === "patient" && <NavLink to="/patient">Patient Panel</NavLink>}
          {isLoggedIn && user?.role === "doctor" && <NavLink to="/doctor">Doctor Panel</NavLink>}
        </nav>
        <div className="topbar-actions">
          {isLoggedIn ? (
            <>
              <span className="role-badge">{user?.role}</span>
              <button className="btn ghost" onClick={handleLogout}>
                Logout
              </button>
            </>
          ) : (
            <Link className="btn ghost" to="/auth">
              Sign In
            </Link>
          )}
        </div>
      </header>
      <main className="container">
        <Outlet />
      </main>
    </div>
  );
}
