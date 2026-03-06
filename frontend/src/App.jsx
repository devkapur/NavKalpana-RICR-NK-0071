import { Navigate, Route, Routes } from "react-router-dom";
import MainLayout from "./components/MainLayout";
import { useAuth } from "./context/AuthContext";
import AuthPage from "./pages/AuthPage";
import DoctorDashboardPage from "./pages/DoctorDashboardPage";
import HomePage from "./pages/HomePage";
import PatientDashboardPage from "./pages/PatientDashboardPage";

function ProtectedRoute({ children, allowRoles }) {
  const { isLoggedIn, user } = useAuth();
  if (!isLoggedIn) return <Navigate to="/auth" replace />;
  if (allowRoles && !allowRoles.includes(user?.role)) return <Navigate to="/" replace />;
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route index element={<HomePage />} />
        <Route path="/auth" element={<AuthPage />} />
        <Route
          path="/patient"
          element={
            <ProtectedRoute allowRoles={["patient"]}>
              <PatientDashboardPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/doctor"
          element={
            <ProtectedRoute allowRoles={["doctor"]}>
              <DoctorDashboardPage />
            </ProtectedRoute>
          }
        />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
