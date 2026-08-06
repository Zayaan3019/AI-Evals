import { NavLink, Route, Routes } from "react-router-dom";
import { TracesPage } from "./pages/TracesPage";
import { TraceDetailPage } from "./pages/TraceDetailPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { HealthBadge } from "./components/HealthBadge";

export function App() {
  return (
    <div className="layout">
      <header className="top">
        <h1>agentlens</h1>
        <nav>
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
            Traces
          </NavLink>
          <NavLink to="/analytics" className={({ isActive }) => (isActive ? "active" : "")}>
            Analytics
          </NavLink>
        </nav>
        <div style={{ marginLeft: "auto" }}>
          <HealthBadge />
        </div>
      </header>

      <Routes>
        <Route path="/" element={<TracesPage />} />
        <Route path="/traces/:traceId" element={<TraceDetailPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route
          path="*"
          element={<div className="notice">No such page.</div>}
        />
      </Routes>
    </div>
  );
}
