import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { TracesPage } from "./pages/TracesPage";
import { TraceDetailPage } from "./pages/TraceDetailPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { EvalsPage } from "./pages/EvalsPage";
import { EvalRunPage } from "./pages/EvalRunPage";
import { HealthBadge } from "./components/HealthBadge";

const navClass = ({ isActive }: { isActive: boolean }) => (isActive ? "active" : "");

export function App() {
  return (
    <div className="layout">
      <header className="top">
        <h1>agentlens</h1>
        <nav>
          <NavLink to="/evals" className={navClass}>
            Evaluations
          </NavLink>
          <NavLink to="/traces" className={navClass}>
            Traces
          </NavLink>
          <NavLink to="/analytics" className={navClass}>
            Analytics
          </NavLink>
        </nav>
        <div className="health">
          <HealthBadge />
        </div>
      </header>

      {import.meta.env.VITE_PUBLIC_READ === "1" && (
        <div className="notice provenance">
          Public, read-only instance of agentlens: traces, analytics and evaluations recorded by
          the platform from real model calls.{" "}
          <a href="https://github.com/Zayaan3019/AI-Evals" target="_blank" rel="noreferrer">
            Source, design notes and how to reproduce →
          </a>
        </div>
      )}

      <Routes>
        <Route path="/" element={<Navigate to="/evals" replace />} />
        <Route path="/evals" element={<EvalsPage />} />
        <Route path="/evals/runs/:runId" element={<EvalRunPage />} />
        <Route path="/traces" element={<TracesPage />} />
        <Route path="/traces/:traceId" element={<TraceDetailPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="*" element={<div className="notice">No such page.</div>} />
      </Routes>
    </div>
  );
}
