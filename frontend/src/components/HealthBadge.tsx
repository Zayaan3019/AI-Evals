import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

/**
 * Live backend status.
 *
 * Shows which dependency is unhealthy rather than a single green/red dot: the
 * platform can serve traces with Postgres down (spans live in ClickHouse) and
 * vice versa, so "degraded" without naming the component is not actionable.
 */
export function HealthBadge() {
  const { data, error } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
  });

  if (error) return <span className="badge err">backend unreachable</span>;
  if (!data) return <span className="badge muted">checking…</span>;

  const failing = Object.entries(data.checks).filter(([, v]) => v !== "ok");
  if (failing.length === 0) {
    return <span className="badge ok">healthy</span>;
  }
  return <span className="badge err">degraded: {failing.map(([k]) => k).join(", ")}</span>;
}
