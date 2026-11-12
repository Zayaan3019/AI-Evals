import { useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { isUnreachable } from "./QueryState";

/**
 * One page-level explanation when the API cannot be reached, instead of a red
 * error in every section. Shares the "health" query with HealthBadge, so it
 * costs no extra request and clears itself (and refetches the page's data) as
 * soon as the backend answers again.
 */
export function OutageBanner() {
  const client = useQueryClient();
  const { error, isFetching } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
  });

  const down = Boolean(error) && isUnreachable(error);
  const wasDown = useRef(false);
  useEffect(() => {
    // Backend just came back: refill every section now, not on its next poll.
    if (wasDown.current && !down) void client.refetchQueries({ type: "active" });
    wasDown.current = down;
  }, [down, client]);

  if (!down) return null;

  return (
    <div className="notice outage" role="status">
      <strong>The data service is not answering right now.</strong> The dashboard is up; the API
      behind it is restarting or briefly unavailable. This page checks again every 15 seconds and
      fills in on its own.{" "}
      <button
        type="button"
        className="linkish"
        disabled={isFetching}
        onClick={() => client.refetchQueries({ type: "active" })}
      >
        {isFetching ? "Checking…" : "Check now"}
      </button>
    </div>
  );
}
