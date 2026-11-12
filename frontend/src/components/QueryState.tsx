import type { ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ApiError } from "../api/client";

/**
 * Uniform loading / error / empty handling.
 *
 * The empty state is distinguished from the error state deliberately. "No
 * traces in this window" and "the query failed" look identical if both render
 * an empty table, and an operator will read the first as the second during an
 * incident.
 *
 * An unreachable backend (gateway 502/503/504, or a network failure) is a
 * different kind of error from a request the API rejected: it says nothing
 * about this section, and every section on the page hits it at once. It gets
 * a quiet per-section placeholder with a retry; the page-level banner in
 * App.tsx says what is going on.
 */
export function isUnreachable(error: unknown): boolean {
  if (error instanceof ApiError) return [502, 503, 504].includes(error.status);
  return error instanceof TypeError; // fetch() rejects with TypeError on network failure
}

export function QueryState({
  isLoading,
  error,
  isEmpty,
  emptyMessage,
  children,
}: {
  isLoading: boolean;
  error: unknown;
  isEmpty?: boolean;
  emptyMessage?: string;
  children: ReactNode;
}) {
  const client = useQueryClient();

  if (isLoading) return <div className="notice">Loading…</div>;

  if (error) {
    if (isUnreachable(error)) {
      return (
        <div className="notice unreachable">
          <span>Waiting for the data service.</span>
          <button type="button" className="linkish" onClick={() => client.refetchQueries({ type: "active" })}>
            Retry now
          </button>
        </div>
      );
    }
    const message =
      error instanceof ApiError
        ? error.detail
          ? `${error.message} (${error.detail})`
          : error.message
        : error instanceof Error
          ? error.message
          : "Unknown error";
    return <div className="notice error">{message}</div>;
  }

  if (isEmpty) {
    return <div className="notice">{emptyMessage ?? "No data in this window."}</div>;
  }

  return <>{children}</>;
}
