import type { ReactNode } from "react";
import { ApiError } from "../api/client";

/**
 * Uniform loading / error / empty handling.
 *
 * The empty state is distinguished from the error state deliberately. "No
 * traces in this window" and "the query failed" look identical if both render
 * an empty table, and an operator will read the first as the second during an
 * incident.
 */
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
  if (isLoading) return <div className="notice">Loading…</div>;

  if (error) {
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
