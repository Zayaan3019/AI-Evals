import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import "./styles.css";

const client = new QueryClient({
  defaultOptions: {
    queries: {
      // Trace data is append-only and a few seconds stale is harmless, but a
      // dashboard that never refreshes is misleading during an incident.
      staleTime: 10_000,
      refetchInterval: 30_000,
      // Retrying a 401 just burns requests against a credential that will not
      // become valid on its own.
      retry: (count, error) =>
        error instanceof Error && "status" in error && (error as { status: number }).status === 401
          ? false
          : count < 2,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
