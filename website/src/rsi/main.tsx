import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../styles.css";
import "../qwen38/qwen38.css";
import "./rsi.css";
import RsiPage from "./RsiPage";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RsiPage />
  </StrictMode>,
);
