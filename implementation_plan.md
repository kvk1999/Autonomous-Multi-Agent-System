# Implementation Plan: Accelerated Fleet Dispatch & Route Intelligence (AFDRI)

We will build a high-performance data intelligence application designed for logistics dispatchers to optimize delivery routes and dispatch vehicles in real-time. It demonstrates how GPU acceleration (via PyTorch CUDA / cuDF RAPIDS workflows) and Google Cloud services (BigQuery, Cloud Storage, Gemini Enterprise Agent Platform) combine to solve large-scale routing and analytics bottlenecks.

## User Review Required

> [!IMPORTANT]
> - **GPU Execution**: The application will use PyTorch CUDA on your local **NVIDIA GeForce RTX 3060 Laptop GPU** to run physical GPU-accelerated distance matrix calculations.
> - **GCP Mock Fallback**: To ensure immediate out-of-the-box execution without requiring active GCP billing, the BigQuery, Cloud Storage, and Gemini components will have dual modes: **Mocked/Simulated** (default) and **Live Connection** (via environment variables or UI textboxes).

## Proposed Changes

### 1. Project Directory Structure
```
Autonomous Multi-Agent System/
│
├── app.py                     # Main Streamlit application
├── styles.css                 # Custom CSS stylesheet matching the design system
├── acceleration_engine.py      # CPU vs GPU route optimization and distance matrix engine
├── gcp_connector.py           # BigQuery & Cloud Storage connector (with mock mode)
├── gemini_agent.py            # Gemini Enterprise Agent Interface (with mock mode)
├── requirements.txt           # Python project dependencies
└── README.md                  # Project instructions
```

---

### Component Specifications

#### [NEW] [app.py](file:///d:/Guvi%20FS%20Programs/Autonomous%20Multi-Agent%20System/app.py)
Streamlit dashboard built using the custom CSS design system. It will contain:
- Brand header, theme toggle (Light/Dark), and custom metric cards.
- **Tab 1: Dispatch Console**: Map visualization (Plotly), vehicle tracking, and active order assignment. Trigger routing optimization and view direct before/after results.
- **Tab 2: GPU vs CPU Benchmarks**: Interactive benchmark runner showing scaling curves (execution time vs number of orders) for CPU Pandas vs GPU.
- **Tab 3: Gemini Dispatch AI**: Interactive conversational chat interface.
- **Tab 4: Architecture**: Graphical representations of the BigQuery & GCS data pipeline.

#### [NEW] [styles.css](file:///d:/Guvi%20FS%20Programs/Autonomous%20Multi-Agent%20System/styles.css)
CSS file declaring CSS variables (supporting dark/light theme shifts), styling metric cards, custom HTML data tables, pill-style navigation tabs, and badge styling.

#### [NEW] [acceleration_engine.py](file:///d:/Guvi%20FS%20Programs/Autonomous%20Multi-Agent%20System/acceleration_engine.py)
Core logic for optimization:
- Calculates Haversine distance matrices for \(N\) points.
- CPU implementation: Standard `pandas` double-loop or `numpy` vectorization.
- GPU implementation: PyTorch CUDA tensors mapping operations directly to the RTX 3060 GPU.
- Evaluates routes and assigns orders using a greedy optimization algorithm.

#### [NEW] [gcp_connector.py](file:///d:/Guvi%20FS%20Programs/Autonomous%20Multi-Agent%20System/gcp_connector.py)
Handles data interaction:
- Simulates or reads active delivery history tables from BigQuery.
- Simulates or uploads route history logs to Cloud Storage as Parquet files.

#### [NEW] [gemini_agent.py](file:///d:/Guvi%20FS%20Programs/Autonomous%20Multi-Agent%20System/gemini_agent.py)
Queries the data using Gemini API to act as an Enterprise Agent for Dispatch.

#### [NEW] [requirements.txt](file:///d:/Guvi%20FS%20Programs/Autonomous%20Multi-Agent%20System/requirements.txt)
Required python packages: `streamlit`, `plotly`, `pandas`, `numpy`, `torch` (for local GPU acceleration), `google-genai` or `google-generativeai`.

---

## Verification Plan

### Automated & Manual Verification
1. Install requirements using local pip.
2. Verify GPU acceleration availability inside the Python environment.
3. Launch Streamlit locally: `streamlit run app.py`.
4. Validate execution speeds under the Benchmark tab.
5. Visually verify light and dark mode toggles, chart themes, and interactive controls.
