# BaffleView AI: Super-Resolution Stratigraphic Modeling & Dual-Life Optimization

BaffleView AI is an enterprise-grade petrophysical machine learning application designed specifically for the heterolithic, deep-water turbidite systems of the Permian Basin (Wolfcamp, Spraberry, and Delaware Mountain Group). 

By leveraging the geological frameworks of Anierobi Ekweogwu, the platform sharpens standard 6-inch vertical resolution log data down to a 3 cm scale to detect hidden flow barriers and bypassed pay zones.

## 🚀 Core Functionality
*   **Super-Resolution Engine:** Inverts spatial aliasing using 1D Ensemble Random Forest Regressors to map thin-bedded siltstone laminations.
*   **Dual-Life Well Management:** Optimizes hydraulic fracture targeting during early-phase asset production and transitions to caprock seal mapping for Carbon Capture and Storage (CCS) planning during depletion.
*   **Privacy-First Architecture:** Implements a zero-data-retention, in-memory processing standard compliant with strict corporate IT policies.

---

## 📂 Quick Start Deployment

### 1. Prerequisite Environment Installation
Ensure you have Python 3.9+ installed. Verify your installation and install required operational packages directly via pip:

```bash
pip install -r requirements.txt
```

### 2. Local Testing and Data Generation
To test the environment locally before linking to a cloud host, execute the automated sample engine to output an East Ford Field test profile:

```bash
python generate_test_data.py
```

### 3. Local Web Dashboard Execution
Launch the Streamlit compilation pipeline directly from your local directory:

```bash
streamlit run app.py
```
Your default browser will launch a portal to `http://localhost:8501`.

---

## 🔐 Data Security & Compliance Architecture

*   **Stateless Computations:** All analysis runs directly in volatile RAM. No data arrays are written to disk or permanent databases.
*   **Corporate Header Scrubbing:** The ingestion engine identifies and strips out identifying strings (`UWI`, `API`, `WELL_NAME`, `OPERATOR`) natively during runtime.
*   **Enterprise SIoT Deployment:** Fully compatible with *Streamlit in Snowflake* (SiS) for native deployment within secure corporate virtual clouds (e.g., ExxonMobil, Chevron, Occidental).

---

## 🛢️ Subsurface Header Mapping Matrix

The ingestion scripts normalize incoming files using the following translation dictionary:


| Source Mnemonic | Translation Target | Metric Baseline |
| :--- | :--- | :--- |
| `DEPTH`, `DEPT`, `Z` | `Depth` | Measured Depth (ft / m) |
| `GR`, `GAM`, `GR_EDTC` | `Gamma_Ray` | API Units (0 - 200+) |
| `RESD`, `ILD`, `AT90` | `Deep_Resistivity` | Ohm-m ($\Omega \cdot m$) |
| `NPHI`, `NPOR`, `PHIN` | `Neutron_Porosity` | Decimal Percentage (0.0 - 1.0) |

---

## 🛠️ Offline Pre-Flight Data Diagnostics
Before running web-scale cloud testing, pass your custom logging suites through the localized python testing node:

```bash
python validate_log.py --file your_well_log.csv
```

---
*Disclaimer: BaffleView AI provides structural approximations based on statistical and probabilistic rock physics modeling. All calculations should be verified by a certified operations geologist prior to field execution or injection workflows.*
