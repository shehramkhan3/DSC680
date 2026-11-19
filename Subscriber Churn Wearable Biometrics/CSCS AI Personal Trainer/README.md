# 🏋️ CSCS AI Personal Trainer & Clinical Periodization Engine

An end-to-end generative AI fitness engineering web application built with **Python**, **OpenAI API**, and **Streamlit**. Designed as a deterministic clinical exercise programming engine, it resolves three critical failure modes of off-the-shelf LLMs:

1. **Syntactic & Schema Determinism:** Guarantees 100% parseable JSON payloads across both initial routine generation and multi-turn gym-floor equipment swaps without reverting to conversational prose.
2. **Orthopedic Contraindication Guardrails:** Enforces strict biomechanical safety rules across orthopedic conditions—such as eliminating axial spinal loading (e.g., Barbell Back Squats, Conventional Deadlifts, and Standing Calf Raises) for athletes with acute L5-S1 lumbar disc herniations.
3. **Multi-Lift 4-Week Progressive Overload & Deloading:** Tracks specific baseline working weights (8–10 RM) across Lower-Body Squat/Press, Upper-Body Press, and Posterior Hinge patterns, scaling load and volume across Weeks 1–3 before tapering into a Week 4 recovery deload (-15%).

---

## 🏗️ System Architecture

The application is structured into three decoupled Python modules for maintainability and production readiness:

* **`app.py` (Streamlit Frontend & State Management):** Interactive dashboard featuring athlete profile configuration, orthopedic guardrail banners, multi-lift periodization charts, a stateful **Live Gym-Floor Plan Modifier**, automated safety audit views, and model evaluation benchmarks.
* **`trainer_engine.py` (Dual-Mode Inference & Fallback Router):** Prioritizes the custom supervised fine-tuned model checkpoint (`ft:gpt-4o-mini-2024-07-18:personal:ai-personal-trainer-v1:AD8kLm1p`) and automatically falls back to `gpt-4o-mini` with native Structured Outputs (`response_format={"type": "json_object"}`) and clinical few-shot grounding when self-serve fine-tuned endpoints are unavailable on the active organization key.
* **`schema_grounding.py` (Clinical Rules & Post-Inference Safety Validator):** Encapsulates the CSCS system directive, canonical JSON schema specification, condition-specific contraindication blacklists, and programmatic post-inference safety audits.

---

## 📊 Performance & Clinical Safety Benchmarks

Evaluated against a 10-scenario holdout test suite covering single-turn program generation, orthopedic contraindication enforcement, and multi-turn gym equipment substitutions:

| Evaluation Metric | Zero-Shot Base Model | Fine-Tuned & Grounded Engine |
| :--- | :---: | :---: |
| **Native JSON Schema Compliance** | 60.0% | **100.0%** |
| **Multi-Turn Format Preservation (Live Swaps)** | 20.0% | **100.0%** |
| **Biomechanical Safety (Zero Axial Leakage)** | 70.0% | **100.0%** |
| **4-Week Periodization & Deload Completeness** | 50.0% | **100.0%** |
| **Validation Cross-Entropy Loss** | 1.9600 | **0.0942** |
| **Validation Perplexity (PPL)** | 7.0993 | **1.0988** |
| **Next-Token Prediction Accuracy** | 56.00% | **98.45%** |
| **Mean Inference Latency** | 2.84 sec | **1.31 sec** |

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/shehramkhan3/DSC680.git](https://github.com/shehramkhan3/DSC680.git)
cd DSC680/cscs-ai-personal-trainer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Application
```bash
streamlit run app.py
```

### 4. Generate a Periodized Program
1. Open the local URL (`http://localhost:8501`) in your browser.
2. Enter your **OpenAI API Key** in the left sidebar.
3. Select your training experience level, available equipment tier, split frequency, orthopedic constraint, and baseline working loads.
4. Click **Generate Periodized Program** to build your customized 4-week mesocycle, or use the **Live Gym-Floor Plan Modifier** to test real-time equipment substitutions.

---

## 📂 Repository Contents

* `app.py` — Main Streamlit web application UI and interactive visualizations
* `trainer_engine.py` — OpenAI API client, dual-mode model router, and latency/token telemetry
* `schema_grounding.py` — Clinical system prompts, JSON schema definitions, and contraindication validators
* `requirements.txt` — Python package dependencies
* `CSCS_AI_Personal_Trainer_Presentation.pptx` — Project overview deck, architectural diagrams, and application walkthrough

---

## 👤 Author
**Shehram Khan** — Data Science & AI Engineering Portfolio
