---
title: PramanaEnv
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---
# 🧹 PramanaEnv: Autonomous ML Data Engineering Pipeline

**Built for the Meta PyTorch OpenEnv Hackathon**
By: Aadithya Ale 

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/sukuna191552s/PramanaEnv)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Phase 2 Validated](https://img.shields.io/badge/OpenEnv-Phase_2_PASSED-success.svg)](#)

---

## 🎯 Overview (Real-World Utility)
Data cleaning and preprocessing consume up to 80% of a Machine Learning Engineer's time. Most LLM environments test if an agent can write Python code, but **PramanaEnv** tests if an agent can autonomously act as a Senior Data Engineer. 

PramanaEnv is a rigorous, state-managed interactive sandbox where agents must sanitize Pandas DataFrames. Rather than relying on rigid text-matching or unit tests, this environment uses **memory-state evaluation** to ensure datasets are strictly formatted, free of nulls, and mathematically ready for downstream PyTorch training pipelines.

## ✨ "Pro-Tier" Environment Design

We engineered this environment to be highly fault-tolerant, strictly complying with the Meta OpenEnv specifications while punishing lazy AI behavior:

* 🧠 **API-Driven Memory Graders:** We bypassed fragile Python-reflection grading. Our `openenv.yaml` routes validations directly to a FastAPI `/grader` endpoint. Graders physically inspect the underlying Pandas DataFrame in memory using strict `.isnull().any()` and `is_integer_dtype` checks. Agents cannot fake success with textual hallucinations.
* 🛡️ **Titanium Client Architecture:** The `inference.py` client is built for production. It utilizes a deep regex parser (`re.DOTALL`) to extract JSON actions from chatty LLMs, and gracefully falls back to an `undo_last_action` state if the LLM hallucination breaks schema.
* 📜 **Strict Pydantic Contracts:** Client-server communication is strictly enforced via typed `Action` and `Observation` Pydantic models with `ConfigDict(extra='ignore')`, guaranteeing clean action spaces.
* 🕰️ **Temporal Safety (Time-Travel):** Data pipelines are fragile. The environment supports an `undo_last_action` tool, allowing agents to revert their own hallucinations and recover from mistakes without destroying the dataset or crashing the episode.
* ⚖️ **Mathematical Boundary Enforcement:** The client and server environments utilize strict mathematical clamping `min(max(reward, 0.05), 0.95)` to guarantee OpenEnv score compliance even during total network failure.

---

## 🚀 Tasks (Difficulty Progression)

The environment features three distinct, atomic data engineering operations that test different facets of schema enforcement:

1. **`task_1_age`:** Tests imputation and type conversion. The agent must fill missing values with a median target and cast the resulting column strictly to an integer type.
2. **`task_2_salary`:** Tests data-loss decision making. The agent must drop corrupted rows based on null constraints without dropping safe data.
3. **`task_3_price`:** Tests numeric cleaning. The agent must cast string representations into clean, ingestible integer formats.

---

## 🛠️ Quickstart & Local Testing

### 1. Installation
Clone the repository and install the strict dependencies:
```bash
git clone [https://github.com/AadithyaAle/PramanaENV.git](https://github.com/AadithyaAle/PramanaENV.git)
cd PramanaENV
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
