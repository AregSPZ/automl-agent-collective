# 🤖 AutoML Agent Collective — AI Building an AI

> Built with [LangChain](https://www.langchain.com/), [LangGraph](https://www.langgraph.dev/), and classic ML tools.

---

## 🧠 What It Is

A **multi-agent system** of LLMs that automates the entire machine learning pipeline — from problem framing to feature engineering, model selection, tuning, and evaluation — using a graph of cooperating agents.

Think of it as an **AI-powered ML team**, where each agent has a role, a purpose, and memory. Together, they solve real-world ML tasks, with minimal human input.s

---

## 🎯 What It Does

- **Frames the problem** (detects target, task type, metrics)
- **Engineers features** (encoding, imputing, scaling)
- **Selects models** (based on task type & data)
- **Evaluates performance** (plots, metric)

All driven by LLM agents orchestrated through **LangGraph**.

---

## 🧱 Architecture

```markdown
![System Architecture](assets\graph.png)

The architecture consists of interconnected agents, each responsible for a specific stage of the ML pipeline. These agents communicate through a shared graph structure, enabling seamless collaboration and task delegation.
```

## 🛠️ Tech Stack

| Layer         | Tool                             |
|---------------|----------------------------------|
| Orchestration | [`LangGraph`](https://www.langgraph.dev/)      |
| Agent Logic   | [`LangChain`](https://www.langchain.com/) + Google Gemini Flash 2.0 |
| Data Handling | `pandas`, `scikit-learn`         |
| Evaluation    | `matplotlib`, `sklearn.metrics`  |
| Memory        | Shared app state |

---

## 🚀 Running It


The agents will take over, step by step. Outputs (logs, plots, model files) will be saved in the /outputs/ directory.