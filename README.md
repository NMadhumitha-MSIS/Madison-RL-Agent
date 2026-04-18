**Live site: https://nmadhumitha-msis.github.io/Madison-RL-Agent/**

# Madison Intelligence Agent - Reinforcement Learning System
**Take-Home Final: Reinforcement Learning for Agentic AI Systems**  
**Framework:** Humanitarians.AI - Madison Intelligence Agents

---

## Overview

This project integrates reinforcement learning into the **Madison Intelligence Agent** framework from Humanitarians.AI. Madison agents autonomously gather, evaluate, and synthesize information from multiple sources. The RL system learns **which sources to trust** for different topic contexts.

### RL Approaches Implemented
| # | Approach | Algorithm | Purpose |
|---|----------|-----------|---------| 
| 1 | Exploration Strategy | Contextual Bandits + UCB1 | Per-step source selection |
| 2 | Policy Gradient | REINFORCE with baseline | Episode-level policy learning |

### Agentic System Type
**Research & Analysis Agents** - learning effective information gathering strategies

---

## Quick Start (Google Colab)

1. Upload `Madison_RL_Agent.ipynb` to [Google Colab](https://colab.research.google.com)
2. Run **Cell 1** to install dependencies
3. In **Cell 2**, set your Groq API key:
   ```python
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
   Get a free key at: [console.groq.com](https://console.groq.com)
4. **Runtime → Run All**
5. Training runs ~5-8 minutes on Colab CPU. No GPU needed.

> **No Groq key?** Leave the key as-is - the RL training still runs fully; only the LLM synthesis step is skipped.

---

## Data Sources (All Free, No Paid Keys)

| Source | API | Key Required? |
|--------|-----|--------------|
| Wikipedia | REST API v1 | No |
| arXiv | Official API | No |
| Reddit/Technology | JSON API | No |
| Reddit/Science | JSON API | No |
| DuckDuckGo Instant | Instant Answer API | No |
| Groq (LLM synthesis) | OpenAI-compatible | Free signup |

---

## Project Structure

```
Madison_RL_Agent.ipynb   Main Colab notebook (run this)
Madison_RL_Report.pdf    Technical report
README.md                This file
LearningCurve.png        Generated after running the notebook
```

---

## System Architecture

Seven-layer pipeline:

1. **Input** - User query + topic context
2. **State** - Context Encoder maps topic to index (0 to 6)
3. **Policy** - UCB Bandit and REINFORCE select information source
4. **Action** - Data Fetchers query Wikipedia, arXiv, Reddit, DuckDuckGo
5. **Reward** - RewardSignalEngine scores on success, length, keyword relevance
6. **Update** - UCB: incremental mean. REINFORCE: policy gradient step
7. **Output** - Groq llama-3.3-70b-versatile synthesizes intelligence report

Full visual architecture diagram available at the project website.

---

## Custom Tool: RewardSignalEngine

The **RewardSignalEngine** (Cell 5) is a standalone, reusable reward scoring tool built specifically for Madison's source selection problem. Unlike a simple success/failure signal, it evaluates three independent quality dimensions:

| Component | Condition | Value |
|-----------|-----------|-------|
| `r_success` | Fetch succeeded | +1.0 |
| `r_success` | Fetch failed | -1.0 |
| `r_length` | Word count > 50 | +0.5 |
| `r_length` | Word count < 10 | -0.2 |
| `r_relevance` | Keyword overlap ratio | 0–0.3 |

Final reward clipped to **[-1.0, 2.0]**.

```python
engine = RewardSignalEngine()

# Scalar reward for RL update
reward = engine.score(fetch_result, query)

# Per-component breakdown for debugging
breakdown = engine.score_breakdown(fetch_result, query)
# → {"r_success": 1.0, "r_length": 0.5, "r_relevance": 0.24, "total": 1.74}
```

The tool is independently testable (self-test runs on Cell 5 load), integrates with both UCB and REINFORCE through a single call, and is reusable across other Humanitarians.AI frameworks.

---

## Mathematical Summary

### UCB1 (Contextual Bandit)
```
UCB(s,a,t) = Q(s,a) + c * sqrt(ln(t_s) / N(s,a))
Q(s,a) = Q(s,a) + (r - Q(s,a)) / N(s,a)   [incremental mean]
```

### REINFORCE (Policy Gradient)
```
pi(a|s; theta) = softmax(theta_s)
grad J(theta) = E[ grad log pi(a|s; theta) * (G_t - b(s)) ]
theta_{s,a} = theta_{s,a} + alpha * (G_t - b(s)) * grad log pi(a|s; theta)
```

### Reward Function
```
R(s,a) = r_success + r_length + r_relevance   clipped to [-1.0, 2.0]
```

---

## Dependencies

```bash
pip install groq requests numpy matplotlib pandas scipy
```

All CPU-compatible. No GPU required.

---

## Results Summary

| Agent | Early Reward | Late Reward | Improvement |
|-------|-------------|-------------|-------------|
| Random Baseline | 0.778 | 0.778 | -- |
| UCB Bandit | 0.590 | 0.873 | +0.283 |
| REINFORCE | 0.465 | 0.553 | +0.088 |

Welch t-test: UCB vs REINFORCE late phase - t=0.987, p=0.329 (not statistically significant at α=0.05)

---

## Ethical Considerations

- **Source bias**: Reward-maximizing agents may amplify source biases. Mitigation: diversity constraints.
- **Misinformation**: Reddit weighted lower in reward function; used as supplementary signal only.
- **Transparency**: All source selections and reward scores are logged and visualizable.
- **Privacy**: No personal data stored; system is stateless between sessions.
- **Boundaries**: Queries only public APIs; all outputs carry human-review disclaimer.
