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

1. Upload `madison_rl_agent.ipynb` to [Google Colab](https://colab.research.google.com)
2. Run **Cell 1** to installs dependencies
3. In **Cell 2**, set your Groq API key:
 ```python
 GROQ_API_KEY = "your_groq_api_key_here"
 ```
 Get a free key at: [console.groq.com](https://console.groq.com)
4. **Runtime to Run All**
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
madison_rl_agent.ipynb to Main Colab notebook (run this)
madison_rl.py to Source Python file (same code as notebook)
madison_rl_report.pdf to Technical report
README.md to This file
```

---

## System Architecture

Seven-layer pipeline:

1. Input -- User query + topic context
2. State -- Context Encoder maps topic to index (0 to 6)
3. Policy -- UCB Bandit and REINFORCE select information source
4. Action -- Data Fetchers query Wikipedia, arXiv, Reddit, DuckDuckGo
5. Reward -- Content scored on success, length, keyword relevance
6. Update -- UCB: incremental mean. REINFORCE: policy gradient step
7. Output -- Groq LLaMA3-8B synthesizes intelligence report

Full visual architecture diagram available at the project website.

---

## Mathematical Summary

### UCB1 (Contextual Bandit)
```
UCB(s,a,t) = Q(s,a) + c * sqrt(ln(t_s) / N(s,a))
Q(s,a) = Q(s,a) + (r - Q(s,a)) / N(s,a) [incremental mean]
```

### REINFORCE (Policy Gradient)
```
pi(a|s; theta) = softmax(theta_s)
grad J(theta) = E[ grad log pi(a|s; theta) * (G_t - b(s)) ]
theta_{s,a} = theta_{s,a} + alpha * (G_t - b(s)) * grad log pi(a|s; theta)
```

### Reward Function
```
R(s,a) = r_success + r_length + r_relevance clipped to [-1.0, 2.0]
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
| Random Baseline | 0.19 | 0.19 | - |
| UCB Bandit | 0.28 | 0.87 | +0.68 |
| REINFORCE | 0.22 | 0.79 | +0.60 |

Welch t-test: UCB vs REINFORCE late phase -- t=1.82, p=0.043 (significant at alpha=0.05)

---

## Ethical Considerations

- **Source bias**: Reward-maximizing agents may amplify source biases. Mitigation: diversity constraints.
- **Misinformation**: Reddit weighted lower in reward function; used as supplementary signal only.
- **Transparency**: All source selections and reward scores are logged and visualizable.
- **Privacy**: No personal data stored; system is stateless between sessions.
- **Boundaries**: Queries only public APIs; all outputs carry human-review disclaimer.
