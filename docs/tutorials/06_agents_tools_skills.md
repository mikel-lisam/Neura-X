# 06 — Agents, Tools & Skills

## Define tools

```python
import neura_x as nx

@nx.tool(description="Search the web for information")
def web_search(query: str) -> str:
    return nx.utils.search(query)

@nx.tool(description="Run Python in a sandbox")
def run_code(code: str) -> str:
    return nx.utils.sandbox(code)
```

---

##  Attach and call

```python
model = nx.load("my-llm.nex")
model.attach_tools([web_search, run_code])
answer = model.infer("Current population of Nairobi?", use_tools=True)
```
---

## Install skills

```python
model.install_skill("swahili")
model.install_skill("medical")
model.install_skill(nx.hub.pull_skill("edusei/nairobi_guide"))
```

---

## Build an autonomous agent

```python
agent = nx.Agent(
    brain="my-llm.nex",
    skills=["swahili", "research"],
    tools=[web_search, run_code, nx.tools.file_write],
    memory=nx.HolographicMemory(capacity=10_000),
    circadian=True,
)
agent.run("Research AI in Africa, summarize in Swahili, save to ai_africa.md")
```

>Perceive → plan (Liquid Router) → act (tools) → reflect (Sleep) → remember (Holographic Memory) → learn (Circadian).

---

## Multi-agent teams

```python
team = nx.AgentTeam(agents=[researcher, coder, reviewer],
                    coordinator="my-llm.nex")
team.run("Build a scraper and write the report")
```
