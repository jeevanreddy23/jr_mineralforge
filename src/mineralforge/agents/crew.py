"""CrewAI multi-agent workflow for the blast vibration risk predictor."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mineralforge.models.prediction import predict_file
from mineralforge.utils.paths import DEFAULT_MODEL_PATH

try:
    from crewai import Agent, Crew, Process, Task
except Exception:  # pragma: no cover - CrewAI is optional at runtime
    Agent = None
    Crew = None
    Process = None
    Task = None


@dataclass(frozen=True)
class CrewResult:
    rows_scored: int
    high_risk_count: int
    medium_risk_count: int
    output_csv: str
    summary: str


@dataclass(frozen=True)
class AgentSpec:
    role: str
    goal: str
    backstory: str


def build_crewai_team():
    """Create the CrewAI agents used by the MVP when CrewAI is installed."""

    if Agent is None or not crewai_llm_configured():
        return agent_specs()
    try:
        data_agent = Agent(
            role="Blast Data QA Agent",
            goal="Validate blast vibration input fields before prediction.",
            backstory="A mining data technician focused on charge weight, burden, spacing, PPV, frequency, and soil type quality.",
            verbose=False,
        )
        model_agent = Agent(
            role="Vibration Risk Model Agent",
            goal="Score Low, Medium, and High vibration risk with the trained model.",
            backstory="A machine learning engineer specializing in blast vibration classification.",
            verbose=False,
        )
        geotech_agent = Agent(
            role="Geotechnical TARP Agent",
            goal="Translate risk predictions into practical field review recommendations.",
            backstory="A geotechnical engineer responsible for safe, explainable blast vibration decisions.",
            verbose=False,
        )
        return [data_agent, model_agent, geotech_agent]
    except Exception:
        return agent_specs()


def agent_specs() -> list[AgentSpec]:
    return [
        AgentSpec(
            role="Blast Data QA Agent",
            goal="Validate charge weight, burden, spacing, PPV, frequency, and soil type.",
            backstory="A deterministic CrewAI-compatible agent specification for input quality review.",
        ),
        AgentSpec(
            role="Vibration Risk Model Agent",
            goal="Score Low, Medium, and High blast vibration risk.",
            backstory="A deterministic CrewAI-compatible agent specification for model inference.",
        ),
        AgentSpec(
            role="Geotechnical TARP Agent",
            goal="Convert risk level into field-review recommendations.",
            backstory="A deterministic CrewAI-compatible agent specification for geotechnical action review.",
        ),
    ]


def crewai_llm_configured() -> bool:
    return any(
        os.getenv(name)
        for name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "AZURE_API_KEY", "AWS_ACCESS_KEY_ID"]
    )


def run_blast_vibration_crew(input_csv: Path, output_csv: Path, model_path: Path = DEFAULT_MODEL_PATH) -> CrewResult:
    """Run the deterministic MVP workflow and expose a CrewAI team definition when installed."""

    agents = build_crewai_team()
    if agents and Crew is not None and Task is not None and agents[0].__class__.__name__ == "Agent":
        # The MVP keeps scoring deterministic. CrewAI agents document the review chain
        # while the trained model performs the auditable prediction step.
        Task(description="Validate blast vibration CSV fields.", expected_output="Validated CSV schema.", agent=agents[0])
        Task(description="Score blast vibration risk with the trained model.", expected_output="Low/Medium/High predictions.", agent=agents[1])
        Task(description="Generate TARP-style recommendations.", expected_output="Field recommendation summary.", agent=agents[2])

    predictions = predict_file(model_path, input_csv, output_csv)
    high_count = int((predictions["Predicted_Vibration_Level"] == "High").sum())
    medium_count = int((predictions["Predicted_Vibration_Level"] == "Medium").sum())
    summary = (
        f"Scored {len(predictions)} blast events. "
        f"High risk: {high_count}. Medium risk: {medium_count}. "
        "Recommendations are included in TARP_Recommendation."
    )
    return CrewResult(len(predictions), high_count, medium_count, str(output_csv), summary)
