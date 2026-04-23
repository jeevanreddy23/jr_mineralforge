"""
ReasoningAgent: The NLI-based Logic Gate for MineralForge.
Transforms swarm outputs into verified geological reasoning.
"""

import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm_factory import get_llm
from utils.logging_utils import get_logger

log = get_logger(__name__)

class ReasoningAgent:
    """
    Applies Natural Language Inference (NLI) to validate hypotheses 
    against integrated geophysical data.
    """

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def validate_prospectivity(self, integrated_data: str, user_hypothesis: str) -> Dict[str, Any]:
        """
        Performs Zero-Shot NLI classification on the relationship 
        between the Data (Premise) and the Geologist's Intent (Hypothesis).
        """
        log.info("ReasoningAgent: Initiating NLI validation loop...")

        system_prompt = (
            "### SYSTEM ROLE\n"
            "You are the 'ReasoningValidator' for JR MineralForge. Your goal is to apply "
            "Natural Language Inference (NLI) to the outputs of the Swarm Agents to "
            "determine the logical validity of a proposed drilling target.\n\n"
            "### OUTPUT FORMAT (JSON)\n"
            "{\n"
            "  \"nli_label\": \"ENTAILMENT | NEUTRAL | CONTRADICTION\",\n"
            "  \"confidence\": 0.00,\n"
            "  \"geological_justification\": \"\",\n"
            "  \"recommended_action\": \"DRILL | EXPAND_SEARCH | DROP\"\n"
            "}"
        )

        user_content = (
            f"### CONTEXT\n"
            f"- **Premise (Data Evidence)**: {integrated_data}\n"
            f"- **Hypothesis (Geologist's Intent)**: {user_hypothesis}\n\n"
            f"### TASK\n"
            f"Perform a Zero-Shot NLI classification on the relationship between the Premise and the Hypothesis.\n\n"
            f"1. **Classification**: Categorize the relationship as ENTAILMENT, NEUTRAL, or CONTRADICTION.\n"
            f"2. **Confidence Score**: Provide a score (0.0 - 1.0) based on data density and resonance.\n"
            f"3. **Reasoning Trace**: Explain WHY the data supports or refutes the hypothesis.\n"
            f"    - If CONTRADICTION: Identify specific conflicts.\n"
            f"    - If NEUTRAL: Specify what additional data is required."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        try:
            # Clean possible markdown format
            cleaned = content.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            log.error("ReasoningAgent: Failed to parse NLI JSON output.")
            return {
                "nli_label": "NEUTRAL",
                "confidence": 0.0,
                "geological_justification": "Failed to generate valid NLI logic trace.",
                "recommended_action": "EXPAND_SEARCH"
            }

def run_reasoning_validator(integrated_data: str, hypothesis: str) -> Dict[str, Any]:
    agent = ReasoningAgent()
    return agent.validate_prospectivity(integrated_data, hypothesis)
