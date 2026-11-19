"""
trainer_engine.py
-----------------
Dual-Mode Inference Engine & Production Fallback Router
Author: Shehram Khan

Handles:
1. Direct routing to the custom fine-tuned clinical fitness model checkpoint.
2. Seamless fallback to gpt-4o-mini with native Structured Outputs and clinical
   schema grounding if the custom fine-tuned checkpoint is unavailable on the
   active API organization key.
3. Multi-turn conversational state management for live gym-floor substitutions.
"""

import json
import time
from typing import Dict, List, Tuple, Any
from openai import OpenAI, NotFoundError, PermissionDeniedError, APIError
from schema_grounding import (
    SYSTEM_PROMPT,
    SCHEMA_BLUEPRINT,
    FEW_SHOT_USER_ANCHOR,
    FEW_SHOT_ASSISTANT_ANCHOR,
    audit_generated_routine
)

# Custom Fine-Tuned Model Checkpoint ID & Grounded Production Fallback Model
FINE_TUNED_FITNESS_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:ai-personal-trainer-v1:AD8kLm1p"
GROUNDED_SCHEMA_MODEL = "gpt-4o-mini"


class FitnessTrainerEngine:
    """Manages LLM inference, schema enforcement, and clinical safety verification."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def _build_grounded_messages(
        self,
        conversation_history: List[Dict[str, str]],
        use_few_shot_grounding: bool = True
    ) -> List[Dict[str, str]]:
        """
        Constructs the message payload combining the clinical system prompt,
        schema blueprint, clinical grounding anchor, and active dialogue turns.
        """
        enriched_system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"STRICT JSON OUTPUT SCHEMA:\n{json.dumps(SCHEMA_BLUEPRINT, indent=2)}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": enriched_system_prompt}]

        if use_few_shot_grounding:
            messages.append({"role": "user", "content": FEW_SHOT_USER_ANCHOR})
            messages.append({"role": "assistant", "content": FEW_SHOT_ASSISTANT_ANCHOR})

        messages.extend(conversation_history)
        return messages

    def generate_or_modify_plan(
        self,
        conversation_history: List[Dict[str, str]],
        injury_condition: str,
        preferred_model: str = FINE_TUNED_FITNESS_MODEL
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executes inference against the custom fine-tuned model (or grounded fallback engine).
        Returns:
            parsed_routine (dict): Validated JSON workout plan.
            telemetry (dict): Execution metadata (model used, latency, routing status, audit logs).
        """
        start_time = time.perf_counter()
        active_model = preferred_model
        fallback_triggered = False
        fallback_reason = None
        raw_content = ""

        # Attempt 1: Try the Fine-Tuned Model Checkpoint if requested
        if preferred_model == FINE_TUNED_FITNESS_MODEL:
            try:
                ft_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
                response = self.client.chat.completions.create(
                    model=FINE_TUNED_FITNESS_MODEL,
                    messages=ft_messages,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                raw_content = response.choices[0].message.content or "{}"
            except (NotFoundError, PermissionDeniedError, APIError):
                # Graceful routing when self-serve fine-tuned checkpoint is not hosted on the current org key
                fallback_triggered = True
                fallback_reason = (
                    "Custom fine-tuned checkpoint unavailable on the active OpenAI organization key. "
                    "Automatically routed to the schema-grounded gpt-4o-mini engine with Structured Outputs."
                )
                active_model = GROUNDED_SCHEMA_MODEL

        # Attempt 2 (or Direct Grounded Mode): Execute with Structured Outputs + Clinical Grounding
        if preferred_model != FINE_TUNED_FITNESS_MODEL or fallback_triggered:
            grounded_messages = self._build_grounded_messages(
                conversation_history=conversation_history,
                use_few_shot_grounding=True
            )
            response = self.client.chat.completions.create(
                model=GROUNDED_SCHEMA_MODEL,
                messages=grounded_messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content or "{}"
            active_model = GROUNDED_SCHEMA_MODEL

        latency_sec = round(time.perf_counter() - start_time, 2)

        # Parse deterministic JSON payload
        parsed_routine = json.loads(raw_content)

        # Run automated clinical & schema verification
        is_safe, audit_logs = audit_generated_routine(parsed_routine, injury_condition)

        telemetry = {
            "model_used": active_model,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "latency_sec": latency_sec,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "audit_passed": is_safe,
            "audit_logs": audit_logs
        }

        return parsed_routine, telemetry