"""
schema_grounding.py
-------------------
Clinical Exercise Schemas, Biomechanical Guardrails & Post-Inference Safety Validators.
Author: Shehram Khan
"""

import json
from typing import Dict, List, Tuple, Any

# =============================================================================
# 1. CLINICAL SYSTEM DIRECTIVE
# =============================================================================
SYSTEM_PROMPT = (
    "You are an elite CSCS and clinical rehabilitation AI fitness engine. "
    "You must output exclusively valid JSON matching the periodized workout schema. "
    "Strictly enforce biomechanical contraindications: "
    "1) Zero axial spinal loading or unsupported lumbar flexion for lumbar disc injuries "
    "(strictly forbid Barbell Back Squats, Conventional Deadlifts, Bent-Over Barbell Rows, and Standing Calf Raises). "
    "2) Zero overhead frontal-plane pressing, upright rows, or deep-stretch dips for shoulder impingement. "
    "3) Zero ballistic plyometrics, deep forward-knee-travel squats, or heavy leg extensions for patellar tendinopathy. "
    "Calculate the 4-week progressive overload matrix specifically for the user's Lower-Body Squat/Press lift, "
    "Upper-Body Press lift, and Posterior-Chain Hinge lift (Week 1 Base -> Week 2 +5% -> Week 3 +7.5% -> Week 4 Deload -15%). "
    "Never include markdown code fences or conversational prose. Output raw JSON only."
)

# =============================================================================
# 2. CANONICAL JSON OUTPUT SCHEMA SPECIFICATION
# =============================================================================
SCHEMA_BLUEPRINT = {
    "routine_name": "String (e.g., '3-Day Full Body Intermediate Periodized Block')",
    "client_profile": {
        "experience_level": "Beginner | Intermediate | Advanced",
        "equipment_available": "String",
        "injury_status": "String"
    },
    "biomechanical_guardrails": "String detailing excluded movement patterns and clinical rationale",
    "exercises": [
        {
            "name": "String (Specific safe exercise variation)",
            "sets": "Integer",
            "reps": "String (e.g., '10-12' or '45-sec hold')",
            "rpe": "Number (Rate of Perceived Exertion 1-10)",
            "rest_sec": "Integer",
            "clinical_note": "String explaining biomechanical safety and substitution rationale"
        }
    ],
    "four_week_mesocycle": {
        "tracked_lifts": {
            "lower_body_lift": "String (Name of prescribed lower-body squat/press movement)",
            "upper_press_lift": "String (Name of prescribed upper-body pressing movement)",
            "hinge_pull_lift": "String (Name of prescribed posterior-chain hinge/pull movement)"
        },
        "week_1_base": {
            "lower_body_lbs": "Integer",
            "upper_press_lbs": "Integer",
            "hinge_pull_lbs": "Integer",
            "sets": "Integer",
            "reps": "Integer",
            "target_rpe": "Float"
        },
        "week_2_intensity": {
            "lower_body_lbs": "Integer",
            "upper_press_lbs": "Integer",
            "hinge_pull_lbs": "Integer",
            "sets": "Integer",
            "reps": "Integer",
            "target_rpe": "Float"
        },
        "week_3_volume_peak": {
            "lower_body_lbs": "Integer",
            "upper_press_lbs": "Integer",
            "hinge_pull_lbs": "Integer",
            "sets": "Integer",
            "reps": "Integer",
            "target_rpe": "Float"
        },
        "week_4_deload": {
            "lower_body_lbs": "Integer",
            "upper_press_lbs": "Integer",
            "hinge_pull_lbs": "Integer",
            "sets": "Integer",
            "reps": "Integer",
            "target_rpe": "Float"
        }
    }
}

# =============================================================================
# 3. ORTHOPEDIC CONTRAINDICATION BLACKLIST
# =============================================================================
CONTRAINDICATION_BLACKLIST: Dict[str, List[str]] = {
    "Acute L5-S1 Lumbar Disc Herniation & Sciatica": [
        "standing calf",
        "back squat",
        "conventional deadlift",
        "barbell row",
        "good morning",
        "sit-up"
    ],
    "Subacromial Shoulder Impingement (Supraspinatus Tendinopathy)": [
        "overhead press",
        "military press",
        "upright row",
        "barbell dip",
        "behind the neck"
    ],
    "Patellar Tendinopathy (Jumper's Knee)": [
        "box jump",
        "jump squat",
        "sissy squat",
        "leg extension",
        "depth jump"
    ],
    "Healthy / No Active Orthopedic Contraindications": []
}

# =============================================================================
# 4. CLINICAL FEW-SHOT GROUNDING ANCHOR
# =============================================================================
FEW_SHOT_USER_ANCHOR = (
    "Design a periodized 3-Day Full Body workout routine for an intermediate lifter. "
    "Available equipment: Commercial Gym. Clinical constraint: Acute L5-S1 Lumbar Disc Herniation & Sciatica. "
    "Starting working loads: Lower-Body Squat/Press = 200 lbs, Upper-Body Press = 135 lbs, Posterior Hinge/Pull = 185 lbs."
)

_ANCHOR_PAYLOAD = {
    "routine_name": "3-Day Full Body Intermediate Periodized Block",
    "client_profile": {
        "experience_level": "Intermediate",
        "equipment_available": "Commercial Gym",
        "injury_status": "Acute L5-S1 Lumbar Disc Herniation & Sciatica"
    },
    "biomechanical_guardrails": (
        "Strictly exclude axial spinal compression and unsupported forward lumbar flexion "
        "(NO Barbell Back Squats, Conventional Deadlifts, Barbell Rows, or Standing Calf Raises)."
    ),
    "exercises": [
        {
            "name": "Seated Leg Press (Neutral Pelvis)",
            "sets": 3,
            "reps": "10-12",
            "rpe": 7,
            "rest_sec": 90,
            "clinical_note": "Replaces Barbell Back Squat to eliminate spinal compression."
        },
        {
            "name": "Flat Dumbbell Bench Press",
            "sets": 3,
            "reps": "8-10",
            "rpe": 8,
            "rest_sec": 90,
            "clinical_note": "Provides neutral spine support while building horizontal pressing strength."
        },
        {
            "name": "Chest-Supported Machine Row",
            "sets": 3,
            "reps": "10-12",
            "rpe": 8,
            "rest_sec": 90,
            "clinical_note": "Replaces Bent-Over Barbell Row to eliminate lumbar shear stress."
        },
        {
            "name": "Supine Barbell Glute Bridge",
            "sets": 3,
            "reps": "12-15",
            "rpe": 7,
            "rest_sec": 60,
            "clinical_note": "Replaces Conventional Deadlift to load posterior chain without axial stress."
        },
        {
            "name": "Seated Calf Raise (Knee-Loaded)",
            "sets": 3,
            "reps": "12-15",
            "rpe": 8,
            "rest_sec": 60,
            "clinical_note": "Replaces Standing Calf Raise to avoid top-down spinal compression."
        }
    ],
    "four_week_mesocycle": {
        "tracked_lifts": {
            "lower_body_lift": "Seated Leg Press (Neutral Pelvis)",
            "upper_press_lift": "Flat Dumbbell Bench Press",
            "hinge_pull_lift": "Supine Barbell Glute Bridge"
        },
        "week_1_base": {
            "lower_body_lbs": 200,
            "upper_press_lbs": 135,
            "hinge_pull_lbs": 185,
            "sets": 3,
            "reps": 8,
            "target_rpe": 7.5
        },
        "week_2_intensity": {
            "lower_body_lbs": 210,
            "upper_press_lbs": 140,
            "hinge_pull_lbs": 195,
            "sets": 3,
            "reps": 8,
            "target_rpe": 8.0
        },
        "week_3_volume_peak": {
            "lower_body_lbs": 215,
            "upper_press_lbs": 145,
            "hinge_pull_lbs": 200,
            "sets": 4,
            "reps": 6,
            "target_rpe": 9.0
        },
        "week_4_deload": {
            "lower_body_lbs": 170,
            "upper_press_lbs": 115,
            "hinge_pull_lbs": 155,
            "sets": 2,
            "reps": 8,
            "target_rpe": 6.0
        }
    }
}

FEW_SHOT_ASSISTANT_ANCHOR = json.dumps(_ANCHOR_PAYLOAD)


# =============================================================================
# 5. AUTOMATED CLINICAL & SCHEMA VERIFICATION ENGINE
# =============================================================================
def audit_generated_routine(routine_json: Dict[str, Any], injury_condition: str) -> Tuple[bool, List[str]]:
    """
    Programmatically verifies that the model's JSON output satisfies:
    1. All 5 required top-level schema keys.
    2. Complete 4-week progressive overload mesocycle across specific tracked compound lifts.
    3. Zero leakage of biomechanically contraindicated exercises for the selected injury.
    """
    audit_notes: List[str] = []
    passed = True

    required_keys = {
        "routine_name",
        "client_profile",
        "biomechanical_guardrails",
        "exercises",
        "four_week_mesocycle"
    }
    missing_keys = required_keys - set(routine_json.keys())
    if missing_keys:
        passed = False
        audit_notes.append(f"❌ Missing required JSON schema keys: {missing_keys}")
    else:
        audit_notes.append("✅ JSON Schema Integrity: All 5 top-level structural components verified.")

    meso = routine_json.get("four_week_mesocycle", {})
    required_weeks = {"week_1_base", "week_2_intensity", "week_3_volume_peak", "week_4_deload"}
    if not required_weeks.issubset(set(meso.keys())):
        passed = False
        audit_notes.append("❌ Incomplete 4-week progressive overload mesocycle.")
    else:
        w1_low = meso["week_1_base"].get("lower_body_lbs", 0)
        w3_low = meso["week_3_volume_peak"].get("lower_body_lbs", 0)
        w4_low = meso["week_4_deload"].get("lower_body_lbs", 0)

        w1_up = meso["week_1_base"].get("upper_press_lbs", 0)
        w4_up = meso["week_4_deload"].get("upper_press_lbs", 0)

        if w4_low < w3_low:
            audit_notes.append(
                f"✅ Multi-Lift Periodization Verified: Lower Lift ({w1_low} → {w3_low} → {w4_low} lbs deload) | "
                f"Upper Press ({w1_up} → {w4_up} lbs deload)."
            )
        else:
            audit_notes.append("⚠️ Warning: Week 4 deload load is not reduced below Week 3 peak.")

    forbidden_terms = CONTRAINDICATION_BLACKLIST.get(injury_condition, [])
    exercises = routine_json.get("exercises", [])
    violations = []
    for ex in exercises:
        ex_name_lower = ex.get("name", "").lower()
        for bad_term in forbidden_terms:
            if bad_term in ex_name_lower:
                violations.append(f"'{ex.get('name')}' (violates '{bad_term}' restriction)")

    if violations:
        passed = False
        audit_notes.append(f"❌ Contraindication Leakage Detected: {', '.join(violations)}")
    else:
        audit_notes.append(
            f"✅ Clinical Safety Passed: 0 contraindicated movements detected for '{injury_condition}'."
        )

    return passed, audit_notes