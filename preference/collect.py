# preference/collect.py — collect user preference data at onboarding
#
# We ask enough questions to give UMAP and PCA meaningful signal.
# Each answer is treated as a separate text sample for embedding.

import json
import os
from config import RAW_PREFS_PATH

# 15 questions so UMAP has enough samples to find clusters
ONBOARDING_QUESTIONS = [
    # Response style
    "Do you prefer concise or detailed responses?",
    "Do you prefer bullet points or flowing prose?",
    "Do you want the assistant to explain its reasoning step by step?",
    "Do you prefer formal language or conversational tone?",
    "How do you feel about analogies and metaphors to explain things?",

    # Technical level
    "How technical should explanations be? (beginner / intermediate / expert)",
    "Should the assistant define acronyms and jargon, or assume you know them?",
    "Do you want code examples included when relevant?",

    # Clinical / health (relevant to a pre-appointment app)
    "How much medical detail do you want in health-related summaries?",
    "Do you prefer symptom descriptions in plain English or clinical terms?",
    "Should medication reminders be brief or include dosage context?",
    "How important is it to highlight changes since your last appointment?",

    # Interaction preferences
    "Do you want the assistant to ask clarifying questions?",
    "Should the assistant proactively suggest follow-up topics?",
    "Any other preferences you want the assistant to know about you?",
]


def collect_preferences(force: bool = False) -> list[str]:
    """
    Ask onboarding questions interactively and persist answers.

    Parameters
    ----------
    force : if True, re-collect even if data already exists.

    Returns
    -------
    list of str — one answer per question
    """
    if not force and os.path.exists(RAW_PREFS_PATH):
        with open(RAW_PREFS_PATH) as f:
            answers = json.load(f)
        print(f"[collect] Loaded {len(answers)} saved preferences.")
        return answers

    print("\n=== Preference Setup ===")
    print("Please answer a few questions so the assistant can personalise responses.")
    print("(Press Enter to skip any question)\n")

    answers = []
    for i, q in enumerate(ONBOARDING_QUESTIONS, 1):
        ans = input(f"[{i}/{len(ONBOARDING_QUESTIONS)}] {q}\n> ").strip()
        # Use the question itself as fallback so we always have text to embed
        answers.append(ans if ans else q)

    with open(RAW_PREFS_PATH, "w") as f:
        json.dump(answers, f, indent=2)

    print(f"\n[collect] Saved {len(answers)} preferences to {RAW_PREFS_PATH}\n")
    return answers