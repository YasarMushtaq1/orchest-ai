#!/usr/bin/env python3
"""
Generate synthetic training_data.json for flexible worker routing.
Uses rule-based decomposition + routing to produce diverse workflows.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

from orchestai.utils.config_loader import load_config


TASK_TYPES = [
    "summarization",
    "translation",
    "generation",
    "analysis",
    "multi-step",
    "research",
    "content",
    "data",
    "presentation",
    "other",
]

TASK_TYPE_INDEX = {name: idx for idx, name in enumerate(TASK_TYPES)}


def normalize_text(text: str) -> str:
    lowered = text.lower()
    for typo in ["presetation", "prestation", "presentaion", "preentation", "presention"]:
        lowered = lowered.replace(typo, "presentation")
    return lowered


def split_into_subtasks(instruction: str) -> List[str]:
    text = normalize_text(instruction).replace("\n", " ").strip()
    for token in [" and then ", " then ", ";", "."]:
        text = text.replace(token, "||")
    parts = [p.strip(" \"'") for p in text.split("||") if p.strip(" \"'")]
    return parts if parts else [instruction]


def infer_task_type(text: str) -> str:
    t = normalize_text(text)
    if "translate" in t:
        return "translation"
    if any(k in t for k in ["summarize", "summary", "condense"]):
        return "summarization"
    if any(k in t for k in ["analyze", "analysis", "analytics", "data", "chart", "plot"]):
        return "analysis"
    if any(k in t for k in ["research", "investigate", "study"]):
        return "research"
    if any(k in t for k in ["presentation", "slides", "deck"]):
        return "presentation"
    if any(k in t for k in ["report", "document", "doc", "pdf"]):
        return "content"
    if any(k in t for k in ["generate", "create", "write", "build"]):
        return "generation"
    return "other"


def infer_worker_type(text: str, keywords: Dict[str, List[str]]) -> str:
    t = normalize_text(text)
    for worker_type, keys in keywords.items():
        if any(k in t for k in keys):
            return worker_type
    return "text"


def infer_level(text: str, worker_type: str) -> int:
    t = normalize_text(text)
    if any(k in t for k in ["simple", "quick", "basic", "short"]):
        return 1
    if any(k in t for k in ["detailed", "complex", "advanced", "high quality", "high-quality"]):
        return 5
    defaults = {
        "text": 2,
        "presentation": 4,
        "video": 4,
        "vision": 3,
        "audio": 3,
        "voice": 2,
        "document": 3,
        "analytics": 4,
        "code": 4,
    }
    return defaults.get(worker_type, 3)


def action_index_for(worker_type: str, level: int, types: List[str], routing_mode: str, levels: int) -> int:
    if worker_type not in types:
        worker_type = "text" if "text" in types else types[0]
    type_idx = types.index(worker_type)
    if routing_mode == "type_level":
        return type_idx * levels + (level - 1)
    return type_idx


def make_workflow(
    instruction: str,
    types: List[str],
    routing_mode: str,
    levels: int,
    keywords: Dict[str, List[str]],
    dependency_mode: str = "chain",
) -> Dict[str, Any]:
    parts = split_into_subtasks(instruction)
    subtasks = []
    for i, part in enumerate(parts):
        worker_type = infer_worker_type(part, keywords)
        level = infer_level(part, worker_type)
        model_selection = action_index_for(worker_type, level, types, routing_mode, levels)
        task_type_name = infer_task_type(part)
        task_type = TASK_TYPE_INDEX.get(task_type_name, 0)
        if dependency_mode == "chain":
            dependencies = list(range(i))
        elif dependency_mode == "fan_in":
            dependencies = list(range(max(0, i - 2), i))
        elif dependency_mode == "skip":
            dependencies = [j for j in range(i) if (i - j) % 2 == 1]
        else:
            dependencies = list(range(i))
        subtasks.append({
            "id": i,
            "task_type": task_type,
            "dependencies": dependencies,
            "model_selection": model_selection,
        })
    return {"instruction": instruction, "subtasks": subtasks}


def generate_instructions(num: int, keywords: Dict[str, List[str]]) -> List[str]:
    base_topics = [
        "artificial intelligence",
        "machine learning",
        "dogs",
        "cats",
        "space exploration",
        "renewable energy",
        "cybersecurity",
        "healthcare",
        "education",
        "blockchain",
    ]
    prompt_wrappers = [
        "Please {}",
        "Can you {}?",
        "I need you to {}",
        "Hey, {}",
        "Quickly {}",
        "Make it detailed: {}",
        "Do this: {}",
        "Yo, {}",
        "Pls {}",
    ]
    multi_step_connectors = [
        " then ",
        " and then ",
        "; ",
    ]
    typo_variants = [
        ("presentation", "presetation"),
        ("presentation", "presentaion"),
        ("summarize", "summrise"),
        ("translate", "transalte"),
        ("document", "documant"),
    ]

    worker_types = list(keywords.keys()) or ["text"]
    out = []
    for _ in range(num):
        # ensure coverage across worker types by forcing at least one keyword
        worker = random.choice(worker_types)
        key = random.choice(keywords.get(worker, ["text"]))
        topic = random.choice(base_topics)

        # build single or multi-step prompt
        steps = []
        steps.append(f"{key} about {topic}")
        if random.random() < 0.5:
            worker2 = random.choice(worker_types)
            key2 = random.choice(keywords.get(worker2, ["text"]))
            steps.append(f"{key2} of it")
        if random.random() < 0.3:
            worker3 = random.choice(worker_types)
            key3 = random.choice(keywords.get(worker3, ["text"]))
            steps.append(f"{key3} and document it")

        instruction = steps[0]
        for s in steps[1:]:
            instruction += random.choice(multi_step_connectors) + s

        # add wrapper
        wrapper = random.choice(prompt_wrappers)
        instruction = wrapper.format(instruction)

        # add typos occasionally
        if random.random() < 0.2:
            for correct, typo in typo_variants:
                instruction = instruction.replace(correct, typo)

        out.append(instruction)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training_data.json")
    parser.add_argument("--num", type=int, default=1000, help="Number of workflows to generate")
    parser.add_argument("--output", type=str, default="training_data.json", help="Output file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dependency-mode", type=str, default="chain", choices=["chain", "fan_in", "skip"],
                        help="Dependency pattern between subtasks")
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)
    routing = config.get("worker_routing", {})
    types = routing.get("types", ["text"])
    routing_mode = routing.get("routing_mode", "type_only")
    levels = int(routing.get("levels", 5))
    keywords = routing.get("keywords", {})

    instructions = generate_instructions(args.num, keywords)
    workflows = [
        make_workflow(instr, types, routing_mode, levels, keywords, dependency_mode=args.dependency_mode)
        for instr in instructions
    ]

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(workflows, f, indent=2)

    print(f"✅ Generated {len(workflows)} workflows → {output_path}")


if __name__ == "__main__":
    main()

