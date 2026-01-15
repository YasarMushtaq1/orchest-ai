#!/usr/bin/env python3
"""
Streamlit GUI to test Phase 1 planner with worker type + level routing.
Shows how planner outputs map to worker types/levels and model mappings.
"""

import os
import sys
from typing import Dict, Any, Optional, List

import streamlit as st
import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from orchestai.utils.config_loader import load_config
from orchestai.planner.planner_model import PlannerModel
from orchestai.worker.worker_layer import WorkerModelLayer


@st.cache_resource
def load_planner_and_workers():
    config_path = os.path.join(project_root, "config.yaml")
    config = load_config(config_path)

    planner_config = config["planner"]
    routing_config = config.get("worker_routing", {})
    routing_mode = routing_config.get("routing_mode", "type_only")
    worker_types = routing_config.get("types", [])
    levels = int(routing_config.get("levels", 5))
    
    # Try to load the newly trained NLP model first, then fallback to older checkpoints
    checkpoint_path = os.path.join(project_root, "checkpoint_best.pth")  # New NLP-trained model
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(project_root, "checkpoints", "phase1_best_model_1000.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(project_root, "checkpoints", "phase1_best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found. Tried: checkpoint_best.pth, checkpoints/phase1_best_model_1000.pth, checkpoints/phase1_best_model.pth")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Resolve action_dim flexibly (prefer checkpoint if present)
    action_dim = None
    for key, tensor in state_dict.items():
        if "model_selector.policy_network" in key and key.endswith("weight") and tensor.ndim == 2:
            action_dim = tensor.shape[0]
    if action_dim is None:
        if worker_types:
            action_dim = len(worker_types) * levels if routing_mode == "type_level" else len(worker_types)
        else:
            action_dim = planner_config["model_selector"].get("action_dim", 8)

    planner_config["model_selector"]["action_dim"] = int(action_dim)
    planner = PlannerModel(
        instruction_encoder_config=planner_config["instruction_encoder"],
        task_decomposer_config=planner_config["task_decomposer"],
        graph_generator_config=planner_config["workflow_graph_generator"],
        model_selector_config=planner_config["model_selector"],
    )

    try:
        planner.load_state_dict(state_dict, strict=True)
    except Exception:
        planner.load_state_dict(state_dict, strict=False)

    planner.eval()

    worker_layer = WorkerModelLayer(
        worker_configs=config["worker_models"],
        routing_config=routing_config,
    )

    return planner, worker_layer, config


def get_actual_subtasks(outputs: Dict[str, Any]) -> int:
    decomposition = outputs.get("decomposition", {})
    max_subtasks = decomposition["subtask_embeddings"].size(1)
    stop_probs = decomposition.get("stop_probs", None)
    if stop_probs is None:
        return max_subtasks
    stop_probs = stop_probs[0].squeeze(-1)
    for i in range(1, max_subtasks):
        if stop_probs[i].item() > 0.5:
            return i
    return max_subtasks


def normalize_text(text: str) -> str:
    lowered = text.lower()
    for typo in ["presetation", "prestation", "presentaion", "preentation", "presention"]:
        lowered = lowered.replace(typo, "presentation")
    return lowered


def rule_based_worker_type(instruction: str, config: Dict[str, Any]) -> str:
    text = normalize_text(instruction)
    keyword_map = config.get("worker_routing", {}).get("keywords", {})
    for worker_type, keywords in keyword_map.items():
        if any(k in text for k in keywords):
            return worker_type
    return "text"


def rule_based_subtasks_with_types(instruction: str) -> List[Dict[str, Optional[str]]]:
    """
    Very simple splitter for simulation: splits on 'then', 'and then', ';', and '.'
    """
    text = normalize_text(instruction).replace("\n", " ").strip()
    for token in [" and then ", " then ", ";", "."]:
        text = text.replace(token, "||")
    parts = [p.strip(" \"'") for p in text.split("||") if p.strip(" \"'")]
    parts = parts if parts else [instruction]

    result: List[Dict[str, Optional[str]]] = []
    for part in parts:
        lowered = part.lower()
        if any(k in lowered for k in ["presentation", "slide", "slides", "deck"]):
            result.append({
                "text": f"Draft content for presentation: {part}",
                "forced_type": "text",
            })
            result.append({
                "text": f"Build presentation slides: {part}",
                "forced_type": "presentation",
            })
        else:
            result.append({
                "text": part,
                "forced_type": None,
            })
    return result


def get_action_index_for_type(worker_layer: WorkerModelLayer, worker_type: str, level: Optional[int] = None) -> Optional[int]:
    if not worker_layer.worker_types:
        return None
    if worker_layer.routing_mode == "type_level":
        levels = max(1, worker_layer.routing_levels)
        if level is None:
            level = 1
        try:
            type_idx = worker_layer.worker_types.index(worker_type)
        except ValueError:
            return None
        return type_idx * levels + (level - 1)
    try:
        return worker_layer.worker_types.index(worker_type)
    except ValueError:
        return None


def rule_based_level(text: str, worker_type: Optional[str] = None) -> int:
    lowered = normalize_text(text)
    if any(k in lowered for k in ["simple", "quick", "basic", "short"]):
        return 1
    if any(k in lowered for k in ["moderate", "normal", "standard"]):
        return 3
    if any(k in lowered for k in ["detailed", "complex", "advanced", "high quality", "high-quality"]):
        return 5
    if worker_type:
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
    return 3


def main():
    st.set_page_config(page_title="OrchestAI Worker Routing Test", page_icon="🧭", layout="wide")
    st.title("🧭 OrchestAI Worker Routing Test (Type + Level)")
    st.caption("Shows how planner outputs map to worker type/level and user mappings.")
    
    # Check which model will be used
    checkpoint_path = os.path.join(project_root, "checkpoint_best.pth")
    if os.path.exists(checkpoint_path):
        st.success("✅ Using NLP-trained model: `checkpoint_best.pth` (trained on 9,515 NLP-diverse examples)")
    else:
        checkpoint_path = os.path.join(project_root, "checkpoints", "phase1_best_model_1000.pth")
        if os.path.exists(checkpoint_path):
            st.info("ℹ️ Using model: `checkpoints/phase1_best_model_1000.pth`")
        else:
            st.warning("⚠️ Using fallback model: `checkpoints/phase1_best_model.pth`")

    try:
        planner, worker_layer, config = load_planner_and_workers()
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.divider()
    st.subheader("Manage Workers (GUI)")
    with st.expander("Add or update worker type/model"):
        col1, col2 = st.columns(2)
        with col1:
            new_type = st.text_input("Worker type (e.g., presentation, 3d)", value="")
            new_model_name = st.text_input("Model name (unique id)", value="")
            new_model_type = st.text_input("Model type (string)", value="")
        with col2:
            new_cost = st.number_input("Cost per token", min_value=0.0, value=0.002, step=0.0001, format="%.4f")
            new_latency = st.number_input("Latency ms", min_value=0.0, value=300.0, step=10.0, format="%.0f")
            new_keywords = st.text_input("Keywords (comma separated)", value="")

        if st.button("Save worker to config"):
            config_path = os.path.join(project_root, "config.yaml")
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)

            cfg.setdefault("worker_models", [])
            cfg.setdefault("worker_routing", {})
            cfg["worker_routing"].setdefault("types", [])
            cfg["worker_routing"].setdefault("mapping", {})
            cfg["worker_routing"].setdefault("keywords", {})
            levels = int(cfg["worker_routing"].get("levels", 5))

            if new_model_name:
                existing = [m for m in cfg["worker_models"] if m.get("name") == new_model_name]
                if existing:
                    existing[0]["model_type"] = new_model_type or existing[0].get("model_type", "custom")
                    existing[0]["cost_per_token"] = float(new_cost)
                    existing[0]["latency_ms"] = float(new_latency)
                else:
                    cfg["worker_models"].append({
                        "name": new_model_name,
                        "model_type": new_model_type or new_type or "custom",
                        "cost_per_token": float(new_cost),
                        "latency_ms": float(new_latency),
                    })

            if new_type:
                if new_type not in cfg["worker_routing"]["types"]:
                    cfg["worker_routing"]["types"].append(new_type)

                cfg["worker_routing"]["mapping"].setdefault(new_type, {})
                for level in range(1, levels + 1):
                    cfg["worker_routing"]["mapping"][new_type].setdefault(level, [])
                    if new_model_name and new_model_name not in cfg["worker_routing"]["mapping"][new_type][level]:
                        cfg["worker_routing"]["mapping"][new_type][level].append(new_model_name)

                if new_keywords:
                    kw_list = [k.strip().lower() for k in new_keywords.split(",") if k.strip()]
                    if kw_list:
                        existing_kw = cfg["worker_routing"]["keywords"].get(new_type, [])
                        for k in kw_list:
                            if k not in existing_kw:
                                existing_kw.append(k)
                        cfg["worker_routing"]["keywords"][new_type] = existing_kw

            with open(config_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            st.success("Saved. Restarting app...")
            st.rerun()

    st.subheader("Input")
    instruction = st.text_area(
        "Instruction",
        value="Create a presentation about artificial intelligence",
        height=80,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        multi_model_strategy = st.selectbox(
            "Multi-model strategy (for same level)",
            ["first", "all"],
            index=0,
        )
    with col_b:
        st.write("Routing mode:", config.get("worker_routing", {}).get("routing_mode", "type_only"))

    use_rule_router = st.checkbox("Use rule-based router (override planner action)", value=True)
    use_rule_decompose = st.checkbox("Simulate subtasks from instruction (rule-based split)", value=True)
    use_rule_level = st.checkbox("Use rule-based level (override complexity)", value=True)
    manual_level = st.slider("Manual level (if rule-based level is off)", min_value=1, max_value=5, value=3)

    if st.button("Run Planner", type="primary"):
        if not instruction.strip():
            st.warning("Please enter an instruction.")
            st.stop()

        with torch.no_grad():
            outputs = planner([instruction.strip()], return_graph=False)

        decomposition = outputs["decomposition"]
        num_subtasks = get_actual_subtasks(outputs)
        simulated_subtasks = rule_based_subtasks_with_types(instruction) if use_rule_decompose else []
        if use_rule_decompose:
            num_subtasks = len(simulated_subtasks)

        task_types_logits = decomposition["task_types"][0][:num_subtasks]
        complexities = decomposition.get("complexities", None)
        if complexities is not None:
            complexities = complexities[0].squeeze(-1)[:num_subtasks]

        model_actions = outputs["model_selections"][0]

        st.subheader("Planner Output → Worker Routing")
        for i in range(num_subtasks):
            action_index = model_actions[i] if i < len(model_actions) else model_actions[-1]
            complexity = complexities[i].item() if complexities is not None else None
            simulated_text = simulated_subtasks[i]["text"] if use_rule_decompose else instruction
            forced_type = simulated_subtasks[i]["forced_type"] if use_rule_decompose else None
            if use_rule_level:
                inferred_type = forced_type or rule_based_worker_type(simulated_text, config)
                level_override = rule_based_level(simulated_text, worker_type=inferred_type)
                complexity = (level_override - 1) / 4.0
            else:
                level_override = manual_level
            selection = worker_layer.decode_action(action_index, complexity=complexity)

            rule_type = None
            if use_rule_router:
                rule_type = forced_type or rule_based_worker_type(simulated_text, config)
                rule_action = get_action_index_for_type(worker_layer, rule_type, level=level_override)
                if rule_action is not None:
                    selection = worker_layer.decode_action(rule_action, complexity=complexity)

            with st.expander(f"Subtask {i+1}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("Action index:", action_index)
                    if complexity is not None:
                        st.write("Complexity:", f"{complexity:.3f}")
                    if use_rule_decompose:
                        st.write("Simulated subtask:", simulated_text)
                        if forced_type:
                            st.write("Forced type:", forced_type)
                with col2:
                    st.write("Worker type:", selection.worker_type)
                    st.write("Level:", selection.level)
                with col3:
                    st.write("Mapped models:", selection.model_names or ["(none mapped)"])
                    if use_rule_router:
                        st.write("Rule-based type:", rule_type)

                # Task type distribution (top)
                probs = torch.softmax(task_types_logits[i], dim=0)
                top_prob, top_idx = torch.topk(probs, k=1)
                st.write("Top task type index:", int(top_idx.item()))
                st.write("Top task type prob:", f"{top_prob.item():.3f}")

        st.subheader("Current Worker Mapping (from config)")
        st.json(config.get("worker_routing", {}))


if __name__ == "__main__":
    main()

