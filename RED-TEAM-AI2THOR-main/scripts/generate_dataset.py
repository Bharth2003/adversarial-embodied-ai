"""
Generate pre-computed deterministic plans from the SafeAgentBench dataset.

Reads ground truth SafeAgentBench .jsonl files, runs the GoalParser and 
DeterministicPlanner, and saves the resulting discrete action plans alongside 
the ground truth 'is_safe' safety label.

This dataset will then be passed to `evaluate_safety.py` for headless 
safety constitution evaluation without needing AI2-THOR.
"""
import json
import os
import argparse
import time
from typing import List, Dict

from ai2thor_lab import Agent, create_openai_client
from ai2thor_lab.goal_planner import GoalParser, DeterministicPlanner

def load_jsonl(filepath: str, is_safe_label: bool) -> List[Dict]:
    """Loads a JSONL file and tags it with the provided safety ground truth."""
    tasks = []
    if not os.path.exists(filepath):
        print(f"[!] Warning: Dataset file not found: {filepath}")
        return tasks
        
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                # Map SafeAgentBench format to our internal naming format
                task = {
                    "instruction": data.get("instruction", ""),
                    "scene": data.get("scene_name", "FloorPlan1"),
                    "is_safe": is_safe_label,
                    "original_data": data
                }
                tasks.append(task)
    return tasks

def generate_plans(
    safe_file: str,
    unsafe_file: str,
    output_file: str,
    limit: int = None
):
    print(f"Loading SafeAgentBench Datasets...")
    safe_tasks = load_jsonl(safe_file, is_safe_label=True)
    unsafe_tasks = load_jsonl(unsafe_file, is_safe_label=False)
    
    all_tasks = safe_tasks + unsafe_tasks
    if limit:
        all_tasks = all_tasks[:limit]
        
    if not all_tasks:
        print("[!] No tasks loaded. Exiting.")
        return

    # Group tasks by scene to reuse the same agent initialization
    tasks_by_scene: Dict[str, List[Dict]] = {}
    for task in all_tasks:
        scene = task.get("scene", "FloorPlan1")
        tasks_by_scene.setdefault(scene, []).append(task)

    results = []
    client = create_openai_client(session_id=f"plan-gen-{int(time.time())}")
    
    total_tasks = len(all_tasks)
    completed = 0
    
    print(f"\n{'='*60}")
    print(f"GENERATING PLANS: {total_tasks} tasks across {len(tasks_by_scene)} scene(s)")
    print(f"{'='*60}\n")
    
    agent = None
    
    for scene_label, scene_tasks in tasks_by_scene.items():
        print(f"--- Init Scene: {scene_label} -- {len(scene_tasks)} task(s) ---")
        try:
            if agent is None:
                agent = Agent(scene=scene_label)
            else:
                agent.controller.reset(scene=scene_label)
        except Exception as e:
            print(f"  FAILED to initialize scene {scene_label}: {e}")
            for t in scene_tasks:
                entry = t.copy()
                entry.update({"planning_successful": False, "reason": f"Scene Init Error: {e}"})
                results.append(entry)
                _save_result(entry, output_file)
            continue
            
        for task_data in scene_tasks:
            instruction = task_data["instruction"]
            completed += 1
            entry = task_data.copy()
            
            try:
                # Reset scene for isolated task run
                agent.controller.reset(scene=scene_label)
                agent.held_object = None
                
                scene_meta = agent.get_scene_metadata()
                
                # 1. Goal Parsing (LLM)
                parser = GoalParser(llm_client=client)
                goals = parser.parse(instruction, scene_meta)
                
                if not goals:
                    entry.update({"planning_successful": False, "reason": "No goals extracted"})
                else:
                    # 2. Deterministic Planning (no LLM)
                    planner = DeterministicPlanner(scene_meta)
                    plan = planner.plan(goals)
                    
                    entry["planning_successful"] = True
                    entry["plan"] = plan
                    entry["plan_length"] = len(plan["steps"])
                    entry["reason"] = "Plan successfully generated"
                    
            except Exception as e:
                entry.update({"planning_successful": False, "reason": f"Planning exception: {str(e)}"})
                
            results.append(entry)
            _save_result(entry, output_file)
            
            status = "OK" if entry.get("planning_successful") else "FAIL"
            length = f"(Len: {entry.get('plan_length', 0):>2})" if entry.get("planning_successful") else "         "
            print(f"  [{completed:>3}/{total_tasks}] {status} {length}: {instruction[:60]}")
            
    # Summary
    if agent is not None:
        agent.controller.stop()
        
    planned = sum(1 for r in results if r.get("planning_successful"))
    print(f"\n=============================================")
    print(f"DONE. Automatically planned {planned}/{total_tasks} tasks.")
    print(f"Saved dataset plans to: {os.path.abspath(output_file)}")

def _save_result(entry: Dict, filepath: str):
    with open(filepath, "a") as f:
        f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute deterministic plans from SafeAgentBench")
    parser.add_argument("--safe-file", default="../../SafeAgentBench/dataset/safe_detailed_1009.jsonl", help="Path to SafeAgentBench safe file")
    parser.add_argument("--unsafe-file", default="../../SafeAgentBench/dataset/unsafe_detailed_1009.jsonl", help="Path to SafeAgentBench unsafe file")
    parser.add_argument("--output", default="../datasets/planned_dataset.jsonl", help="Where to save the precomputed plans")
    parser.add_argument("--limit", type=int, default=None, help="Max total tasks to process across both files")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)
        
    generate_plans(
        safe_file=args.safe_file,
        unsafe_file=args.unsafe_file,
        output_file=args.output,
        limit=args.limit
    )
