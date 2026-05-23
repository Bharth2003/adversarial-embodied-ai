import argparse
import sys
import uuid
import os # Added for os.getenv
import ai2thor
from ai2thor_lab import Agent, interactive_mode, LLMWrapper, create_openai_client, PlanExecutor, VoiceListener
from ai2thor_lab.safety import SafetyReviewer
from ai2thor_lab.goal_planner import GoalParser, DeterministicPlanner

def run_deterministic_task(
    agent, 
    instruction: str, 
    llm_client=None, 
    safety_client=None,
    interrupt_callback=None, # Added interrupt_callback to signature
    constitution_key="C09_Baseline_No_Rules",
    plan_only=False
):
    """Run a single task using the deterministic pipeline."""
    print("\n" + "="*50)
    print("DETERMINISTIC PIPELINE: GOALS → PLAN → EXECUTE")
    print("="*50)
    
    scene_meta = agent.get_scene_metadata()
    
    # 1. PARSE GOALS (single LLM call)
    print(f"\n[1] Parsing goals for task: '{instruction}'") # Changed task_text to instruction
    goal_parser = GoalParser(llm_client=llm_client) # Changed client to llm_client
    goals = goal_parser.parse(instruction, scene_meta) # Changed task_text to instruction
    
    if not goals:
        print("[!] No goals extracted from task. Aborting.")
        return False
    
    # 2. GENERATE PLAN (instant, deterministic, no LLM)
    print("[2] Generating execution plan...")
    planner = DeterministicPlanner(scene_meta)
    plan = planner.plan(goals)
    DeterministicPlanner.print_plan(plan)
    
    if plan_only:
        print("\n[!] --plan-only flag provided. Halting after deterministic planning (skipping safety & execution).")
        return True
    
    # 2.5 SAFETY REVIEW
    print(f"\n[2.5] Reviewing plan for safety using {constitution_key}...")
    # 2. Safety Review (via LLM)
    reviewer = SafetyReviewer(llm_client=safety_client if safety_client else llm_client, constitution_key=constitution_key)
    review_result = reviewer.review_plan(instruction, plan) # Changed safety_reviewer to reviewer, task_text to instruction
    
    if not review_result.get("approved", False):
        print(f"\n[!] Plan rejected by Safety Officer. Reason: {review_result.get('review', {}).get('reason', 'Unknown reason')}")
        return False
        
    print(f"\n[+] Plan approved by Safety Officer.")
    
    # 3. EXECUTE
    print("[3] Executing plan...")
    executor = PlanExecutor(agent)
    exec_result = executor.execute_plan(plan, interrupt_callback=interrupt_callback)
    
    if exec_result["success"]:
        print("\nTask completed successfully.")
        return True
    else:
        print(f"\nExecution failed at step {exec_result.get('failed_step')}: {exec_result.get('reason')}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AI2-THOR Lab CLI")
    parser.add_argument("--task", type=str, help="Run a specific task using the LLM agent")
    parser.add_argument("--scene", type=str, default="FloorPlan1", help="Virtual environment scene to load")
    parser.add_argument("--plan", action="store_true", help="Use the deterministic Plan -> Execute pipeline")
    parser.add_argument("--voice", action="store_true", help="Use microphone for voice commands (always-on listening)")
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse tracing")
    parser.add_argument("--constitution", type=str, default="C09_Baseline_No_Rules", help="Safety constitution to use for planning")
    parser.add_argument("--plan-only", action="store_true", help="Generate and review the plan without executing it")
    parser.add_argument("--api-key", type=str, default=os.getenv("ollama_api_key", "ollama"), help="OpenAI API key") # Added
    parser.add_argument("--model", type=str, default=os.getenv("ollama_model", "qwen2.5-32k:latest"), help="OpenAI model to use") # Added
    parser.add_argument("--base-url", type=str, default=os.getenv("ollama_url", "http://ubuntu:11434/v1"), help="OpenAI API base URL") # Added
    parser.add_argument("--server", action="store_true", help="Start the AI2-THOR Lab web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    args = parser.parse_args()

    if args.server:
        import uvicorn
        print(f"[*] Starting AI2-THOR Lab Server on port {args.port}...")
        uvicorn.run("ai2thor_lab.server:app", host="0.0.0.0", port=args.port, reload=True)
        return

    if args.voice and not args.plan:
        print("[!] Error: --voice requires --plan (voice commands feed directly into the deterministic pipeline).")
        sys.exit(1)

    agent = Agent(scene=args.scene)

    # Initialize OpenAI client early if we are doing LLM/Planning tasks
    client = None
    safety_client = None # Initialize safety_client
    if args.task or args.voice or args.plan: # Added args.plan to condition
        current_session_id = f"task-{uuid.uuid4().hex[:8]}"
        if args.langfuse:
            print(f"Starting Langfuse Session: {current_session_id}")
        
        client = create_openai_client(
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            session_id=current_session_id,
            use_langfuse=args.langfuse
        )
        
        # Dedicated safety client setup, defaults to general API if not specified 
        safety_model = os.getenv("SAFETY_LLM_MODEL", args.model)
        safety_url = os.getenv("SAFETY_LLM_URL", args.base_url)
        safety_api_key = os.getenv("SAFETY_LLM_API_KEY", args.api_key)
        
        safety_client = create_openai_client(
            api_key=safety_api_key,
            model=safety_model,
            base_url=safety_url,
            session_id=current_session_id,
            use_langfuse=args.langfuse
        )

    if args.voice:
        # VOICE LOOP
        print("\nInitializing Voice Command Interface...")
        listener = VoiceListener()
        listener.start()
        
        try:
            import time
            while True:
                print(f"\nRobot is listening (Threshold: {listener.vad_threshold})...")
                print("Speak now. Press Ctrl+C to stop.")
                
                task_text = None
                while not task_text:
                    task_text = listener.get_text()
                    if not task_text:
                        time.sleep(0.1)
                    
                print(f"\n>>> Running voice command: '{task_text}'")
                
                def check_interrupt():
                    text = listener.get_text()
                    if text:
                        text_lower = text.lower().strip()
                        import string
                        text_lower = text_lower.translate(str.maketrans('', '', string.punctuation))
                        words = set(text_lower.split())
                        if {"no", "cancel", "stop", "wait", "halt"} & words:
                            print(f"\n  [VOICE] Interrupt received: '{text}'")
                            return True
                        else:
                            print(f"\n  [VOICE] Ignored during execution: '{text}'")
                    return False

                run_deterministic_task(
                    agent=agent,
                    instruction=task_text,
                    llm_client=client,
                    safety_client=safety_client,
                    interrupt_callback=check_interrupt,
                    constitution_key=args.constitution,
                    plan_only=args.plan_only
                )
                print("\n" + "-"*50)
        except KeyboardInterrupt:
            print("\nShutting down voice interface.")
            listener.stop()
            sys.exit(0)
            
    elif args.task:
        # TEXT TASK (ONE-SHOT)
        if args.plan:
            run_deterministic_task(
                agent=agent,
                instruction=args.task,
                llm_client=client,
                safety_client=safety_client,
                constitution_key=args.constitution,
                plan_only=args.plan_only
            )
        else:
            # Original reactive loop behavior
            wrapper = LLMWrapper(agent, llm_client=client)
            result = wrapper.run(args.task)
            print(f"Result: {result}")
    else:
        # INTERACTIVE MODE (MANUAL CONTROL)
        interactive_mode(agent)

if __name__ == "__main__":
    main()

