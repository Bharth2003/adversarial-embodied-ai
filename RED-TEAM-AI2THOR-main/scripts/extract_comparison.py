import json
import csv
import os

def extract_comparison_data(filepath):
    """
    Extracts high-level reference steps and generated planning tool calls
    from a JSONL planned dataset for comparison.
    """
    results = []
    if not os.path.exists(filepath):
        print(f"Error: File not found {filepath}")
        return results

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                
                instruction = data.get('instruction', 'N/A')
                # High-level steps from original SafeAgentBench metadata
                ref_steps = data.get('original_data', {}).get('step', [])
                
                # Detailed generated plan tool calls
                plan_steps = data.get('plan', {}).get('steps', [])
                generated_actions = [f"{s['tool']}({json.dumps(s.get('arguments', {}))})" for s in plan_steps]
                
                results.append({
                    'row': i + 1,
                    'instruction': instruction,
                    'is_safe': data.get('is_safe', 'N/A'),
                    'reference_steps': ", ".join(ref_steps),
                    'generated_actions': " -> ".join(generated_actions) if generated_actions else "FAIL/NO PLAN",
                    'planning_successful': data.get('planning_successful', False)
                })
            except Exception as e:
                print(f"Error parsing line {i+1}: {e}")
    return results

if __name__ == "__main__":
    input_file = '../datasets/planned_dataset.jsonl'
    output_file = '../datasets/dataset_comparison.csv'
    
    print(f"Extracting comparison data from {input_file}...")
    comparison_results = extract_comparison_data(input_file)
    
    if comparison_results:
        fields = ['row', 'instruction', 'is_safe', 'reference_steps', 'generated_actions', 'planning_successful']
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparison_results)
            
        print(f"Extraction complete. Results saved to {output_file}")
        print(f"Total rows processed: {len(comparison_results)}")
        
        # Display a small sample for verification
        print("\n--- Sample Comparison (First 5 Rows) ---")
        for row in comparison_results[:5]:
            print(f"Row {row['row']}: {row['instruction']}")
            print(f"  Ref: {row['reference_steps']}")
            print(f"  Gen: {row['generated_actions']}")
            print("-" * 40)
    else:
        print("No results extracted.")
