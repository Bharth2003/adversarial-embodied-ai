import csv
import json

def extract_failures():
    comparison_data = {}
    with open('../datasets/dataset_comparison.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use instruction as key. Some instructions might repeat, but let's hope they are unique enough or we handle them.
            comparison_data[row['instruction']] = {
                'reference_steps': row['reference_steps'],
                'generated_actions': row['generated_actions'],
                'planning_successful': row['planning_successful']
            }

    failures = []
    with open('../datasets/evaluation_report.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['correct'].lower() == 'false':
                is_safe = row['is_safe'] == 'True'
                model_approved = row['model_approved'] == 'True'
                
                failure_type = ""
                if not is_safe and model_approved:
                    failure_type = "False Positive (Dangerous)"
                elif is_safe and not model_approved:
                    failure_type = "False Negative (Annoying)"
                else:
                    failure_type = "Other Error"

                # Join with comparison data
                comp = comparison_data.get(row['instruction'], {})
                
                failures.append({
                    'failure_type': failure_type,
                    'instruction': row['instruction'],
                    'scene': row['scene'],
                    'is_safe': row['is_safe'],
                    'model_approved': row['model_approved'],
                    'safety_reason': row['safety_reason'],
                    'reference_steps': comp.get('reference_steps', 'N/A'),
                    'generated_actions': comp.get('generated_actions', 'N/A'),
                    'planning_successful': comp.get('planning_successful', 'N/A')
                })

    # Sort failures: FP first, then FN
    failures.sort(key=lambda x: x['failure_type'])

    output_file = '../datasets/failure_analysis.csv'
    fieldnames = ['failure_type', 'instruction', 'scene', 'is_safe', 'model_approved', 'safety_reason', 'reference_steps', 'generated_actions', 'planning_successful']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fail in failures:
            writer.writerow(fail)

    print(f"Extracted {len(failures)} failures to {output_file}")
    
    # Print summary of findings
    fp_count = len([f for f in failures if "False Positive" in f['failure_type']])
    fn_count = len([f for f in failures if "False Negative" in f['failure_type']])
    print(f"False Positives: {fp_count}")
    print(f"False Negatives: {fn_count}")

if __name__ == "__main__":
    extract_failures()
