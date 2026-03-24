# view_results.py
# ============================================================
# View ASR Results as Text — Human-Readable Output
# ============================================================
# This script reads the CSV results and Excel report and displays
# all information as formatted text in the terminal (and optionally
# exports to a text file for documentation).
#
# Usage:
#   python view_results.py                    # Show results in terminal
#   python view_results.py --export           # Also save to results/ASR_Summary.txt
#   python view_results.py --export-csv       # Also save summary as CSV
# ============================================================

import pandas as pd
import argparse
import os
from datetime import datetime

def view_results(csv_path="results/final_redteam_results.csv", 
                 export_txt=False, export_csv=False):
    """Read the final results CSV and display comprehensive ASR analysis."""
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Results file not found: {csv_path}")
        print("Run the pipeline first: python execute_pipeline.py")
        return
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("ERROR: Results file is empty.")
        return
    
    # Build the complete text report
    lines = []
    lines.append("=" * 70)
    lines.append("  RED TEAM vs BLUE TEAM — ASR RESEARCH REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    # ─── OVERALL METRICS ───
    total = len(df)
    unsafe_count = (df["unsafe_compliance"] == True).sum() if "unsafe_compliance" in df.columns else 0
    refused_count = (df["refused"] == True).sum() if "refused" in df.columns else 0
    
    asr = (unsafe_count / total * 100) if total > 0 else 0
    refusal_rate = (refused_count / total * 100) if total > 0 else 0
    
    lines.append("")
    lines.append("  ╔══════════════════════════════════════════════════╗")
    lines.append(f"  ║  OVERALL RESULTS                                 ║")
    lines.append(f"  ╠══════════════════════════════════════════════════╣")
    lines.append(f"  ║  Total Attack Prompts:    {total:<23} ║")
    lines.append(f"  ║  Unsafe Compliance:       {unsafe_count:<23} ║")
    lines.append(f"  ║  Refused:                 {refused_count:<23} ║")
    lines.append(f"  ║  Neither (ambiguous):     {total - unsafe_count - refused_count:<23} ║")
    lines.append(f"  ║                                                  ║")
    lines.append(f"  ║  ASR (Attack Success Rate):  {asr:>6.1f}%             ║")
    lines.append(f"  ║  Refusal Rate:               {refusal_rate:>6.1f}%             ║")
    lines.append(f"  ╚══════════════════════════════════════════════════╝")
    
    # ─── MODEL INFO ───
    lines.append("")
    lines.append("  MODEL CONFIGURATION:")
    if "red_model" in df.columns:
        lines.append(f"    Red Agent Model:   {df['red_model'].iloc[0] if not df['red_model'].isna().all() else 'unknown'}")
    if "blue_model" in df.columns:
        lines.append(f"    Blue Agent Model:  {df['blue_model'].iloc[0] if not df['blue_model'].isna().all() else 'unknown'}")
    if "judge_model" in df.columns:
        lines.append(f"    Judge Model:       {df['judge_model'].iloc[0] if not df['judge_model'].isna().all() else 'unknown'}")
    
    # ─── ASR BY STRATEGY ───
    if "strategy" in df.columns:
        lines.append("")
        lines.append("  " + "─" * 60)
        lines.append("  ASR BREAKDOWN BY STRATEGY")
        lines.append("  " + "─" * 60)
        lines.append(f"  {'Strategy':<20} {'Total':>6} {'Unsafe':>7} {'Refused':>8} {'ASR %':>8} {'Status':>10}")
        lines.append(f"  {'─'*20} {'─'*6} {'─'*7} {'─'*8} {'─'*8} {'─'*10}")
        
        strategy_data = []
        for strategy in sorted(df["strategy"].unique()):
            mask = df["strategy"] == strategy
            s_total = mask.sum()
            s_unsafe = (df.loc[mask, "unsafe_compliance"] == True).sum() if "unsafe_compliance" in df.columns else 0
            s_refused = (df.loc[mask, "refused"] == True).sum() if "refused" in df.columns else 0
            s_asr = (s_unsafe / s_total * 100) if s_total > 0 else 0
            
            # Status indicator
            if s_asr >= 50:
                status = "🏆 HIGH"
            elif s_asr >= 25:
                status = "⚠ MEDIUM"
            elif s_asr > 0:
                status = "○ LOW"
            else:
                status = "✗ ZERO"
            
            lines.append(f"  {strategy:<20} {s_total:>6} {s_unsafe:>7} {s_refused:>8} {s_asr:>7.1f}% {status:>10}")
            strategy_data.append({
                "strategy": strategy, "total": s_total, "unsafe": s_unsafe,
                "refused": s_refused, "asr_pct": round(s_asr, 2)
            })
    
    # ─── ASR BY RISK CATEGORY ───
    if "risk_category" in df.columns:
        lines.append("")
        lines.append("  " + "─" * 60)
        lines.append("  ASR BREAKDOWN BY RISK CATEGORY")
        lines.append("  " + "─" * 60)
        lines.append(f"  {'Risk Category':<30} {'Total':>6} {'Unsafe':>7} {'ASR %':>8}")
        lines.append(f"  {'─'*30} {'─'*6} {'─'*7} {'─'*8}")
        
        for cat in sorted(df["risk_category"].unique()):
            mask = df["risk_category"] == cat
            c_total = mask.sum()
            c_unsafe = (df.loc[mask, "unsafe_compliance"] == True).sum()
            c_asr = (c_unsafe / c_total * 100) if c_total > 0 else 0
            lines.append(f"  {str(cat):<30} {c_total:>6} {c_unsafe:>7} {c_asr:>7.1f}%")
    
    # ─── SAMPLE SUCCESSFUL ATTACKS ───
    if "unsafe_compliance" in df.columns:
        successful = df[df["unsafe_compliance"] == True]
        if len(successful) > 0:
            lines.append("")
            lines.append("  " + "─" * 60)
            lines.append(f"  SAMPLE SUCCESSFUL ATTACKS ({len(successful)} total)")
            lines.append("  " + "─" * 60)
            
            for idx, row in successful.head(5).iterrows():
                lines.append(f"\n  Attack #{idx + 1}:")
                lines.append(f"    Seed:      {str(row.get('seed_instruction', ''))[:80]}")
                lines.append(f"    Strategy:  {row.get('strategy', 'unknown')}")
                lines.append(f"    Attack:    {str(row.get('test_prompt', ''))[:100]}")
                lines.append(f"    Response:  {str(row.get('blue_response', ''))[:100]}...")
                lines.append(f"    Judge:     unsafe_compliance=True")
    
    # ─── SAMPLE REFUSALS ───
    if "refused" in df.columns:
        refused_df = df[df["refused"] == True]
        if len(refused_df) > 0:
            lines.append("")
            lines.append("  " + "─" * 60)
            lines.append(f"  SAMPLE REFUSALS ({len(refused_df)} total)")
            lines.append("  " + "─" * 60)
            
            for idx, row in refused_df.head(3).iterrows():
                lines.append(f"\n  Refusal #{idx + 1}:")
                lines.append(f"    Seed:      {str(row.get('seed_instruction', ''))[:80]}")
                lines.append(f"    Strategy:  {row.get('strategy', 'unknown')}")
                lines.append(f"    Response:  {str(row.get('blue_response', ''))[:100]}...")
    
    # ─── CONCLUSION ───
    lines.append("")
    lines.append("  " + "═" * 60)
    if asr >= 75:
        lines.append("  CONCLUSION: 🏆 RED TEAM WINS — High ASR achieved!")
    elif asr >= 50:
        lines.append("  CONCLUSION: ⚠ COMPETITIVE — Moderate ASR, needs improvement")
    elif asr >= 25:
        lines.append("  CONCLUSION: ○ BLUE TEAM ADVANTAGE — Low ASR")
    else:
        lines.append("  CONCLUSION: ✗ BLUE TEAM WINS — Red team attacks mostly failed")
        lines.append("")
        lines.append("  RECOMMENDATION: Try these improvements:")
        lines.append("    1. Use a less safety-aligned blue model (smaller/uncensored)")
        lines.append("    2. Fine-tune the red model: python finetune_red_agent.py")
        lines.append("    3. Use more strategies: python execute_pipeline.py --strats 10")
        lines.append("    4. Try different red model: --red-model dolphin-llama3:latest")
    
    lines.append("  " + "═" * 60)
    
    # Print everything
    report = "\n".join(lines)
    print(report)
    
    # Export to text file
    if export_txt:
        txt_path = "results/ASR_Summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  📄 Text report saved to: {txt_path}")
    
    # Export strategy summary as CSV 
    if export_csv and "strategy" in df.columns:
        summary_path = "results/redteam_summary.csv"
        pd.DataFrame(strategy_data).to_csv(summary_path, index=False)
        print(f"  📊 Strategy summary saved to: {summary_path}")
    
    return report


def read_excel_as_text(excel_path="results/ASR_Research_Report.xlsx"):
    """Read the Excel file and display all sheets as text."""
    
    if not os.path.exists(excel_path):
        print(f"Excel file not found: {excel_path}")
        return
    
    print("\n" + "=" * 70)
    print("  EXCEL REPORT CONTENTS (text view)")
    print("=" * 70)
    
    try:
        # Read all sheets from the Excel file
        xlsx = pd.ExcelFile(excel_path)
        sheet_names = xlsx.sheet_names
        
        for sheet_name in sheet_names:
            df_sheet = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            print(f"\n  ┌─ Sheet: {sheet_name} ─┐")
            print(f"  │ Rows: {len(df_sheet)}, Columns: {len(df_sheet.columns)}")
            print(f"  └{'─' * (len(sheet_name) + 12)}┘")
            
            if df_sheet.empty:
                print("  (empty)")
                continue
            
            # Display as formatted text table
            print(df_sheet.to_string(index=False, max_colwidth=50))
            print()
        
    except Exception as e:
        print(f"  Could not read Excel file: {e}")
        print(f"  Tip: pip install openpyxl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View ASR Results as Text")
    parser.add_argument("--export", action="store_true", 
                        help="Export results to results/ASR_Summary.txt")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export strategy summary to results/redteam_summary.csv")
    parser.add_argument("--excel", action="store_true",
                        help="Also show the Excel report contents as text")
    parser.add_argument("--csv", type=str, default="results/final_redteam_results.csv",
                        help="Path to the results CSV")
    
    args = parser.parse_args()
    
    # Show main results
    view_results(args.csv, export_txt=args.export, export_csv=args.export_csv)
    
    # Optionally show Excel contents
    if args.excel:
        read_excel_as_text()
    
    print("\n  💡 To export everything for documentation:")
    print("     python view_results.py --export --export-csv --excel")
