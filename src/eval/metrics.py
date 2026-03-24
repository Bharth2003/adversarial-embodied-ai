# src/eval/metrics.py
# Enhanced metrics and report generation

import pandas as pd

def export_to_formatted_excel(csv_path, excel_path):
    """
    Converts the final results CSV into a formatted Excel report 
    with ASR and Refusal metrics, broken down by strategy and risk category.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        print("No data found to export.")
        return

    # === Sheet 1: Strategy Summary ===
    strategy_summary = df.groupby(["strategy"]).agg(
        Total_Attacks=("unsafe_compliance", "count"),
        Successful_Attacks=("unsafe_compliance", lambda x: (x == True).sum()),
        Refusals=("refused", lambda x: (x == True).sum() if "refused" in df.columns else 0)
    ).reset_index()
    
    strategy_summary["ASR_%"] = (strategy_summary["Successful_Attacks"] / strategy_summary["Total_Attacks"]) * 100
    strategy_summary["Refusal_%"] = (strategy_summary["Refusals"] / strategy_summary["Total_Attacks"]) * 100
    strategy_summary = strategy_summary.round(2)

    # === Sheet 2: Risk Category Summary ===
    risk_summary = pd.DataFrame()
    if "risk_category" in df.columns:
        risk_summary = df.groupby(["risk_category"]).agg(
            Total=("unsafe_compliance", "count"),
            Unsafe=("unsafe_compliance", lambda x: (x == True).sum()),
        ).reset_index()
        risk_summary["ASR_%"] = (risk_summary["Unsafe"] / risk_summary["Total"]) * 100
        risk_summary = risk_summary.round(2).sort_values("ASR_%", ascending=False)

    # === Sheet 3: Overall Summary ===
    total = len(df)
    unsafe = (df["unsafe_compliance"] == True).sum()
    refused = (df.get("refused", pd.Series([False]*total)) == True).sum()
    
    overall = pd.DataFrame([{
        "Total_Attacks": total,
        "Successful_Attacks (Unsafe Compliance)": unsafe,
        "Refusals": refused,
        "ASR_%": round(unsafe / total * 100, 2) if total > 0 else 0,
        "Refusal_%": round(refused / total * 100, 2) if total > 0 else 0,
        "Red_Model": df.get("red_model", pd.Series(["unknown"])).iloc[0] if "red_model" in df.columns else "unknown",
        "Blue_Model": df.get("blue_model", pd.Series(["unknown"])).iloc[0] if "blue_model" in df.columns else "unknown",
        "Judge_Model": df.get("judge_model", pd.Series(["unknown"])).iloc[0] if "judge_model" in df.columns else "unknown",
    }])

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        overall.to_excel(writer, sheet_name='Overall_Summary', index=False)
        strategy_summary.to_excel(writer, sheet_name='Strategy_Breakdown', index=False)
        if not risk_summary.empty:
            risk_summary.to_excel(writer, sheet_name='Risk_Category_Breakdown', index=False)
        df.to_excel(writer, sheet_name='Item_Level_Results', index=False)

    print(f"--- SUCCESS: Results exported to {excel_path} ---")
    print(f"    Sheets: Overall_Summary, Strategy_Breakdown, Risk_Category_Breakdown, Item_Level_Results")