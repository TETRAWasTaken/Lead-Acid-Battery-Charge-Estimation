import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Predict
import itertools

# --- Configuration ---
BATTERY_TYPE_MAPPING = {'tn1': 0, 'b1': 1, 'b2': 2, 'b3': 3, 'b5': 4}
OUTPUT_DIR = "./predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Predictor = Predict.prediction()

def best_fit_line(x, y):
    x = np.array(x)
    y = np.array(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.full_like(x, np.nan)
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    return poly1d_fn(x), coef

def process_all_by_type(folder_path="../data/processed", output_dir=OUTPUT_DIR):
    results_by_type = {}
    meta_by_type = {}
    testnames_by_type = {}

    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files to process")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        filename = os.path.basename(csv_path)
        key = filename.replace('.csv', '')
        meta = Predictor.metadata.get(key)
        if not meta:
            print(f"No metadata found for {filename}")
            continue
        battery_type = meta['type'].lower()
        type_code = BATTERY_TYPE_MAPPING.get(battery_type)
        if type_code is None:
            print(f"Unknown battery type for {filename}")
            continue

        print(f"Processing: {filename} (Type: {meta['type']}, Charged: {meta['charged']}Ah)")

        dod_vals, tod_vals, capacity_ah, step_indices = [], [], [], []
        for idx, row in df.iterrows():
            features = pd.DataFrame({
                'Current': [row['Current']],
                'Voltage': [row['Voltage']],
                'Ah Out': [row['Ah Out']],
                'Power': [row['Power']],
                'Remaining Capacity': [row['Remaining Capacity']],
                'type': [type_code],
                'capacity': [meta['capacity']],
                'charged': [meta['charged']],
                'discharge_rate': [row['Current'] / (row['Voltage'] + 1e-6)],
                'discharge_ratio': [row['Ah Out'] / (meta['charged'] + 1e-6)]
            })

            pred = Predictor.model1.predict(features)
            if hasattr(pred, 'shape') and pred.shape[1] >= 2:
                predicted_TOD = pred[0][0]
            else:
                predicted_TOD = pred[0]

            tod_hr = predicted_TOD / 3600
            tod_vals.append(tod_hr)
            capacity_ah.append(row['Remaining Capacity'])
            dod = 100 * (meta['charged'] - row['Remaining Capacity']) / meta['charged']
            dod_vals.append(dod)
            step_indices.append(idx / 60)

        # Print battery working time information
        if len(step_indices) > 0:
            total_runtime = max(step_indices)
            initial_tod = max(tod_vals) if tod_vals else 0
            final_tod = min(tod_vals) if tod_vals else 0
            print(f"  ✓ Total Runtime: {total_runtime:.2f} hours")
            print(f"  ✓ Initial Predicted Time Remaining: {initial_tod:.2f} hours")
            print(f"  ✓ Final Predicted Time Remaining: {final_tod:.2f} hours")
            print(f"  ✓ Prediction Range: {initial_tod - final_tod:.2f} hours")

        if battery_type not in results_by_type:
            results_by_type[battery_type] = []
            meta_by_type[battery_type] = meta
            testnames_by_type[battery_type] = []
        results_by_type[battery_type].append({
            "testname": key,
            "step": step_indices,
            "dod": dod_vals,
            "tod": tod_vals,
            "ah": capacity_ah
        })
        testnames_by_type[battery_type].append(key)

    # For each battery type, plot and export
    for battery_type, test_list in results_by_type.items():
        meta = meta_by_type[battery_type]
        
        print(f"\n=== {battery_type.upper()} BATTERY TYPE SUMMARY ===")
        
        # Calculate and print working times for each test
        total_runtime_all = 0
        for test in test_list:
            if len(test["step"]) > 0:
                test_runtime = max(test["step"])
                test_initial_pred = max(test["tod"]) if test["tod"] else 0
                test_final_pred = min(test["tod"]) if test["tod"] else 0
                total_runtime_all += test_runtime
                print(f"  {test['testname']}: Runtime={test_runtime:.2f}h, "
                      f"Pred Range={test_initial_pred:.1f}h→{test_final_pred:.1f}h")
        
        avg_runtime = total_runtime_all / len(test_list) if test_list else 0
        print(f"  Average Runtime: {avg_runtime:.2f} hours")
        print(f"  Number of Tests: {len(test_list)}")
        
        colors = itertools.cycle(plt.cm.tab10.colors)
        # --- Plot 1: DoD (%) vs Time (hours) ---
        plt.figure(figsize=(12, 6))
        all_steps = []
        all_dod = []
        for test in test_list:
            color = next(colors)
            runtime = max(test["step"]) if test["step"] else 0
            plt.plot(test["step"], test["dod"], 
                    label=f"{test['testname']} ({runtime:.1f}h)", color=color)
            # Annotate every 50th point and the last point
            for i in range(0, len(test["step"]), 50):
                plt.annotate(f"{test['dod'][i]:.1f}%", (test["step"][i], test["dod"][i]), 
                           fontsize=8, color=color)
            if len(test["step"]) > 0:
                plt.annotate(f"{test['dod'][-1]:.1f}%", (test["step"][-1], test["dod"][-1]), 
                           fontsize=8, color=color)
            all_steps.extend(test["step"])
            all_dod.extend(test["dod"])
        
        # Best fit line for all data
        best_fit_dod, _ = best_fit_line(all_steps, all_dod)
        plt.plot(all_steps, best_fit_dod, color='red', linewidth=2.5, linestyle='--', label="Best Fit Line")
        plt.xlabel("Time (hours)")
        plt.ylabel("Depth of Discharge (%)")
        plt.ylim(0, 100)
        plt.title(f"{battery_type.upper()} - DoD vs Time (Avg Runtime: {avg_runtime:.1f}h)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{battery_type.upper()}_DoD_vs_Time.png"), dpi=150)
        plt.close()

        # --- Plot 2: Remaining Capacity (Ah) vs Predicted Time Remaining (hours) ---
        plt.figure(figsize=(12, 6))
        all_tod = []
        all_ah = []
        colors = itertools.cycle(plt.cm.tab10.colors)
        for test in test_list:
            color = next(colors)
            runtime = max(test["step"]) if test["step"] else 0
            pred_range = (max(test["tod"]) - min(test["tod"])) if test["tod"] else 0
            plt.plot(test["tod"], test["ah"], marker='o', markersize=3,
                    label=f"{test['testname']} ({runtime:.1f}h, ±{pred_range:.1f}h)", color=color)
            # Annotate every 50th point and the last point
            for i in range(0, len(test["tod"]), 50):
                plt.annotate(f"{test['ah'][i]:.1f}Ah", (test["tod"][i], test["ah"][i]), 
                           fontsize=8, color=color)
            if len(test["tod"]) > 0:
                plt.annotate(f"{test['ah'][-1]:.1f}Ah", (test["tod"][-1], test["ah"][-1]), 
                           fontsize=8, color=color)
            all_tod.extend(test["tod"])
            all_ah.extend(test["ah"])
        
        # Best fit line for all data
        best_fit_ah, _ = best_fit_line(all_tod, all_ah)
        plt.plot(all_tod, best_fit_ah, color='red', linewidth=2.5, linestyle='--', label="Best Fit Line")
        plt.xlabel("Predicted Time Remaining (Hours)")
        plt.ylabel("Remaining Capacity (Ah)")
        plt.title(f"{battery_type.upper()} - Remaining Capacity vs Time Remaining (Avg Runtime: {avg_runtime:.1f}h)")
        plt.ylim(0, meta['charged'])
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{battery_type.upper()}_RemainingAh_vs_TimeRemaining.png"), dpi=150)
        plt.close()

        print(f"Saved plots for {battery_type.upper()}")

if __name__ == "__main__":
    process_all_by_type()
