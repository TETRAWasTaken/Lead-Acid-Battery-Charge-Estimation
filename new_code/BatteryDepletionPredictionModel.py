import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

MODEL1_PATH = "../Final_codes/battery_random_forest_model1.joblib"
MODEL2_PATH = "../Final_codes/battery_random_forest_model2.joblib"
TIME_STEPS = 10

# --- Simulation Parameters ---
SIM_INTERVAL_MINUTES = 5.0
SIM_INTERVAL_HOURS = SIM_INTERVAL_MINUTES / 60.0
DEGRADATION_RATE_PER_INTERVAL = 0.000005
MAX_OPERATIONAL_VOLTAGE = 12.6
MIN_OPERATIONAL_VOLTAGE = 9.4
MAX_SIMULATED_CURRENT = 15.0

try:
    model1 = joblib.load(MODEL1_PATH)
    model2 = joblib.load(MODEL2_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found. Please check paths: \n{MODEL1_PATH}\n{MODEL2_PATH}")
    exit() # Or handle as appropriate
except Exception as e:
    print(f"Error loading models: {e}")
    exit()


battery_type_capacities = {
    "B1":81.28,
    "B2":85.0,
    "B3":88.35,
    "TN1":85.0,
    "B5":85.0
}

def simulate_battery_values_revised(
    current_simulation_step,
    cumulative_ah_discharged_start_of_step,
    capacity,
    charged_ah_feature
):
    # 1. Capacity Degradation
    degradation_multiplier = 1.0 - (DEGRADATION_RATE_PER_INTERVAL * current_simulation_step)
    current_effective_capacity = capacity * max(0.20, degradation_multiplier)

    # 2. Simulate Current Draw (using consistent 5-minute interval logic)
    intervals_per_24h = (24 * 60) / SIM_INTERVAL_MINUTES
    time_of_day_cycle_position = (current_simulation_step % intervals_per_24h) / intervals_per_24h

    # Sinusoidal current pattern with some noise
    base_current_value = 8 + 7 * np.sin(2 * np.pi * time_of_day_cycle_position) # e.g. 8A avg, 15A peak, 1A min
    simulated_current = min(MAX_SIMULATED_CURRENT, base_current_value + np.random.normal(0, 0.3)) # Reduced noise
    simulated_current = max(0.1, simulated_current) # Ensure current is always discharging a bit

    # 3. Amp-hours for this interval
    ah_discharged_this_interval = simulated_current * SIM_INTERVAL_HOURS

    # 4. Update Cumulative Amp-hours Discharged
    total_cumulative_ah_discharged = cumulative_ah_discharged_start_of_step + ah_discharged_this_interval

    # 5. Calculate Remaining Capacity
    if current_effective_capacity > 0:
        current_rc = 1.0 - (total_cumulative_ah_discharged / current_effective_capacity)
    else:
        current_rc = 0.0
    current_rc = max(0.0, min(1.0, current_rc))

    # 6. Calculate Voltage
    simulated_voltage = MIN_OPERATIONAL_VOLTAGE + (MAX_OPERATIONAL_VOLTAGE - MIN_OPERATIONAL_VOLTAGE) * current_rc
    simulated_voltage += np.random.normal(0, 0.03)
    simulated_voltage = max(MIN_OPERATIONAL_VOLTAGE - 0.2, simulated_voltage)
    simulated_voltage = min(MAX_OPERATIONAL_VOLTAGE + 0.2, simulated_voltage)

    # 7. Calculate Power
    simulated_power = simulated_voltage * simulated_current

    # 8. Remaining Capacity Percentage
    remaining_capacity_percent = current_rc * 100.0

    return (
        simulated_current,
        simulated_voltage,
        ah_discharged_this_interval,
        total_cumulative_ah_discharged,
        simulated_power,
        remaining_capacity_percent,
        charged_ah_feature
    )

# --- Main Simulation Loop ---
battery_type_index = int(input("Enter battery type index (0-B1,1-B2,2-B3,3-TN1,4-B5): "))

TIME_STEP = 0
cumulative_ah_total = 0.0

data_buffer = []
prediction_timestamps = []
prediction_values = []

battery_keys_list = list(battery_type_capacities.keys())
if battery_type_index >= len(battery_keys_list):
    print(f"Warning: battery_type_index {battery_type_index} out of bounds. Defaulting to 0.")
    battery_type_index = 0
current_battery_code = battery_keys_list[battery_type_index]
capacity = float(battery_type_capacities[current_battery_code])


charged_ah = int(input("Enter charged Ah value (0-100): "))
if charged_ah < 0 or charged_ah > 100:
    print("Warning: charged Ah value out of bounds. Defaulting to 0.")
    charged_ah = 0
if charged_ah > 100:
    charged_ah = 100


print(f"Starting simulation for: {current_battery_code} (Nominal Capacity: {capacity} Ah)")
simulation_start_time = datetime.now()



for i in range(600):
    TIME_STEP += 1

    current, voltage, ah_out, updated_cumulative_ah, power, remaining_perc, charged_ah_val = \
        simulate_battery_values_revised(
            current_simulation_step=(TIME_STEP - 1),
            cumulative_ah_discharged_start_of_step=cumulative_ah_total,
            capacity=capacity,
            charged_ah_feature=charged_ah
        )

    cumulative_ah_total = updated_cumulative_ah

    if voltage < MIN_OPERATIONAL_VOLTAGE:
        print(f"INFO: Voltage ({voltage:.2f}V) fell below cutoff ({MIN_OPERATIONAL_VOLTAGE:.2f}V) at step {TIME_STEP}. Stopping.")
        break
    if remaining_perc < 0.1:
        print(f"INFO: Remaining capacity ({remaining_perc:.2f}%) is critical at step {TIME_STEP}. Stopping.")
        break

    base_row = {
        'Current': float(current),
        'Voltage': float(voltage),
        'Ah Out': float(ah_out),
        'Cumulative Actual Disch Ah': float(cumulative_ah_total),
        'Power': float(power),
        'Remaining Capacity': float(remaining_perc),
        'type': current_battery_code,
        'capacity': float(capacity),
        'charged': float(charged_ah_val),
    }

    data_buffer.append(base_row)

    if len(data_buffer) >= TIME_STEPS:
        X_input_df = pd.DataFrame(data_buffer)

        try:
            pred_model1 = model1.predict(X_input_df)
            X_input_df_for_model2 = X_input_df.copy()

            if isinstance(pred_model1, np.ndarray) and len(pred_model1) > 0 :
                 X_input_df_for_model2['prediction'] = pred_model1[-1]
            else:
                 X_input_df_for_model2['prediction'] = pred_model1

            y_pred_model2 = model2.predict(X_input_df_for_model2)
            time_remaining_seconds = y_pred_model2[-1]

            '''
            print(f"--- Step {TIME_STEP} ---")
            print(f"⚡ Current (A) {current:.2f}")
            print(f"🔋 Voltage (V) {voltage:.2f}")
            h = int(time_remaining_seconds // 3600)
            m = int((time_remaining_seconds % 3600) // 60) # Corrected from original: % 60 to (val % 3600) // 60
            s = int(time_remaining_seconds % 60)
            print(f"⏳ Remaining Time {h:02d} hr(s) : {m:02d} min(s) : {s:02d} sec(s)")
            run_duration = datetime.now() - simulation_start_time
            print(f"🕒 Running Time {str(run_duration).split('.')[0]}")
            '''
            prediction_timestamps.append(TIME_STEP)
            prediction_values.append(time_remaining_seconds)

            data_buffer.pop(0)

        except Exception as e:
            print(f"ERROR during model prediction at step {TIME_STEP}: {e}")
            if data_buffer:
                data_buffer.pop(0)
            pass

    # Periodic status update
    if TIME_STEP % 50 == 0:
        print(f"Progress: Step {TIME_STEP}, V={voltage:.2f}, SoC={remaining_perc:.1f}%")

        # --- Plotting Results ---
        if prediction_values:
            plt.figure(figsize=(10, 6))
            # Ensure proper stepping for x-axis labels (1, 2, 3, ...)
            plt.plot(range(1, len(prediction_values) + 1), np.array(prediction_values) / 3600.0, marker='.', linestyle='-')
            for i, (step, remaining_time) in enumerate(zip(range(1, len(prediction_values) + 1), prediction_values)):
                remaining_hours = remaining_time / 3600.0
                plt.text(step, remaining_hours, f"{remaining_hours:.1f}h", fontsize=8, ha='center', va='bottom')
            plt.xlabel(f"Simulation Step (1, 2, 3, ...)")
            plt.ylabel("Predicted Remaining Useful Life (Hours)")
            plt.title(f"Battery Life Prediction for {current_battery_code} (Charged Ah: {charged_ah})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No predictions were generated to plot. Simulation may have ended early or buffer not filled.")

        print(f"\nSimulation for {current_battery_code} finished at step {TIME_STEP}.")