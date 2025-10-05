import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# --- Complete Battery Metadata ---
BATTERY_METADATA = {
    "TEST_1_processed":  {"capacity": 85,    "charged": 85,    "type": "b5"},
    "TEST_2_processed":  {"capacity": 81.28, "charged": 81.28, "type": "b1"},
    "TEST_3_processed":  {"capacity": 85,    "charged": 85,    "type": "b5"},
    "TEST_4_processed":  {"capacity": 85,    "charged": 85,    "type": "b2"},
    "TEST_5_processed":  {"capacity": 88.81, "charged": 88.81, "type": "b2"},
    "TEST_6_processed":  {"capacity": 81.84, "charged": 81.84, "type": "b1"},
    "TEST_7_processed":  {"capacity": 81.84, "charged": 36,    "type": "b1"},
    "TEST_8_processed":  {"capacity": 88.81, "charged": 27,    "type": "b2"},
    "TEST_9_processed":  {"capacity": 85,    "charged": 80,    "type": "tn1"},
    "TEST_10_processed": {"capacity": 85,    "charged": 54,    "type": "tn1"},
    "TEST_11_processed": {"capacity": 85,    "charged": 85,    "type": "b5"},
    "TEST_12_processed": {"capacity": 85,    "charged": 67,    "type": "b5"},
    "TEST_13_processed": {"capacity": 85,    "charged": 85,    "type": "b5"},
    "TEST_14_processed": {"capacity": 88.83, "charged": 52,    "type": "b3"},
    "TEST_15_processed": {"capacity": 88.35, "charged": 70,    "type": "b3"},
    "TEST_16_processed": {"capacity": 88.35, "charged": 61,    "type": "b3"},
    "TEST_17_processed": {"capacity": 88.35, "charged": 88.35, "type": "b3"},
}

# Define model paths relative to the script's location
MODEL1_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "battery_random_forest_model1.joblib")
MODEL2_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "battery_random_forest_model2.joblib")

TIME_STEPS = 10 # Number of past data points to consider for prediction

SIM_INTERVAL_MINUTES = 5.0
SIM_INTERVAL_HOURS = SIM_INTERVAL_MINUTES / 60.0
DEGRADATION_RATE_PER_INTERVAL = 0.000005 # Simulates battery degradation over time
MAX_OPERATIONAL_VOLTAGE = 12.6
MIN_OPERATIONAL_VOLTAGE = 9.4
MAX_SIMULATED_CURRENT = 15.0

def simulate_battery_values_revised(
    current_simulation_step,
    cumulative_ah_discharged_start_of_step,
    capacity,
    charged_ah_feature
):
    """
    Simulates battery behavior (current, voltage, power, remaining capacity)
    based on a simplified discharge model with some degradation and noise.
    """
    # Simulate capacity degradation
    degradation_multiplier = 1.0 - (DEGRADATION_RATE_PER_INTERVAL * current_simulation_step)
    current_effective_capacity = capacity * max(0.20, degradation_multiplier) # Capacity doesn't go below 20% due to degradation

    # Simulate fluctuating current based on a daily cycle
    intervals_per_24h = (24 * 60) / SIM_INTERVAL_MINUTES
    time_of_day_cycle_position = (current_simulation_step % intervals_per_24h) / intervals_per_24h

    base_current_value = 8 + 7 * np.sin(2 * np.pi * time_of_day_cycle_position)
    simulated_current = min(MAX_SIMULATED_CURRENT, base_current_value + np.random.normal(0, 0.3))
    simulated_current = max(0.1, simulated_current) # Ensure current is always positive

    # Calculate Ah discharged and update cumulative
    ah_discharged_this_interval = simulated_current * SIM_INTERVAL_HOURS
    total_cumulative_ah_discharged = cumulative_ah_discharged_start_of_step + ah_discharged_this_interval

    # Calculate remaining capacity percentage (RC)
    if current_effective_capacity > 0:
        current_rc = 1.0 - (total_cumulative_ah_discharged / current_effective_capacity)
    else:
        current_rc = 0.0
    current_rc = max(0.0, min(1.0, current_rc)) # Clamp RC between 0 and 1

    # Simulate voltage based on remaining capacity with some noise
    simulated_voltage = MIN_OPERATIONAL_VOLTAGE + (MAX_OPERATIONAL_VOLTAGE - MIN_OPERATIONAL_VOLTAGE) * current_rc
    simulated_voltage += np.random.normal(0, 0.03)
    simulated_voltage = max(MIN_OPERATIONAL_VOLTAGE - 0.2, simulated_voltage)
    simulated_voltage = min(MAX_OPERATIONAL_VOLTAGE + 0.2, simulated_voltage)

    simulated_power = simulated_voltage * simulated_current
    remaining_capacity_percent = current_rc * 100.0

    return (
        simulated_current,
        simulated_voltage,
        ah_discharged_this_interval,
        total_cumulative_ah_discharged,
        simulated_power,
        remaining_capacity_percent,
        charged_ah_feature # This feature is simply passed through
    )

def main():
    # Load models and their respective training feature columns
    try:
        # Verify model file existence
        if not os.path.exists(MODEL1_PATH):
            print(f"Error: Model 1 file not found at {MODEL1_PATH}")
            return
        if not os.path.exists(MODEL2_PATH):
            print(f"Error: Model 2 file not found at {MODEL2_PATH}")
            return

        # Load models and their feature lists
        # The training script saves a tuple (pipeline, list_of_feature_names)
        model1, model1_raw_feature_cols = joblib.load(MODEL1_PATH)
        model2, model2_raw_feature_cols = joblib.load(MODEL2_PATH)

        # Basic check to ensure loaded objects are indeed models
        if not hasattr(model1, 'predict') or not hasattr(model2, 'predict'):
            print("Error: Loaded objects are not valid machine learning models. "
                  "Please ensure your .joblib files contain scikit-learn models (Pipeline objects).")
            return

    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure 'Preprocessing and Training.py' was run successfully and model files exist.")
        return

    # Group batteries by type from metadata
    battery_types = defaultdict(list)
    for test, data in BATTERY_METADATA.items():
        battery_types[data['type']].append({
            'test_name': test,
            'capacity': data['capacity'],
            'max_charge': data['charged']
        })

    # --- User Input for Battery Selection ---
    print("Available battery types:")
    types = sorted(battery_types.keys())
    for i, t in enumerate(types):
        print(f"{i+1}. {t.upper()}")
    type_choice_idx = int(input("\nSelect battery type (number): ")) - 1
    if not (0 <= type_choice_idx < len(types)):
        print("Invalid selection. Using first available type.")
        type_choice_idx = 0
    selected_type = types[type_choice_idx]

    type_batteries = battery_types[selected_type]
    print(f"\nAvailable batteries for {selected_type.upper()}:")
    for i, batt in enumerate(type_batteries):
        print(f"{i+1}. {batt['test_name']} | {batt['capacity']}Ah (Max charge: {batt['max_charge']}Ah)")
    cap_choice = int(input("\nSelect battery (number): ")) - 1
    if not (0 <= cap_choice < len(type_batteries)):
        print("Invalid selection. Using first battery for the chosen type.")
        cap_choice = 0
    selected_battery = type_batteries[cap_choice]
    capacity = selected_battery['capacity']
    max_charge = selected_battery['max_charge']

    # User selects initial charged level
    charged_ah = float(input(f"\nEnter initial charged Ah (0-{max_charge:.2f}): "))
    if charged_ah > max_charge:
        print(f"Charged value too high, using max {max_charge:.2f}Ah.")
        charged_ah = max_charge
    elif charged_ah < 0:
        print("Charged value too low, using 0Ah.")
        charged_ah = 0.0

    print(f"\nSimulating for type: {selected_type.upper()}, battery: {selected_battery['test_name']}, capacity: {capacity:.2f}Ah, charged: {charged_ah:.2f}Ah")

    # --- Simulation Initialization ---
    TIME_STEP = 0 # Represents the current discrete simulation step
    cumulative_ah_total = capacity - charged_ah  # Initial discharged Ah based on charge level
    data_buffer = [] # Stores recent historical data for prediction
    prediction_timestamps = [] # Stores time steps when predictions were made
    prediction_values = [] # Stores predicted 'Time to Depletion'
    remaining_capacity_percent_list = [] # Stores simulated 'Remaining Capacity'

    # --- Main Simulation Loop ---
    for _ in range(1200): # Simulate up to 1200 steps (100 hours at 5-min intervals)
        TIME_STEP += 1

        # Simulate current battery values
        current, voltage, ah_out, updated_cumulative_ah, power, remaining_perc, _ = \
            simulate_battery_values_revised(
                current_simulation_step=(TIME_STEP - 1), # (TIME_STEP - 1) for degradation calc from step 0
                cumulative_ah_discharged_start_of_step=cumulative_ah_total,
                capacity=capacity,
                charged_ah_feature=charged_ah
            )

        cumulative_ah_total = updated_cumulative_ah

        # Check for critical conditions to stop simulation
        if voltage < MIN_OPERATIONAL_VOLTAGE - 0.5: # A bit lower threshold to prevent premature stop
            print(f"Voltage critical ({voltage:.2f}V) at step {TIME_STEP} (approx {TIME_STEP * SIM_INTERVAL_MINUTES / 60:.2f} hours)")
            break
        if remaining_perc < 0.1 and TIME_STEP > TIME_STEPS: # Stop if capacity is near zero, but only after enough buffer data
            print(f"Capacity critical ({remaining_perc:.2f}%) at step {TIME_STEP} (approx {TIME_STEP * SIM_INTERVAL_MINUTES / 60:.2f} hours)")
            break
        if remaining_perc <= 0.01 and TIME_STEP > TIME_STEPS:
            print(f"Capacity fully depleted ({remaining_perc:.2f}%) at step {TIME_STEP} (approx {TIME_STEP * SIM_INTERVAL_MINUTES / 60:.2f} hours)")
            break

        # Calculate engineered features for the current step (must match training script)
        discharge_rate = current / (voltage + 1e-6)
        discharge_ratio = ah_out / (charged_ah + 1e-6)
        step_index = TIME_STEP # Using current simulation step as step_index

        # Create a dictionary for the current step's features
        current_step_features = {
            'Current': float(current),
            'Voltage': float(voltage),
            'Ah Out': float(ah_out),
            'Cumulative Actual Disch Ah': float(cumulative_ah_total),
            'Power': float(power),
            'Remaining Capacity': float(remaining_perc),
            'capacity': float(capacity),
            'charged': float(charged_ah),
            'discharge_rate': float(discharge_rate),
            'step_index': float(step_index),
            'discharge_ratio': float(discharge_ratio),
            'type': selected_type # Pass the raw 'type' string for the pipeline to encode
        }

        # --- Data Validation (CRUCIAL for 'ufunc isnan' error) ---
        for key, value in current_step_features.items():
            if isinstance(value, (float, int)) and (np.isnan(value) or np.isinf(value)):
                print(f"Warning: Detected NaN/Inf in feature '{key}': {value} at step {TIME_STEP}")
                # You might want to handle this more robustly, e.g., impute or log.
                # For now, let's replace with 0.0 to allow pipeline to process, but this can affect predictions.
                current_step_features[key] = 0.0 # Default fallback, consider better imputation if NaNs are expected

        data_buffer.append(current_step_features)

        # --- Prediction Logic ---
        # Only make predictions once enough historical data points are collected
        if len(data_buffer) >= TIME_STEPS:
            # Create DataFrame from the buffer
            # IMPORTANT: Reindex the DataFrame using the *raw* feature columns used for training.
            # The pipeline handles the preprocessing (scaling, OHE) internally.
            X_input_df_model1 = pd.DataFrame(data_buffer).reindex(columns=model1_raw_feature_cols)

            try:
                # Predict with Model 1 (Depletion Ratio)
                # Model 1 is a Pipeline, it expects the raw feature columns
                pred_model1_ratio = model1.predict(X_input_df_model1)

                # Prepare DataFrame for Model 2 by adding Model 1's prediction
                # Ensure the 'prediction' column is added correctly for model2's input
                X_input_df_model2_raw = pd.DataFrame(data_buffer).reindex(columns=model2_raw_feature_cols)
                
                # Add the prediction from model1 to the dataframe for model2.
                # Take the prediction corresponding to the *last* data point in the buffer.
                X_input_df_model2_raw['prediction'] = pred_model1_ratio[-1] 
                
                # Reindex for model2. The model2_raw_feature_cols should *include* 'prediction' if it was
                # trained with it, or you might need to manually ensure order if it's dynamic.
                # Check your training script RandomForest2: `features_for_model` for model2 should include 'prediction'.
                # Assuming 'prediction' is one of the features in model2_raw_feature_cols:
                X_input_df_model2 = X_input_df_model2_raw.reindex(columns=model2_raw_feature_cols)


                # Predict with Model 2 (Time to Depletion in seconds)
                y_pred_model2_seconds = model2.predict(X_input_df_model2)
                time_remaining_seconds = y_pred_model2_seconds[-1] # Get prediction for the current latest step

                # Store predictions and actual remaining capacity for plotting
                prediction_timestamps.append(TIME_STEP)
                prediction_values.append(time_remaining_seconds)
                remaining_capacity_percent_list.append(remaining_perc)

                # Remove the oldest data point to maintain buffer size for the next iteration
                data_buffer.pop(0)

            except Exception as e:
                print(f"Prediction error at step {TIME_STEP}: {e}")
                # Still remove oldest data point even on error to prevent buffer from growing indefinitely
                if data_buffer:
                    data_buffer.pop(0)

        # --- Plotting Logic ---
        # Plot every 50 simulation steps if predictions are available
        if TIME_STEP % 50 == 0 and prediction_values:
            plt.figure(figsize=(12, 7))
            time_remaining_hours = np.array(prediction_values) / 3600.0 # Convert seconds to hours

            # Plotting predicted time remaining vs. actual remaining capacity
            plt.plot(time_remaining_hours, remaining_capacity_percent_list,
                     marker='o', markersize=6, linestyle='-', linewidth=1.5, color='#3498db')

            plt.gca().invert_xaxis() # Invert x-axis to show time decreasing
            # Dynamically adjust x-axis limits to show relevant range
            if len(time_remaining_hours) > 0:
                max_x = max(time_remaining_hours[0] + 0.5, 8) # Ensure at least 8 hours for initial view
                min_x = max(0, time_remaining_hours[-1] - 0.5)
                plt.xlim(max_x, min_x)
            else:
                 plt.xlim(8, 0) # Default if no data yet

            plt.ylim(0, 100) # Y-axis for percentage
            plt.yticks(np.arange(0, 101, 10)) # Set Y-axis ticks
            plt.grid(True, alpha=0.3)
            plt.title(f"Battery Discharge Simulation: {selected_type.upper()} | {selected_battery['test_name']} | {capacity:.2f}Ah | Charged: {charged_ah:.2f}Ah")
            plt.xlabel("Predicted Time Remaining (hours)")
            plt.ylabel("Remaining Capacity (%)")

            # Annotate a few points on the plot for clarity
            step_interval_for_annotations = max(1, len(time_remaining_hours) // 5)
            for i in range(0, len(time_remaining_hours), step_interval_for_annotations):
                 t_rem = time_remaining_hours[i]
                 cap = remaining_capacity_percent_list[i]
                 plt.text(t_rem, cap, f"({t_rem:.1f}h, {cap:.1f}%)",
                          ha='center', va='bottom', fontsize=8, color='darkred')

            plt.tight_layout()
            plt.show()

    print(f"\nSimulation completed for {selected_type.upper()} ({selected_battery['test_name']}, {capacity:.2f}Ah, {charged_ah:.2f}Ah charged) after {TIME_STEP} steps")

if __name__ == "__main__":
    main()