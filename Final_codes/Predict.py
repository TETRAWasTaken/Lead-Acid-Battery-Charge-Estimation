import joblib
import pandas as pd

class prediction():
    def __init__(self):
        self.model1_path = "../Final_codes/battery_random_forest_model1.joblib"
        self.model2_path = "../Final_codes/battery_random_forest_model2.joblib"
        self.model1 = joblib.load(self.model1_path)
        self.model2 = joblib.load(self.model2_path)

        self.metadata = {
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

    def predict_model1(self, data):
        Y_pred = self.model1.predict(data)
        return self.predict_model2(Y_pred, data)

    def predict_model2(self, Y_pred, data):
        type_encoding = {'b1': 0, 'b2': 1, 'b3': 2, 'b5': 3, 'tn1': 4}

        # If Y_pred is 2D with 2 columns, assume [TOD, CDC]
        if Y_pred.ndim == 2 and Y_pred.shape[1] == 2:
            pred_TOD = Y_pred[:, 0]
            pred_CDC = Y_pred[:, 1]
        else:
            pred_TOD = Y_pred
            pred_CDC = [0] * len(Y_pred)

        data = data.copy()
        data['predicted TOD'] = pred_TOD
        data['predicted CDC'] = pred_CDC

        # Map battery type to encoded int if string
        if 'type' in data.columns and data['type'].dtype == object:
            data['type'] = data['type'].map(type_encoding)

        final_output = self.model2.predict(data)
        return final_output

    def feature_derivator(self, voltage, current, battery_type, ini_charge):
        time_step = 1  # seconds
        ah_out = (abs(current) * time_step) / 3600

        battery_capacity = None
        for meta in self.metadata.values():
            if meta['type'] == battery_type:
                battery_capacity = meta['capacity']
                break
        if battery_capacity is None:
            battery_capacity = ini_charge  # fallback

        power = voltage * current
        remaining_capacity = battery_capacity - ah_out
        discharge_rate = abs(current) / battery_capacity
        discharge_ratio = ah_out / battery_capacity

        # Use BATTERY_TYPE_MAPPING to encode battery type as int
        type_encoding = {'tn1': 0, 'b1': 1, 'b2': 2, 'b3': 3, 'b5': 4}
        df = pd.DataFrame({
            'Voltage': [float(voltage)],
            'Current': [float(current)],
            'Ah Out': [float(ah_out)],
            'Power': [float(power)],
            'Remaining Capacity': [float(remaining_capacity)],
            'type': [type_encoding[battery_type]],
            'capacity': [float(battery_capacity)],
            'charged': [float(ini_charge)],
            'discharge_rate': [float(discharge_rate)],
            'discharge_ratio': [float(discharge_ratio)]
        })

        return self.predict_model1(df)
