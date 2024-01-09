# model_handling.py

import joblib
import numpy as np
from stable_baselines3 import PPO

class ModelHandler:
    def __init__(self, model_path, scaler_path):
        self.model = PPO.load(model_path)
        self.scaler = joblib.load(scaler_path)


    def prepare_observation(self, df_enhanced, long_position, short_position, current_balance, stop_loss_flag, drawdown, open_pnl):
        live_features_scaled = self.scaler.transform(df_enhanced)
        obs = np.hstack((live_features_scaled[0], [long_position, short_position, current_balance, stop_loss_flag, drawdown, open_pnl]))   
        return obs


    def predict_action(self, observation):
        action, _states = self.model.predict(observation)
        return action

