from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.fast.policies import ResidualActor, ResidualSACPolicy
from stable_baselines3.fast.fast import FAST

__all__ = ["FAST", "CnnPolicy", "MlpPolicy", "MultiInputPolicy", "ResidualActor", "ResidualSACPolicy"]
