from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, List, Any
from omegaconf import DictConfig
import numpy as np


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _get_reward(self, action: str) -> float:
        return utils.compute_score(action, self.ground_truth)
    
    def _calculate_metrics(self, action: str, reward: float) -> Dict[str, Any]:
        """Calculate environment-specific metrics for training insights."""
        response_length = len(action)
        word_count = len(action.split())
        contains_boxed = float("\\boxed{" in action)
        return {
            "response_length": response_length,
            "word_count": word_count,
            "contains_boxed": contains_boxed,
        } 
    
    @staticmethod
    def aggregate_response_length(values: List[int]) -> Dict[str, float]:
        """Aggregate response length metrics (continuous integer)."""
        if not values:
            return {}
        values_arr = np.array(values)
        return {
            "response_length/avg": np.mean(values_arr).item(),
            "response_length/min": np.min(values_arr).item(), 
            "response_length/max": np.max(values_arr).item(),
            "response_length/std": np.std(values_arr).item(),
            "response_length/median": np.median(values_arr).item(),
        }

    @staticmethod
    def aggregate_word_count(values: List[int]) -> Dict[str, float]:
        """Aggregate word count metrics (continuous integer)."""
        if not values:
            return {}
        values_arr = np.array(values)
        return {
            "word_count/avg": np.mean(values_arr).item(),
            "word_count/min": np.min(values_arr).item(),
            "word_count/max": np.max(values_arr).item(), 
            "word_count/std": np.std(values_arr).item(),
            "word_count/median": np.median(values_arr).item(),
        }

    @staticmethod
    def aggregate_contains_boxed(values: List[float]) -> Dict[str, float]:
        """Aggregate binary contains_boxed metrics."""
        if not values:
            return {}
        values_arr = np.array(values)
        return {
            "contains_boxed/rate": np.mean(values_arr).item(),           # Success rate (0.0 to 1.0)
            "contains_boxed/count": np.sum(values_arr).item(),           # Total successes
            "contains_boxed/total": len(values_arr),                     # Total attempts
            "contains_boxed/std": np.std(values_arr).item(),             # Variability
        }

    @staticmethod
    def aggregate_environment_metrics(env_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate all GSM8k environment metrics across a batch."""
        if not env_metrics:
            return {}
        
        # Group metrics by key
        metric_groups = {}
        for metrics_dict in env_metrics:
            if metrics_dict:
                for key, value in metrics_dict.items():
                    if key not in metric_groups:
                        metric_groups[key] = []
                    metric_groups[key].append(value)
        
        # Aggregate each metric group using specialized functions
        aggregated = {}
        for key, values in metric_groups.items():
            if not values:
                continue
                
            if key == "response_length":
                aggregated.update(GSM8kEnv.aggregate_response_length(values))
            elif key == "word_count": 
                aggregated.update(GSM8kEnv.aggregate_word_count(values))
            elif key == "contains_boxed":
                aggregated.update(GSM8kEnv.aggregate_contains_boxed(values))
            else:
                # Default fallback for unknown metrics (treat as continuous)
                if values:
                    values_arr = np.array(values)
                    aggregated.update({
                        f"{key}/avg": np.mean(values_arr).item(),
                        f"{key}/min": np.min(values_arr).item(),
                        f"{key}/max": np.max(values_arr).item(),
                        f"{key}/std": np.std(values_arr).item(),
                    })
        
        return aggregated

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        metrics = self._calculate_metrics(action, reward)

        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={}, metrics=metrics)