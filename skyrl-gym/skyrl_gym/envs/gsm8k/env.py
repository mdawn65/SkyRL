from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any
from omegaconf import DictConfig
import re


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
        
        # Basic response metrics
        response_length = len(action)
        word_count = len(action.split())
        
        # Answer accuracy (based on reward)
        answer_accuracy = float(reward > 0)
        
        # Format compliance - check if response follows #### format
        has_final_answer_format = bool(re.search(r"#### (\-?[0-9\.\,]+)", action))
        
        # Extract the model's answer using both strict and flexible methods
        strict_answer = utils.extract_solution(action, method="strict")
        flexible_answer = utils.extract_solution(action, method="flexible")
        
        # Check if model provided any numerical answer
        has_numerical_answer = (strict_answer is not None) or (flexible_answer is not None)
        
        # Count mathematical operations and reasoning indicators
        math_operations = len(re.findall(r'[+\-*/=]', action))
        contains_calculation = math_operations > 0
        
        # Count reasoning steps (sentences or numbered steps)
        reasoning_steps = max(
            len(re.findall(r'\d+\.\s', action)),  # numbered steps like "1. "
            len(re.findall(r'[.!?]+', action))    # sentences
        )
        
        # Check for confidence indicators
        uncertainty_phrases = ['maybe', 'probably', 'might', 'could be', 'not sure', 'think']
        confidence_indicators = sum(1 for phrase in uncertainty_phrases if phrase.lower() in action.lower())
        
        # Check for step-by-step reasoning
        has_step_by_step = bool(re.search(r'(step|first|then|next|finally)', action, re.IGNORECASE))
        
        return {
            # Core accuracy metrics
            "answer_accuracy": answer_accuracy,
            "has_numerical_answer": float(has_numerical_answer),
            "format_compliance": float(has_final_answer_format),
            
            # Response quality metrics
            "response_length": response_length,
            "word_count": word_count,
            "reasoning_steps": reasoning_steps,
            
            # Mathematical reasoning metrics
            "math_operations_count": math_operations,
            "contains_calculation": float(contains_calculation),
            "has_step_by_step_reasoning": float(has_step_by_step),
            
            # Confidence and uncertainty metrics
            "uncertainty_indicators": confidence_indicators,
            
            # Method comparison (useful for understanding model behavior)
            "strict_method_success": float(strict_answer is not None),
            "flexible_method_success": float(flexible_answer is not None),
        }

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        
        # Calculate comprehensive metrics for training insights
        metrics = self._calculate_metrics(action, reward)

        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(
            observations=[], 
            reward=reward, 
            done=done, 
            metadata={},
            metrics=metrics  # ← Added metrics for TODO 3!
        )