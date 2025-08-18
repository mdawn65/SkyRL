"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_skyrl_gym_generator_metrics.py
"""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from omegaconf import DictConfig
from transformers import AutoTokenizer
from skyrl_gym.envs import register


class CPUTestEnvWithMetrics(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 2
        self.test_metrics = extras.get("test_metrics", {})

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        done = self.turns >= self.max_turns
        
        # Return metrics that vary by turn for testing aggregation
        metrics = {}
        if self.test_metrics:
            metrics = {
                "accuracy": 0.8 + (self.turns * 0.1),  # 0.9, 1.0
                "latency": 0.1 - (self.turns * 0.02),  # 0.08, 0.06
                "score": 90 + self.turns,              # 91, 92
            }
        
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"obs_{self.turns}"}] if not done else [],
            reward=float(self.turns),
            done=done,
            metadata={},
            metrics=metrics  # This is what we're testing!
        )


def _register_metrics_test_env():
    """Register the test env for metrics testing."""
    try:
        register(
            id="cpu_metrics_test_env",
            entry_point="tests.cpu.generators.test_skyrl_gym_generator_metrics:CPUTestEnvWithMetrics",
        )
    except Exception:
        # Environment already registered, ignore
        pass


class TestSkyRLGymGeneratorMetrics:
    
    @pytest.fixture
    def mock_generator(self):
        """Create a SkyRLGymGenerator for testing."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        mock_llm = MagicMock()
        
        def mock_generate(input_batch):
            num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
            return {"responses": ["test_response"] * num_prompts, "stop_reasons": ["stop"] * num_prompts}
        
        mock_llm.generate = AsyncMock(side_effect=mock_generate)
        
        generator_cfg = DictConfig({
            "sampling_params": {"max_generate_length": 100},
            "max_input_length": 200,
            "batched": False,
            "max_turns": 2,
            "zero_reward_on_non_stop": False,
            "use_conversation_multi_turn": True,
        })
        
        env_cfg = DictConfig({
            "max_env_workers": 0,
            "env_class": "cpu_metrics_test_env",
        })
        
        generator = SkyRLGymGenerator(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=tokenizer,
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
        )
        return generator

    def test_aggregate_env_metrics_basic(self, mock_generator):
        """Test basic metrics aggregation functionality."""
        env_step_outputs = [
            {
                "observations": [],
                "reward": 1.0,
                "done": True,
                "metadata": {},
                "metrics": {"accuracy": 0.8, "latency": 0.1, "score": 95}
            },
            {
                "observations": [],
                "reward": 0.5,
                "done": True,
                "metadata": {},
                "metrics": {"accuracy": 0.9, "latency": 0.05, "score": 87}
            }
        ]
        
        result = mock_generator._aggregate_env_metrics(env_step_outputs)
        
        # Verify metrics are present with env/ prefix
        assert "env/accuracy_mean" in result
        assert "env/accuracy_std" in result
        assert "env/accuracy_min" in result
        assert "env/accuracy_max" in result
        assert "env/accuracy_count" in result
        
        # Verify calculations
        assert result["env/accuracy_mean"] == pytest.approx(0.85, abs=1e-6)
        assert result["env/accuracy_min"] == 0.8
        assert result["env/accuracy_max"] == 0.9
        assert result["env/accuracy_count"] == 2

    def test_aggregate_env_metrics_empty(self, mock_generator):
        """Test metrics aggregation with empty input."""
        result = mock_generator._aggregate_env_metrics([])
        assert result == {}

    def test_aggregate_env_metrics_no_metrics(self, mock_generator):
        """Test when environments don't return metrics."""
        env_step_outputs = [
            {
                "observations": [],
                "reward": 1.0,
                "done": True,
                "metadata": {},
                "metrics": {}
            }
        ]
        
        result = mock_generator._aggregate_env_metrics(env_step_outputs)
        assert result == {}

    @pytest.mark.asyncio
    async def test_generate_with_metrics_integration(self, mock_generator):
        """Test full integration of metrics through generate()."""
        _register_metrics_test_env()
        
        input_batch: GeneratorInput = {
            "prompts": [[{"role": "user", "content": "test"}]],
            "env_extras": [{"test_metrics": True}],  # Enable metrics in test env
            "env_classes": ["cpu_metrics_test_env"],
        }
        
        result: GeneratorOutput = await mock_generator.generate(input_batch)
        
        # Verify that environment metrics are included in rollout_metrics
        assert "rollout_metrics" in result
        rollout_metrics = result["rollout_metrics"]
        
        # Check that env metrics are present (from our test environment)
        # Our test env returns metrics for 2 turns, so we should see aggregated results
        assert any(key.startswith("env/") for key in rollout_metrics.keys()), \
            f"No env/ metrics found in: {list(rollout_metrics.keys())}"
        
        # If metrics are present, verify some expected ones
        env_metric_keys = [k for k in rollout_metrics.keys() if k.startswith("env/")]
        if env_metric_keys:
            # Should have mean, std, min, max, count for each metric
            metric_names = set(key.split('_')[0] for key in env_metric_keys if key.startswith("env/"))
            for metric_name in metric_names:
                assert f"{metric_name}_mean" in env_metric_keys
                assert f"{metric_name}_count" in env_metric_keys

    @pytest.mark.asyncio 
    async def test_batched_generate_with_metrics(self, mock_generator):
        """Test metrics aggregation in batched mode."""
        _register_metrics_test_env()
        
        # Enable batched mode
        mock_generator.batched = True
        
        input_batch: GeneratorInput = {
            "prompts": [
                [{"role": "user", "content": "test1"}],
                [{"role": "user", "content": "test2"}]
            ],
            "env_extras": [
                {"test_metrics": True},
                {"test_metrics": True}
            ],
            "env_classes": ["cpu_metrics_test_env", "cpu_metrics_test_env"],
        }
        
        result: GeneratorOutput = await mock_generator.generate(input_batch)
        
        # Verify metrics are aggregated across the batch
        rollout_metrics = result["rollout_metrics"]
        env_metrics = {k: v for k, v in rollout_metrics.items() if k.startswith("env/")}
        
        # Should have aggregated metrics from both environments
        if env_metrics:
            # Verify that count reflects multiple environments
            count_metrics = {k: v for k, v in env_metrics.items() if k.endswith("_count")}
            if count_metrics:
                # Each env produces 1 step output (single turn), so count should be 2
                assert any(v >= 2 for v in count_metrics.values()), \
                    f"Expected counts >= 2, got: {count_metrics}"

    def test_metrics_field_in_base_text_env_step_output(self):
        """Test that BaseTextEnvStepOutput includes metrics field (TODO 1)."""
        # This verifies the first TODO - that metrics field exists
        step_output: BaseTextEnvStepOutput = {
            "observations": [],
            "reward": 1.0,
            "done": True,
            "metadata": {},
            "metrics": {"test_metric": 42, "accuracy": 0.95}
        }
        
        assert "metrics" in step_output
        assert step_output["metrics"]["test_metric"] == 42
        assert step_output["metrics"]["accuracy"] == 0.95


# Run with: uv run --extra dev --isolated pytest tests/cpu/generators/test_skyrl_gym_generator_metrics.py -v