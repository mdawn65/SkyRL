import sys
sys.path.append('skyrl-gym')

from skyrl_gym.envs.gsm8k.env import GSM8kEnv
from omegaconf import DictConfig

def test_gsm8k_metrics():
    """Test the GSM8K environment with various response types to verify metrics."""
    
    # Create test data
    extras = {
        "reward_spec": {
            "ground_truth": "24"  # Example ground truth answer
        }
    }

    # Initialize environment
    env = GSM8kEnv(env_config=DictConfig({}), extras=extras)

    # Test different types of responses
    test_responses = [
        # Correct answer with good reasoning
        "To solve this problem, I need to multiply 6 by 4. First, I'll calculate: 6 × 4 = 24. Therefore, the answer is #### 24",
        
        # Correct answer, poor format
        "The answer is 24",
        
        # Wrong answer with good reasoning
        "Let me work step by step. First I'll add 6 + 4 = 10. Then multiply by 2: 10 × 2 = 20. So the answer is #### 20",
        
        # No numerical answer
        "I'm not sure how to solve this problem. Maybe I need more information.",
        
        # Correct answer with uncertainty
        "I think the calculation might be 6 × 4 = 24, but I'm not completely sure. The answer could be #### 24",
    ]

    print("Testing GSM8K Environment with Metrics")
    print("=" * 60)
    print(f"Ground Truth Answer: {extras['reward_spec']['ground_truth']}")
    print("=" * 60)

    for i, response in enumerate(test_responses, 1):
        print(f"\n🧪 Test {i}:")
        print(f"Response: {response}")
        print("-" * 40)
        
        result = env.step(response)
        
        print(f"💰 Reward: {result['reward']}")
        print("📊 Metrics:")
        
        # Group metrics by category for better readability
        metrics = result['metrics']
        
        # Core accuracy metrics
        print("   🎯 Accuracy Metrics:")
        print(f"      answer_accuracy: {metrics['answer_accuracy']}")
        print(f"      has_numerical_answer: {metrics['has_numerical_answer']}")
        print(f"      format_compliance: {metrics['format_compliance']}")
        
        # Response quality metrics  
        print("   📝 Response Quality:")
        print(f"      response_length: {metrics['response_length']}")
        print(f"      word_count: {metrics['word_count']}")
        print(f"      reasoning_steps: {metrics['reasoning_steps']}")
        
        # Mathematical reasoning metrics
        print("   🔢 Math Reasoning:")
        print(f"      math_operations_count: {metrics['math_operations_count']}")
        print(f"      contains_calculation: {metrics['contains_calculation']}")
        print(f"      has_step_by_step_reasoning: {metrics['has_step_by_step_reasoning']}")
        
        # Other metrics
        print("   🤔 Confidence & Methods:")
        print(f"      uncertainty_indicators: {metrics['uncertainty_indicators']}")
        print(f"      strict_method_success: {metrics['strict_method_success']}")
        print(f"      flexible_method_success: {metrics['flexible_method_success']}")
        
        print("=" * 60)

    print("\n✅ GSM8K Metrics Test Complete!")
    print("\nThese metrics will now be automatically aggregated and logged to wandb")
    print("under the 'env/' namespace when used in training!")


if __name__ == "__main__":
    test_gsm8k_metrics()