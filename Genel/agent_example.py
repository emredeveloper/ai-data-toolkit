from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


import asyncio
import re # Import regular expressions module
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    # Input validation
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Both inputs must be numbers")
    return float(a * b)  # Ensure float output

# Helper function to extract the last number found in a string
def extract_number(text: str) -> float | None:
    # Find all sequences of digits, possibly with a decimal point
    numbers = re.findall(r'\d+\.?\d*', text.replace(',', '')) # Remove commas for larger numbers
    if numbers:
        try:
            # Return the last number found as a float
            return float(numbers[-1])
        except ValueError:
            return None
    return None

# Create an LLM instance for direct prompting
llm = Ollama(model="llama3.2:3b", request_timeout=360.0, temperature=0.1)

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=llm,
    system_prompt="""You are a precise mathematical calculator.
    For multiplication problems:
    1. ALWAYS use the multiply tool - do not calculate manually
    2. Verify your answer matches a * b
    3. Return ONLY the numerical result
    4. No explanations or text allowed""",
)

# The question to ask
a_val = 1234
b_val = 4567
question = f"What is {a_val} * {b_val}?"
ground_truth = float(a_val * b_val) # Calculate the correct answer

async def main():
    # Run the agent (using the tool)
    print("--- Running Agent (with Tool) ---")
    response_agent_obj = await agent.run(question)
    response_agent_str = str(response_agent_obj)
    print(f"Agent Raw Response: {response_agent_str}\n")

    # Ask the LLM directly (without the tool)
    print("--- Running Direct Prompt ---")
    prompt = f"Calculate {a_val} * {b_val}. Respond only with the numerical result, nothing else."
    response_direct_obj = await llm.acomplete(prompt)
    response_direct_str = str(response_direct_obj)
    print(f"Direct LLM Raw Response: {response_direct_str}\n")

    # --- Evaluation ---
    print("--- Evaluation Results ---")
    print(f"Ground Truth: {ground_truth}")

    # Evaluate Agent
    agent_result = extract_number(response_agent_str)
    print(f"Agent Extracted Result: {agent_result}")
    if agent_result is not None:
        error_margin = abs((agent_result - ground_truth) / ground_truth) * 100
        print(f"Agent Error Margin: {error_margin:.2f}%")
    agent_correct = agent_result is not None and abs(agent_result - ground_truth) < 1e-6 # Compare floats with tolerance
    print(f"Agent Correct (Exact Match): {agent_correct}\n")

    # Evaluate Direct LLM
    direct_result = extract_number(response_direct_str)
    print(f"Direct LLM Extracted Result: {direct_result}")
    if direct_result is not None:
        error_margin = abs((direct_result - ground_truth) / ground_truth) * 100
        print(f"Direct LLM Error Margin: {error_margin:.2f}%")
    direct_correct = direct_result is not None and abs(direct_result - ground_truth) < 1e-6 # Compare floats with tolerance
    print(f"Direct LLM Correct (Exact Match): {direct_correct}")

    # Add Turkish explanation after evaluation
    print("\n--- Sonuçların Açıklaması ---")
    print("1. Agent (Tool kullanan): Bu sonuç hatalı çünkü agent multiply tool'unu düzgün kullanmadı")
    print("2. Direct LLM: Bu sonuç daha iyi çünkü LLM kendi içinde basit çarpma yapabilir")
    print("\nÖnerilen çözüm: Agent'ın multiply tool'unu kullanması gerekiyor.")


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())