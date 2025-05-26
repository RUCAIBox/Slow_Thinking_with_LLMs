import os, sys
import argparse

# Get the project root directory by going up 3 levels from current file
p1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(p1)

# Alternative way to get project root (2 levels up)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from searchagent.utils.utils import remove_think_tags
import json
import re
from openai import OpenAI

def parse_args():
    """Parse command line arguments for evaluation configuration"""
    parser = argparse.ArgumentParser(description="Do Eval datasets with openai api")

    # Required arguments
    parser.add_argument('--model_name', type=str, required=True, help="Name of the planner model to use")
    parser.add_argument('--api_base', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--api_key', type=str, required=True, help="api key for the planner model API endpoint")
    parser.add_argument('--file_path', type=str, required=True, help="Set you file to eval")
    
    return parser.parse_args()

args = parse_args()

# Initialize OpenAI client with provided credentials
client = OpenAI(
    api_key=args.api_key,
    base_url=args.api_base
)

def generate(messages, model_name):
    """Generate response using OpenAI's chat completion API"""
    response = client.chat.completions.create(
        **{
            "model": model_name,
            "messages": messages,
            "max_tokens": 2048,
        }
    )
    response = response.choices[0].message.content
    return response

def validate_data(file, model_name):
    """Validate prediction accuracy against golden answers and save results"""
    # Evaluation prompt template
    PROMPT = '''Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.
Golden Answer may have multiple options, and matching any one of them is considered correct.

Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}
    '''

    print("Begin processing:", file)
    file_path = file
    print("Current file:", file_path)

    # Initialize counters
    valid_num = 0      # Total valid records processed
    correct_num = 0    # Correct predictions
    incorrect_num = 0  # Incorrect predictions
    result_data = []   # Stores processed results with evaluations

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("Invalid JSON:", line)
                continue

            valid_num += 1
            # Extract and clean prediction answer
            prediction = obj.get("answer", "I don't know")
            if isinstance(prediction, dict):
                prediction = prediction.get("content")
                if isinstance(prediction, dict):
                    prediction = prediction.get("concise_answer", "")
            else:
                prediction = remove_think_tags(prediction)
                
            question = obj.get("question", "")
            answer = obj.get("gold", [])
            
            print("="*70)
            print("Question:", question)
            print("Prediction:", prediction)
            print("Golden Answer:", answer)

            # Format the evaluation prompt
            gpt4o_input = PROMPT.format(
                question=question,
                reference=answer, 
                prediction=prediction
            )
            messages = [{'role': 'user', 'content': gpt4o_input}]

            # Get evaluation from GPT model
            model_output = generate(messages, model_name)

            # Determine correctness based on model output
            if "false" in model_output.lower():
                is_correct = False
            else:
                is_correct = True

            if is_correct:
                correct_num += 1
            else:
                incorrect_num += 1

            # Store evaluation result
            obj['check_ans'] = model_output
            print("Record no.", valid_num, ":", model_output)
            result_data.append(obj)

    # Calculate accuracy
    if valid_num > 0:
        accuracy = correct_num / valid_num * 100
    else:
        accuracy = 0

    # Print summary statistics
    print(f"File: {file}")
    print(f"Valid objects: {valid_num}")
    print(f"Correct objects: {correct_num}")
    print(f"Incorrect objects: {incorrect_num}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    # Save results to new JSONL file
    output_file_path = file_path.replace('.jsonl', '_with_check_ans.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for obj in result_data:
            output_file.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    validate_data(args.file_path, args.model_name)