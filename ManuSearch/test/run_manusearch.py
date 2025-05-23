import sys
import os
import json
import time
import argparse
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from searchagent.agent.agent_test import AgentInterface




def parse_args():
    parser = argparse.ArgumentParser(description="Run ManuSearch for various datasets and models.")
    parser.add_argument('--single_question', type=str, default=None, help="Single question to process instead of dataset")
    parser.add_argument('--dataset_name', type=str, required=False, default='custom', help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=False, default='test', help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")

    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument('--min_p', type=float, default=0.05, help="Minimum p sampling parameter.")
    parser.add_argument('--top_k_sampling', type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help="Repetition penalty. If not set, defaults based on the model.")
    parser.add_argument('--max_tokens', type=int, default=81920, help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.")

    parser.add_argument('--eval', action='store_true', help="Whether to run evaluation after generation.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for generation. If not set, will use current timestamp as seed.")
    return parser.parse_args()

# Initialize tokenizers
args = parse_args()


def get_data(data_file):

    with open(data_file, "r", encoding='utf-8') as f:
        json_dict = [json.loads(line) for line in f]

    Question = [item['question'] for item in json_dict]
    Answer = [item['answer'] for item in json_dict]
    q_id = [item['q_id'] for item in json_dict]

    return q_id, Question, Answer



def process_single_sequence(message):
    agent = AgentInterface()
    for step, use_en in agent.get_answer(message, solve_method='iterative'):
        answer = step.get('final_resp', '')
    think = agent.recorder.generate_reason_process()
    return answer, think


def main():
    # Set random seed
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)  

    # Modified data loading section
    if args.single_question:
        # Create a single item in the same format as dataset items
        filtered_data = [{
            'Question': args.single_question,
        }]
        args.dataset_name = 'custom'  # Set dataset name to custom for single questions
    
    else:
        # Original dataset loading logic
        if args.dataset_name == 'GAIA':
            data_path = f'./data/GAIA/{args.split}.json'
        elif args.dataset_name == 'FRAMES':
            data_path = f'./data/FRAMES/{args.split}.json'
        elif args.dataset_name == 'ORION':
            data_path = f'./data/ORION/{args.split}.json'
        else:
            data_path = f'./data/{args.dataset_name}.json'
        
        print('-----------------------')
        print(f'Using {args.dataset_name} {args.split} set.')
        print('-----------------------')


    # Define output directory
    if 'qwq' in args.model_name.lower():
        model_short_name = 'qwq'
        if 'webthinker' in args.model_name.lower():
            model_short_name = f'webthinker{args.model_name.split("webthinker")[-1]}'
    elif 'deepseek' in args.model_name.lower():
        if 'llama-8b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-8b'
        elif 'llama-70b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-70b'
        elif 'qwen-1.5b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-1.5b'
        elif 'qwen-7b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-14b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-14b'
        elif 'qwen-32b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-32b'
        if 'webthinker' in args.model_name.lower():
            model_short_name = f'webthinker{args.model_name.split("webthinker")[-1]}'
    else:
        model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')

    output_dir = f'./outputs/{args.dataset_name}.{model_short_name}.manusearch'
    os.makedirs(output_dir, exist_ok=True)

    
    if not args.single_question:
        # Load and prepare data
        with open(data_path, 'r', encoding='utf-8') as json_file:
            filtered_data = json.load(json_file)

        if args.subset_num != -1:
            indices = list(range(len(filtered_data)))
            selected_indices = random.sample(indices, min(args.subset_num, len(indices)))
            filtered_data = [filtered_data[i] for i in selected_indices]


    # Initialize batch output records
    batch_output_records = []
    start_time = time.time()

    # Create semaphore for concurrent API calls
    semaphore = asyncio.Semaphore(args.concurrent_limit)

    try:
        # Process all sequences concurrently
        tasks = [
            process_single_sequence(
                message=question,
            )
            for question in filtered_data
        ]

        # Run all sequences concurrently with progress bar
        with tqdm(total=len(tasks)) as pbar:
            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result
            
            tracked_tasks = [track_progress(task) for task in tasks]
            completed_sequences = await asyncio.gather(*tracked_tasks)
    finally:
        # Unload LoRA adapter if it was loaded
        if args.lora_name:
            print(f"Unloading LoRA adapter '{args.lora_name}'")
            await unload_lora_adapter(args.api_base_url, args.lora_name)
            print("LoRA adapter unloaded successfully")

    total_time = time.time() - start_time

    if args.eval:
        # Prepare output list and save results
        output_list = [seq['output'] for seq in completed_sequences]
        run_evaluation(filtered_data, [seq['original_prompt'] for seq in completed_sequences], output_list, args.dataset_name, output_dir, total_time, args.split)
    else:
        t = time.localtime()
        random_num = str(random.randint(0, 99)).zfill(2)
        result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{random_num}.json'

        for item, seq in zip(filtered_data, completed_sequences):
            item['prompt'] = seq['original_prompt']
            item['Output'] = seq['output']
            item['WebExplorer'] = seq['web_explorer']  # Updated field name
            
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    # Save caches
    save_caches()
    print("Process completed.")

if __name__ == "__main__":
    main()
