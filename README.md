
<div align=center>
<h1>STILL: Slow Thinking with LLMs</h1>
<a href="https://arxiv.org/abs/2412.09413" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/RUC-AIBOX/long_form_thought_data_5k"><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
<a><img src="https://img.shields.io/github/stars/RUCAIBox/Slow_Thinking_with_LLMs"></a>
</div>



We are STILL exploring the uncharted territory of o1-like reasoning systems.

## News

+ [22 Dec 2024] We open-source part of the **training data** in [Github](data/public_long_form_thought_data_5k.jsonl) or [HuggingFace](https://huggingface.co/datasets/RUC-AIBOX/long_form_thought_data_5k) and the [**model**](https://huggingface.co/RUC-AIBOX/STILL-2) for community researchers to use for research purposes.
+ [12 Dec 2024] [**STILL-2**](#imitate-explore-and-self-improve-a-reproduction-report-on-slow-thinking-reasoning-systems-report): We preliminarily reproduce **a slow-thinking reasoning system**, achieving competitive performance compared to industry-level reasoning systems on these benchmarks! And we also release the [technical report](https://arxiv.org/pdf/2412.09413), which presents the details about our reproduction.
+ [18 Nov 2024] [**STILL-1**](#enhancing-llm-reasoning-with-reward-guided-tree-search-report): We release our first [technical report](https://arxiv.org/abs/2411.11694), where we leverage **reward-guided tree search algorithm** to assist LLM reasoning process and largely enhance the performance of LLM on complex reasoning tasks.

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("RUC-AIBOX/STILL-2")
model = AutoModelForCausalLM.from_pretrained("RUC-AIBOX/STILL-2")

# PROMPT
PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.\n\nPlease structure your response into two main sections: Thought and Solution.\n\nIn the Thought section, detail your reasoning process using the specified format:\n\n```\n<|begin_of_thought|>\n{thought with steps separated with \"\n\n\"}\n<|end_of_thought|>\n```\n\nEach step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. Try to use casual, genuine phrases like: \"Hmm...\", \"This is interesting because...\", \"Wait, let me think about...\", \"Actually...\", \"Now that I look at it...\", \"This reminds me of...\", \"I wonder if...\", \"But then again...\", \"Let's see if...\", \"Alternatively...\", \"Let's summaize existing information...\", \"This might mean that...\", \"why/how/when/where...\", etc, to make your thought process be coherent, clear, and logically sound, effectively simulating human cognitive processes.\n\nIn the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:\n\n```\n<|begin_of_solution|>\n{final formatted, precise, and clear solution}\n<|end_of_solution|>\n```\n\nNow, try to solve the following question through the above guidlines:\n"

# Input text
question = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"

input_prompts = tokenizer.apply_chat_template(
                [
                {"role": "user", "content": PROMPT + question}],
                tokenize=False,
                add_generation_prompt=True
            )


# Params
stop_words = ['<|im_end|>', '<|endoftext|>']

llm = LLM(model=model_path, tensor_parallel_size=8, max_model_len=int(1.5*20000), gpu_memory_utilization=0.95, dtype='bfloat16')

sampling_params_gs = SamplingParams(temperature=0, top_p=1.0, max_tokens=20000, stop=stop_words, seed=42, skip_special_tokens=False)


# Completion
responses = model.generate(input_prompts, sampling_params)
print(responses[0].outputs[0].text)
```



## Technical Report

### Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems [[Report]](https://arxiv.org/pdf/2412.09413)

+ Slow-thinking reasoning systems, such as o1, have demonstrated remarkable capabilities in solving complex reasoning tasks, and are primarily developed and maintained by industry, with their core techniques not publicly disclosed. This paper presents a reproduction report on implementing o1-like reasoning systems. We introduce an **imitate, explore, and self-improve framework** as our primary technical approach to train the reasoning model. In the initial phase, we use distilled long-form thought data to fine-tune the reasoning model, enabling it to invoke a slow-thinking mode. The model is then encouraged to explore challenging problems by generating multiple rollouts, which can result in increasingly more high-quality trajectories that lead to correct answers. Furthermore, the model undergoes self-improvement by iteratively refining its training dataset.
  <img src="figures/report_2.jpg" alt="report_1" style="zoom:50%;" />

  <img src="figures/part_2_main_res.png" alt="report_1" style="zoom:50%;" />

### Enhancing LLM Reasoning with Reward-guided Tree Search [[Report]](https://arxiv.org/abs/2411.11694)

+ Recently, test-time scaling has garnered significant attention from the research community, largely due to the substantial advancements of the o1 model released by OpenAI. However,  develop an o1-like reasoning approach is challenging, and  researchers have been making various attempts to advance this open area of research. In this paper, we present a preliminary exploration into enhancing the reasoning abilities of  LLMs through **reward-guided tree search algorithms**. This framework is implemented by integrating the policy model, reward model, and search algorithm. It is primarily constructed around a tree search algorithm, where the policy model navigates a dynamically expanding tree guided by a specially trained reward model. 

  <img src="figures/report_1.jpg" alt="report_1" style="zoom:50%;" />

## Future Work

Despite the promising results, our exploration remains preliminary, and there is still a substantial capacity gap compared to industry-level systems. As future work, we plan to investigate how to scale our training approach and extend its capacity to more complex tasks. 

As always, we are committed to keeping our technical approach *open*, and we will release the data, model, and other resources. We welcome collaboration and support in computational resources.

## Reference

Please kindly cite our reports if they are helpful for your research.

```
@article{Slow_Thinking_with_LLMs_1,
  title={Enhancing LLM Reasoning with Reward-guided Tree Search},
  author={Jiang, Jinhao and Chen, Zhipeng and Min, Yingqian and Chen, Jie and Cheng, Xiaoxue and Wang, Jiapeng and Tang, Yiru and Sun, Haoxiang and Deng, Jia and Zhao, Wayne Xin and Liu, Zheng and Yan, Dong and Xie, Jian and Wang, Zhongyuan and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2411.11694},
  year={2024}
}
```

```
@article{Slow_Thinking_with_LLMs_2,
  title={Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems},
  author={Min, Yingqian and Chen, Zhipeng and Jiang, Jinhao and Chen, Jie and Deng, Jia and Hu, Yiwen and Tang, Yiru and Wang, Jiapeng and Cheng, Xiaoxue and Song, Huatong and Zhao, Wayne Xin and Liu, Zheng and Wang, Zhongyuan and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2412.09413},
  year={2024}
}
```

