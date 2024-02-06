# make sure to use vllm 0.2.7 and transformers 4.36+

import json
from vllm import LLM, SamplingParams
import csv

jsonl_path = f"test.jsonl"
out_csv_path = f"pred2.csv"

def read_json(json_file):
    print(f'Reading : {json_file}')
    with open(json_file, 'r', encoding='utf-8') as f:
        rows = [json.loads(x) for x in f]
    return rows

questions = read_json(jsonl_path)


sampling_params = SamplingParams(temperature=0.0, max_tokens=1, stop=["</s>"])
model_path = "SeaLLMs/SeaLLM-7B-v2"
# model_path = "/mnt/workspace/workgroup/phi/pret_models/models--SeaLLMs--SeaLLM-7B-v2/snapshots/06c960c2b8fb75d453533c281fe31f550a04c61a"
model = LLM(model_path, dtype="bfloat16")


CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.</s>
<|im_start|>user
{prompt}</s>
<|im_start|>assistant
"""
question_template = """Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau:

{question}
{choices}
Đáp án:"""

def to_prompt(item):
    question = question_template.format(
        question=item['question'],
        choices="\n".join(item['choices'])
    )
    prompt = CHAT_TEMPLATE.format(prompt=question)
    return prompt


prompts = [to_prompt(q) for q in questions]

generated = model.generate(prompts, sampling_params)
responses = [g.outputs[0].text for g in generated]
answers = [r.strip()[0] for r in responses]

assert len(answers) == len(questions)
print(answers[:10])

with open(out_csv_path, 'w', encoding='utf-8') as f:
    write = csv.writer(f)
    _ = write.writerow(["id", "answer"])
    for q, pred in zip(questions, answers):
        _ = write.writerow([q['id'], pred])



