
# SeaLLM-7B-v2 is chat model, so any zero-shot prompt instruction must be put in chat format, otherwise it will be invalid

EXAMPLE_PROMPT = """<|im_start|>system
You are a helpful assistant.</s>
<|im_start|>user
Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau:

Một nền kinh tế trong trạng thái toàn dụng nhân công có nghĩa là:
A. Không còn lạm phát nhưng có thể còn thất nghiệp
B. Không còn thất nghiệp nhưng có thể còn lạm phát
C. Không còn thất nghiệp và không còn lạm phát
D. Vẫn còn một tỷ lệ lạm phát và tỷ lệ thất nghiệp nhất định
Đáp án:</s>
<|im_start|>assistant
"""

from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.0, max_tokens=5, stop=["</s>"])
model_path = "SeaLLMs/SeaLLM-7B-v2"

# Must use bfloat16, because float16 or lower may lead to NaN computations.
model = LLM(model_path, dtype="bfloat16")

# must apply chatformat like this.
question_prompt = """Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau:

Một nền kinh tế trong trạng thái toàn dụng nhân công có nghĩa là:
A. Không còn lạm phát nhưng có thể còn thất nghiệp
B. Không còn thất nghiệp nhưng có thể còn lạm phát
C. Không còn thất nghiệp và không còn lạm phát
D. Vẫn còn một tỷ lệ lạm phát và tỷ lệ thất nghiệp nhất định
Đáp án:"""

# no \n at the beginning, and there must be \n at the end of <|im_start|>assistant
example_prompt = f"<|im_start|>system\nYou are a helpful assistant.</s>\n<|im_start|>user\n{question_prompt}</s>\n<|im_start|>assistant\n"

prompts = [example_prompt]

generated = model.generate(prompts, sampling_params)
responses = [g.outputs[0].text for g in generateds]

# extract the first non-empty character as answers
answers = [r.strip()[0] for r in responses]

print(answers)
