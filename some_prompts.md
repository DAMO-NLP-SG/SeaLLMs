
### Some interesting prompts for SeaLLM

```
Answer the following query exclusively based on the information provided in the document above. \
If the information is not found, please say so instead of making up facts! Remember to answer the question in the same language as the user query!

###
### Multilingual World Knowledge


We evaluate models on 3 benchmarks following the recommended default setups: 5-shot MMLU for En, 3-shot [M3Exam](https://arxiv.org/pdf/2306.05179.pdf) (M3e) for En, Zh, Vi, Id, Th, and zero-shot [VMLU](https://vmlu.ai/) for Vi.

| Model | Langs | En<br>MMLU | En<br>M3e | Zh<br>M3e | Vi<br>M3e | Vi<br>VMLU | Id<br>M3e | Th<br>M3e
|-----| -----  | --- |  -- | ----- | ---- | --- | --- | --- |
| GPT-3.5         | Multi | 68.90 | 75.46 | 60.20 | 58.64 | 46.32 | 49.27 | 37.41
| Vistral-7B-chat | Mono  | 56.86 | 67.00 | 44.56 | 54.33 | 50.03 | 36.49 | 25.27
| Qwen1.5-7B-chat | Multi | 61.00 | 52.07 | 81.96 | 43.38 | 45.02 | 24.29 | 20.25
| SailorLM        | Multi | 52.72 | 59.76 | 67.74 | 50.14 | ---   | 39.53 | 37.73
| SeaLLM-7B-v2    | Multi | 61.89 | 70.91 | 55.43 | 51.15 | 45.74 | 42.25 | 35.52
| SeaLLM-7B-v2.5  | Multi | 64.05 | 76.87 | 62.54 | 63.11 | 53.30 | 48.64 | 46.86

###

What is the VMLU score for SeaLLM-7B-v2?
```







