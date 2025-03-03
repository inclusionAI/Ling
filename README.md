# Ling

<p align="center">
    <img src="./figures/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>

## Introduction

Ling is a MoE LLM provided and open-sourced by InclusionAI. We introduce two different sizes, which are Ling-Lite and Ling-Plus. Ling-Lite has 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus has 290 billion parameters with 28.8 billion activated parameters. Both models demonstrate impressive performance compared to existing models in the industry.

Their structure makes it easy to scale up and down and adapt to different tasks, so users can use these models for a wide range of tasks, from processing natural language to solving complex problems. Furthermore, the open-source nature of Ling promotes collaboration and innovation within the AI community, fostering a diverse range of use cases and enhancements.

As more developers and researchers engage with the platform, we can expect rapid advancements and improvements, leading to even more sophisticated applications. This collaborative approach accelerates development and ensures that the models remain at the forefront of technology, addressing emerging challenges in various fields.

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on Modulescope.cn to speed up the download process.

<div align="center">

|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ling-lite-base |       16.8B       |         2.75B         |        64K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-base) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-lite-base) |
| Ling-lite |       16.8B       |         2.75B         |        64K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-lite)     |
| Ling-plus-base |       290B        |         28.8B         |        64K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-plus-base) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-plus-base)     |
| Ling-plus |       290B        |         28.8B         |        64K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-plus) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-plus)     |
</div>

## Evaluation

Detailed evaluation results are reported in our technical report [TBD].

## Quickstart
### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-lite"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### 🤖 ModelScope

If you're in mainland China, we strongly recommend you to use our model from 🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>.

## Deployment

### vLLM

### MindIE

## Finetuning

We recommend you to use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to finetune Ling with SFT, DPO, etc.

We use [`identity`](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json) to demonstrate how to finetune our Ling models by replacing `name` with `Ling` and `author` with `inclusionAI`.

```json
{
    "instruction": "hi",
    "input": "",
    "output": "Hello! I am Ling, an AI assistant developed by inclusionAI. How can I assist you today?"
}
```

We provide a demo configuration of `Llama-Factory` to SFT Ling models as follows:

```bash
llamafactory-cli train examples/sft/ling_full_sft.yaml
```

## License
This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ling/blob/master/LICENCE).

## Citation
[TBD]

## Contact
If you have any questions, please raise an issue or contact us at xxx.