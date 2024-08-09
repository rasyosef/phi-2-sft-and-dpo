
# phi-2-instruct-v0.1
[Phi-2](https://huggingface.co/microsoft/phi-2) is a Transformer with **2.7 billion** parameters. It was trained using the same data sources as [Phi-1.5](https://huggingface.co/microsoft/phi-1.5), augmented with a new data source that consists of various NLP synthetic texts and filtered websites (for safety and educational value). When assessed against benchmarks testing common sense, language understanding, and logical reasoning, Phi-2 showcased a nearly state-of-the-art performance among models with less than 13 billion parameters.

The model has underwent a post-training process that incorporates both **supervised fine-tuning** and **direct preference optimization** for instruction following. I used the [trl](https://huggingface.co/docs/trl/en/index) library and a single **A100 40GB** GPU during both the SFT and DPO steps.

- Supervised Fine-Tuning
  - SFT Model: [phi-2-sft](https://huggingface.co/rasyosef/phi-2-sft-openhermes-128k-v2)
  - Used 128,000 instruction, response pairs from the [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) dataset

- Direct Preference Optimization (DPO)
  - LoRA Adapter: [phi-2-dpo](https://huggingface.co/rasyosef/phi-2-openhermes-128k-v2-dpo-combined)
  - Used a combination of the following preference datasets
    - [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
    - [argilla/distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs)
    - [argilla/distilabel-math-preference-dpo](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo)

## How to use
### Chat Format

Given the nature of the training data, the phi-2 instruct model is best suited for prompts using the chat format as follows. 
You can provide the prompt as a question with a generic template as follow:
```markdown
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question?<|im_end|>
<|im_start|>assistant
```

For example:
```markdown
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
How to explain Internet for a medieval knight?<|im_end|>
<|im_start|>assistant
```
where the model generates the text after `<|im_start|>assistant` .

### Sample inference code

This code snippets show how to get quickly started with running the model on a GPU:

```python
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 

model_id = "rasyosef/phi-2-instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained( 
    model_id,  
    device_map="cuda",  
    torch_dtype="auto" 
) 

tokenizer = AutoTokenizer.from_pretrained(model_id) 

messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 256, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])  
```

Note: If you want to use flash attention, call _AutoModelForCausalLM.from_pretrained()_ with _attn_implementation="flash_attention_2"_
