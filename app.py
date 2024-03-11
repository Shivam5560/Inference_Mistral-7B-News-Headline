import streamlit as st

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
        "shivam9980/mistral-7b-news", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True,)
tokenizer = AutoTokenizer.from_pretrained("shivam9980/mistral-7b-news")

# alpaca_prompt = You MUST copy from above!

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

content = st.text_input('Content')
inputs = tokenizer(
[
    alpaca_prompt.format(
        "The following passage is content from a news report. Please summarize this passage in one sentence or less.", # instruction
        content, # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
results= tokenizer.batch_decode(outputs)
out = results[0].split('\n')[-1]
st.text_area(label='Headline',value=out[:len(out)-4])
