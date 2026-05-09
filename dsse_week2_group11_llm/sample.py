# ==========================================
# 0. IMPORT THE REQUIRED DEPENDENCIES
# ==========================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. ENVIRONMENT & MODEL SETUP
# ==========================================
model_name = "Qwen/Qwen2.5-72B-Instruct" 

import os
hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    print("WARNING: HF_TOKEN not found. Proceeding with open-weights model download.")


# ==========================================
# X Hardware optimization (Quantization) X ====> NO need for this step here [WHY?!]
# ==========================================

# ==========================================
# 2. LOAD THE TOKENIZER
# ==========================================
print(f"Loading Tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    token=hf_token, 
    trust_remote_code=True 
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# ==========================================
# 3. LOAD THE MODEL
# ==========================================
print("Loading Model across 2x A100 GPUs...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,             
    device_map="auto"                   
)

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================

# Design the prompt
messages = [
    {
        "role": "system", 
        "content": "You are a helpful software assistant. Your job is to explain the functionality of the provided code in simple terms."
    },
    {
        "role": "user", 
        "content": """Please analyze the following source code:
        
    <source_code>
    public class Task2 { 
    
        public static void main(String[] args) { 
            String id1   = "AMQ-2104"; 
            double files = 8.0; 
            String id2   = "AMQ-317"; 
            double dmm   = 0.45; 
    
            int n = Integer.parseInt(id1.split("-")[1]) 
                + Integer.parseInt(id2.split("-")[1]); 
    
            int digits = 0; 
            while (n > 0) { 
                n = n / 10; 
                digits++; 
            } 
    
            int impact = (int) (files * dmm); 
    
            System.out.println("Combined digits: " + digits); 
            System.out.println("Impact: " + impact); 
        } 
    } 
</source_code>
"""
    }
]

# Convert the prompt into tokens and move to GPUs
inputs = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    return_dict=True, 
    return_tensors="pt"
).to(model.device)

# Model inference
print("Generating response...\n")
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,      
    temperature=0.1,
    top_p=0.8,                
    do_sample=True,           
    pad_token_id=tokenizer.eos_token_id
)

# Convert the output back from tokens into human-readable text and display it (in the log file)
input_length = inputs['input_ids'].shape[1]
response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
print("--- Model Output ---")
print(response)