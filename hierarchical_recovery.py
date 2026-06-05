import os
import csv
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# dynamic path resolution for slurm execution
script_dir = os.path.dirname(os.path.abspath(__file__))
rsfs_dir = os.path.join(script_dir, "rsfs")
tika_source_root = os.path.join(script_dir, "tika", "tika-core", "src", "main", "java")
output_dir = os.path.join(script_dir, "architectural_summaries_final")

model_name = "Qwen/Qwen2.5-72B-Instruct"
hf_token = os.environ.get('HF_TOKEN')

print(f"initializing pipeline in {script_dir}")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("loading model weights...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

def generate_response(prompt_text, max_tokens):
    messages = [
        {"role": "system", "content": "You are a software architect. Strictly follow formatting and extraction constraints."},
        {"role": "user", "content": prompt_text}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.2,
        top_p=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    input_length = inputs['input_ids'].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

def process_leaf(raw_code):
    prompt = f"""Extract a semantic summary of the following Java source code.
Your summary MUST explicitly detail these four points:
1. Key functionality
2. Core logic
3. Inputs/Outputs
4. Dependencies

Source Code:
{raw_code}"""
    # 300 tokens gives the model enough room to answer the 4 points without bloating the context
    return generate_response(prompt, max_tokens=300)

def process_branch(compiled_leaf_summaries):
    prompt = f"""Below are the semantic summaries of files within a single architectural cluster.
Based strictly on these summaries, generate a cluster-level architectural description.

Constraints:
1. Provide a short, descriptive title.
2. The description MUST be STRICTLY UNDER 150 WORDS.
3. The description MUST explicitly state:
   - Components and Interactions: How the distinct parts work together.
   - Quality Attributes: Non-functional requirements achieved (e.g., scalability, security).
   - Technology Used: Frameworks, languages, or tools identified.

Format your exact output as:
TITLE: <title>
DESCRIPTION: <description>

File Summaries:
{compiled_leaf_summaries}"""
    
    response = generate_response(prompt, max_tokens=250)
    
    title = "Unknown"
    description = response
    
    if "TITLE:" in response and "DESCRIPTION:" in response:
        parts = response.split("DESCRIPTION:")
        title = parts[0].replace("TITLE:", "").strip()
        description = parts[1].strip()
        
    return title, description

def parse_rsf(filepath):
    clusters = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "contain":
                cluster_name = parts[1]
                # remove inner classes to align with arc
                file_path = parts[2].split('$')[0].replace('.', '/') + ".java"
                
                if cluster_name not in clusters:
                    clusters[cluster_name] = set()
                clusters[cluster_name].add(file_path)
    return clusters

def main():
    os.makedirs(output_dir, exist_ok=True)
    rsf_files = glob.glob(os.path.join(rsfs_dir, "*.rsf"))
    
    if not rsf_files:
        print("no rsf files found. exiting.")
        return

    for rsf_path in rsf_files:
        filename = os.path.basename(rsf_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"\n--- processing {filename} ---")
        clusters = parse_rsf(rsf_path)
        
        csv_filename = os.path.join(output_dir, f"{base_name}.csv")
        
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['cluster_ID', 'files', 'title', 'description'])
            
            for cluster_name, file_paths in clusters.items():
                print(f"summarizing {cluster_name} ({len(file_paths)} files)")
                
                compiled_leaves = ""
                csv_file_list = []
                
                for file_path in file_paths:
                    full_path = os.path.join(tika_source_root, file_path)
                    java_filename = os.path.basename(full_path)
                    csv_file_list.append(java_filename)
                    
                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            # 4000 char truncation for A100 memory safety
                            raw_code = f.read()[:4000] 
                        
                        leaf_summary = process_leaf(raw_code)
                        compiled_leaves += f"\n--- File: {java_filename} ---\n{leaf_summary}\n"
                    else:
                        print(f"warning: file missing -> {full_path}")
                
                title, description = process_branch(compiled_leaves)
                
                files_string = ", ".join(csv_file_list)
                writer.writerow([cluster_name, files_string, title, description])
                
        print(f"saved output to {csv_filename}")

if __name__ == "__main__":
    main()
