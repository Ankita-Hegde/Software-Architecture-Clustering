import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 1. ENVIRONMENT, I/O & MODEL SETUP
# ==========================================
model_name = "Qwen/Qwen2.5-72B-Instruct"
ACDC_FILE_PATH = "./tika-acdc.rsf"       # Ensure this is in your HPC working directory
TIKA_SOURCE_ROOT = "./tika/tika-core/src/main/java/"
OUTPUT_DIR = "./architectural_summaries_final/"

hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    print("WARNING: HF_TOKEN not found. Proceeding with open-weights model download.")

def save_leaf_summary(cluster_name, filename, summary_text):
    out_dir = os.path.join(OUTPUT_DIR, cluster_name, "leaves")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{filename}.txt"), 'w', encoding='utf-8') as f:
        f.write(summary_text)

def save_branch_summary(cluster_name, arch_summary_text):
    out_dir = os.path.join(OUTPUT_DIR, cluster_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{cluster_name}_ARCHITECTURE.txt"), 'w', encoding='utf-8') as f:
        f.write(arch_summary_text)

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
# 3. LOAD THE MODEL (WITH 4-BIT QUANTIZATION)
# ==========================================
print("Configuring 4-bit Quantization parameters...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("Loading Model across 2x A100 GPUs with optimizations...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True  # ⚠️ CRUCIAL: Prevents loading the uncompressed model into CPU RAM first
)

# ==========================================
# 4. INFERENCE ENGINE
# ==========================================
def generate_response(prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful software assistant. Your job is to explain the functionality of the provided code in simple terms."},
        {"role": "user", "content": prompt_text}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Temperature lowered from 0.9 to 0.2 to prevent hallucination during summarization
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.2,
        top_p=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    input_length = inputs['input_ids'].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

# ==========================================
# 5. PIPELINE EXECUTION
# ==========================================
def main():
    if not os.path.exists(ACDC_FILE_PATH):
        raise FileNotFoundError(f"Missing {ACDC_FILE_PATH}. Upload it to your HPC working directory.")

    print(f"Parsing ACDC RSF File...")
    with open(ACDC_FILE_PATH, 'r', encoding='utf-8') as f:
        acdc_raw_text = f.read()

    # Group all files by their cluster dynamically
    clusters = {}
    for line in acdc_raw_text.strip().split('\n'):
        parts = line.split()
        if len(parts) == 3 and parts[0] == "contain":
            cluster_name = parts[1]
            file_path = parts[2].replace('.', '/')
            if '$' in file_path:
                file_path = file_path.split('$')[0]
            file_path += ".java"

            if cluster_name not in clusters:
                clusters[cluster_name] = []
            if file_path not in clusters[cluster_name]:
                clusters[cluster_name].append(file_path)

    total_clusters = len(clusters)
    print(f"Found {total_clusters} clusters to process.")

    for idx, (cluster_name, java_files) in enumerate(clusters.items(), 1):
        print(f"\n=======================================================")
        print(f"PROCESSING CLUSTER {idx}/{total_clusters}: {cluster_name}")
        print(f"=======================================================")

        leaf_summaries = {}

        for file_path in java_files:
            full_path = os.path.join(TIKA_SOURCE_ROOT, file_path)
            filename = os.path.basename(full_path)

            if not os.path.exists(full_path):
                print(f"  [!] Missing file: {full_path}")
                continue

            with open(full_path, 'r', encoding='utf-8') as f:
                raw_code = f.read()

            prompt = f"""Extract a semantic summary from the following raw source code file.
Ensure the summary explicitly details the following:
- Key functionality
- Core logic
- Inputs/Outputs
- Dependencies

Source Code:
{raw_code}"""

            print(f"  -> Summarizing Leaf: {filename}")
            try:
                summary = generate_response(prompt)
                save_leaf_summary(cluster_name, filename, summary)
                leaf_summaries[filename] = summary
            except Exception as e:
                print(f"  [!] Error processing file {filename}: {str(e)}")
                continue

        if not leaf_summaries:
            print(f"  [!] No valid leaves found for {cluster_name}. Skipping branch generation.")
            continue

        compiled_text = ""
        for fname, summ in leaf_summaries.items():
            compiled_text += f"\n--- File: {fname} ---\n{summ}\n"

        branch_prompt = f"""Below are the summaries of constituent files within a specific directory cluster.

Based strictly on this list of summaries, generate:
1. A title.
2. A high-level descriptive summary explaining the module’s overall behaviour, architecture, and how the components interact within the cluster.

Constituent Summaries:
{compiled_text}"""

        print(f"  -> Generating Branch Architecture for: {cluster_name}...")
        try:
            arch_summary = generate_response(branch_prompt)
            save_branch_summary(cluster_name, arch_summary)
        except Exception as e:
            print(f"  [!] Error generating branch summary for {cluster_name}: {str(e)}")

    print("\n✅ PIPELINE COMPLETE. All clusters have been successfully summarized.")

if __name__ == "__main__":
    main()

