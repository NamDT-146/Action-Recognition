This is a solid engineering approach. Since your goal is **"Zero Miss Rate"** (Recall > Precision), we need to test if the base models are "smart" enough to catch violence out-of-the-box, or if they need fine-tuning to stop being "polite."

Here is the **Implementation Plan** to test **Qwen2-VL-2B** vs. **Qwen2.5-VL-3B** on your RTX 3090.

---

### **Phase 1: Environment Setup**

You need a unified environment that supports both models.

```bash
# 1. Create a fresh conda env (Recommended to avoid conflict)
conda create -n video_detect python=3.10 -y
conda activate video_detect

# 2. Install Pytorch (Cuda 12.1 recommended for 3090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install Qwen-VL dependencies & Flash Attention (CRITICAL for speed)
pip install git+https://github.com/huggingface/transformers 
pip install qwen-vl-utils accelerate
pip install flash-attn --no-build-isolation

```

---

### **Phase 2: Unified Inference Script (Pre-trained Test)**

Use this script to run the **same video** through **both models** and compare the outputs side-by-side.

**File:** `compare_models.py`

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time

# CONFIGURATION
VIDEO_PATH = "test_violence.mp4" # <--- Put a violent clip here
MODELS = {
    "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2.5-VL-3B": "Qwen/Qwen2.5-VL-3B-Instruct"
}

def run_inference(model_name, model_id):
    print(f"\n--- Loading {model_name} ---")
    
    # Select correct class based on model generation
    if "2.5" in model_name:
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        model_class = Qwen2VLForConditionalGeneration

    # Load Model (Flash Attention 2 is KEY for 3090 speed)
    model = model_class.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Prompt specifically for TEMPORAL LOCALIZATION
    prompt_text = (
        "detect violent events. "
        "If found, output format: [START_TIME - END_TIME] Description. "
        "If none, say 'Safe'."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": VIDEO_PATH,
                    "max_pixels": 360 * 420, # Low res is fine for action detection, speeds up inference
                    "fps": 2.0,              # 2 FPS is enough to catch a punch
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    end_time = time.time()

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Clean output (remove system prompt echo if any)
    response = output_text.split("assistant")[-1].strip()
    
    print(f"[{model_name}] Inference Time: {end_time - start_time:.2f}s")
    print(f"[{model_name}] Result: {response}")
    
    # Cleanup VRAM for next model
    del model
    torch.cuda.empty_cache()

# Run both
for name, path in MODELS.items():
    run_inference(name, path)

```

**What to look for:**

* **Qwen2-VL-2B:** Might say "A fight starts at the beginning." (Vague)
* **Qwen2.5-VL-3B:** Should say "00:04 - 00:06 Two men push each other." (Precise)

---

### **Phase 3: Fine-Tuning (If Pre-trained Fails)**

If the models are missing "subtle" violence (e.g., bullying, grabbing), you **must** fine-tune.
We will use **ms-swift** (by Alibaba). It is the best tool for Qwen models and supports video natively.

#### **1. Install Training Tool**

```bash
pip install ms-swift[llm] -U

```

#### **2. Prepare Data (JSONL Format)**

You need a `train.jsonl` file. The "response" must include the **timestamps** you want the model to learn.

```json
{"query": "<video>Find violence.", "response": "00:02-00:05 A man punches another man.", "videos": ["/path/to/clip1.mp4"]}
{"query": "<video>Find violence.", "response": "Safe.", "videos": ["/path/to/clip2.mp4"]}

```

#### **3. Run Fine-Tuning (LoRA)**

This command fits on your **24GB 3090**. It freezes the main model and only trains a small adapter (LoRA).

**Command for Qwen2.5-VL-3B:**

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen2_5-vl-3b-instruct \
    --dataset train.jsonl \
    --sft_type lora \
    --learning_rate 1e-4 \
    --lora_target_modules ALL \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --num_train_epochs 3 \
    --output_dir output/violence_v1

```

* **`--lora_target_modules ALL`**: Crucial for Vision-Language models. It trains the vision connector, not just the text part.
* **`--batch_size 1`**: Keeps VRAM usage low (~14GB).

---

### **Phase 4: Export & Merge**

Once training finishes, you have a "LoRA adapter" (small file). To use it in your C++ or Python production app, you usually want to merge it back into the base model.

```bash
swift merge --ckpt_dir output/violence_v1/checkpoint-xxx --merge_lora true

```

### **Summary of Next Steps**

1. **Run the `compare_models.py` script** (Phase 2) on 5-10 difficult clips.
2. If **Qwen2.5-VL-3B** is accurate -> **Use it directly** (cheapest path).
3. If it misses events -> **Collect 50-100 clips** and run the **Swift command** (Phase 3).

**Would you like me to generate a script to automatically download the "UCF-Crime" dataset sample so you have violent clips to test immediately?**