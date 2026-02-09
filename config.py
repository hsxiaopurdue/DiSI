import os
import torch

# -------------------------
# Training Configuration
# -------------------------
MODEL_ID = "CompVis/stable-diffusion-v1-4"
IMAGES_DIR = "./data/diffusion_images" 

# A100 80G Config
BATCH_SIZE = 32         
GRAD_ACCUM_STEPS = 1    
LEARNING_RATE = 1e-4    # <--- NOTE: LoRA usually needs a higher LR (1e-4) than full finetune (1e-5)
MAX_STEPS = 1000
SAVE_EVERY = 100
SEED = 42

# Image Config
IMAGE_SIZE = 512

# PSG & Voting Config
ENABLE_PSG = True            # Set to True to use the aggregation logic
USE_MAJORITY_VOTING = False  # Legacy voting logic

# --- Gap Optimizer Config (NEW) ---
ENABLE_GAP_OPTIMIZER = True # Enable the new Gap/Count logic
GAP_STEP = 1e-7              # Quantization step size
GAP_THRESHOLD = 0            # Threshold for gap count

OUTPUT_DIR = f"./sd14-finetuned-psg-lora-{ENABLE_PSG}-{ENABLE_GAP_OPTIMIZER}"

# LoRA Configuration
ENABLE_LORA = True      
LORA_RANK = 64          
LORA_ALPHA = 128        
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"] 

# Class Prompts (Mapping class ID to prompt)
CLASS_PROMPTS = {
    1: "a painting of Dunhuang",
    2: "a painting of mountain by artist Kuan Fan",
    3: "a painting by artist Kaizhi Gu",
    4: "a painting of bird by artist Di Li",
    5: "a painting of shrimps by artist Baishi Qi",
    6: "a painting of chrysanth by artist Changshuo Wu",
    7: "a painting of horse by artist Beihong Xu",
    8: "a painting of portrait by artist Liben Yan",
    9: "a painting of plum blossom by artist Wujiu Yang",
    10: "a painting of mountain by artist Mengfu Zhao",
    11: "a painting of bamboo by artist Banqiao Zheng",
    12: "a painting of lotus flowers by artist Daqian Zhang",
    13: "a painting of ink landscape by artist Baoshi Fu",
    14: "a painting of donkey by artist Zhou Huang",
}

# System
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16