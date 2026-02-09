import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler

# Import PEFT for LoRA
from peft import LoraConfig, get_peft_model

import config
import utils
from dataset import ImageClassDataset, collate_fn

def main():
    print(f"--- Starting Training on {config.DEVICE} ---")
    print(f"--- Mode: {'LoRA' if config.ENABLE_LORA else 'Full Fine-Tuning'} ---")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Load Components
    pipe = StableDiffusionPipeline.from_pretrained(
        config.MODEL_ID,
        safety_checker=None 
    ).to(config.DEVICE)

    # Freeze VAE and Text Encoder and move to BF16 (always safe)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.to(dtype=config.DTYPE)
    pipe.text_encoder.to(dtype=config.DTYPE)

    # 2. Configure UNet and LoRA
    if config.ENABLE_LORA:
        # --- LoRA Path ---
        # 1. Freeze base UNet
        pipe.unet.requires_grad_(False)
        
        # 2. Move base UNet to BFloat16 immediately (Saves 50% VRAM)
        pipe.unet.to(dtype=config.DTYPE)
        
        # 3. Apply LoRA Adapter
        lora_config = LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        
        # 4. Print stats
        pipe.unet.print_trainable_parameters()
        
    else:
        # --- Full Fine-Tuning Path ---
        pipe.unet.requires_grad_(True)
        # Keep Master Weights in Float32 for stability
        # Do not cast pipe.unet to BFloat16 here
    
    # 3. Setup Optimizer
    # Filter for only parameters that require gradients (works for both LoRA and Full)
    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE)
    
    noise_scheduler = DDPMScheduler.from_pretrained(config.MODEL_ID, subfolder="scheduler")

    # 4. Dataset
    dataset = ImageClassDataset(config.IMAGES_DIR, image_size=config.IMAGE_SIZE)
    loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        drop_last=True,
        collate_fn=collate_fn
    )

    print(f"Training Samples: {len(dataset)}")
    print(f"Batch Size: {config.BATCH_SIZE} | PSG: {config.ENABLE_PSG}")
    if config.ENABLE_PSG and config.ENABLE_GAP_OPTIMIZER:
        print(f"Gap Optimizer Enabled (Step={config.GAP_STEP}, T={config.GAP_THRESHOLD})")

    # 5. Training Loop
    global_step = 0
    pipe.unet.train()
    
    # Pre-fetch list of trainable params for PSG
    plist = [p for p in pipe.unet.parameters() if p.requires_grad]

    while global_step < config.MAX_STEPS:
        for batch in loader:
            if global_step >= config.MAX_STEPS:
                break
            
            # --- A. Prepare Inputs ---
            with torch.no_grad():
                tokens = pipe.tokenizer(
                    batch["caption"],
                    padding="max_length",
                    truncation=True,
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(config.DEVICE)
                
                text_embeddings = pipe.text_encoder(tokens)[0]
                
                # Inputs must be BF16
                imgs = batch["pixel_values"].to(config.DEVICE, dtype=config.DTYPE)
                latents = pipe.vae.encode(imgs).latent_dist.sample() * 0.18215
                
                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=config.DEVICE)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # --- B. Forward Pass (Mixed Precision) ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample
                loss_per_sample = F.mse_loss(model_pred.float(), noise.float(), reduction="none").mean(dim=[1, 2, 3])
            
            # --- C. Gradient Computation ---
            if config.ENABLE_PSG:
                grad_buffer = [] 
                
                for i in range(bsz):
                    utils.clear_grads(plist)
                    is_last = (i == bsz - 1)
                    
                    # Backward pass
                    loss_per_sample[i].backward(retain_graph=not is_last)
                    
                    # Flatten only trainable params (LoRA layers only if enabled)
                    flat_g = utils.flatten_grads(plist)
                    grad_buffer.append(flat_g)
                
                # Stack gradients [BSZ, NumParams]
                all_grads = torch.stack(grad_buffer)
                
                tags = batch["tag"].to(config.DEVICE)
                unique_tags = torch.unique(tags)
                tag_means = []

                # Aggregate per tag first
                for tag in unique_tags:
                    tag_mask = (tags == tag)
                    tag_grads = all_grads[tag_mask]
                    
                    # Per-tag aggregation strategy
                    if config.USE_MAJORITY_VOTING and tag_grads.shape[0] > 1:
                        # Voting within a single tag's samples
                        tag_avg = utils.majority_voting(tag_grads, config.DEVICE)
                    else:
                        # Standard mean within a single tag
                        tag_avg = tag_grads.mean(dim=0)
                    
                    tag_means.append(tag_avg)
                
                # Stack of representative gradients per tag [NumTags, NumParams]
                g_all_tag = torch.stack(tag_means, dim=0)

                if config.ENABLE_GAP_OPTIMIZER and len(tag_means) > 1:
                    # --- NEW GAP OPTIMIZER LOGIC ---
                    # We treat the 'tags' as the voting dimension (sample_dim=0)
                    outs = utils.has_value_count_gap_any_with_values_first_logic_2(
                        g_all_tag, 
                        step=config.GAP_STEP, 
                        t=config.GAP_THRESHOLD, 
                        sample_dim=0
                    )
                    
                    # Unpack outputs based on index
                    # Indices: 6=prefix_mask, 7=prefix_vals, 8=prefix_vals_second, 9=prefix_mask_second
                    prefix_mask = outs[6]
                    prefix_vals = outs[7]
                    prefix_vals_second = outs[8]
                    prefix_mask_second = outs[9]
                    
                    # Index 12 is gap_of_gaps
                    gap_of_gaps = outs[12]
                    
                    # Calculate selected averages based on masks
                    # Sum over tag dimension (dim=-1 after logic output reshape)
                    num_selected = prefix_mask.sum(dim=-1)
                    sum_selected = prefix_vals.sum(dim=-1)
                    avg_selected = sum_selected / num_selected.clamp_min(1)
                        
                    num_selected_2 = prefix_mask_second.sum(dim=-1)
                    sum_selected_2 = prefix_vals_second.sum(dim=-1)
                    avg_selected_second = sum_selected_2 / num_selected_2.clamp_min(1)
                    
                    # Decision logic based on gap_of_gaps
                    final_grad_avg = torch.where(
                        (gap_of_gaps >= 1).reshape(avg_selected.shape),
                        avg_selected,
                        avg_selected_second,
                    )
                else:
                    # Fallback to simple mean if optimizer off or only 1 tag
                    final_grad_avg = g_all_tag.mean(dim=0)
                
                # Assign final aggregated gradient back to params
                utils.assign_flat_grad_to_params(plist, final_grad_avg)

                # Cleanup
                del grad_buffer
                del all_grads
                del g_all_tag
                
            else:
                loss = loss_per_sample.mean()
                loss.backward()

            # --- D. Optimizer Step ---
            optimizer.step()
            optimizer.zero_grad()

            # --- E. Logging & Validation ---
            if global_step % 10 == 0:
                print(f"Step {global_step:05d} | Loss: {loss_per_sample.mean().item():.4f}")

            if global_step == 0 or global_step % config.SAVE_EVERY == 0:
                save_path = os.path.join(config.OUTPUT_DIR, f"checkpoint-{global_step}")
                pipe.unet.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
                
                print(f"Generating validation images for {len(config.CLASS_PROMPTS)} classes...")
                
                # Validation
                pipe.unet.eval()
                g = torch.Generator(device=config.DEVICE).manual_seed(2026)
                
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        for class_id, val_prompt in config.CLASS_PROMPTS.items():
                            val_img = pipe(
                                prompt=val_prompt, 
                                num_inference_steps=50, 
                                guidance_scale=7.5,
                                height=512, 
                                width=512, 
                                generator=g
                            ).images[0]
                            
                            out_filename = f"val_step{global_step:05d}_class{class_id:02d}.png"
                            out_path = os.path.join(config.OUTPUT_DIR, out_filename)
                            val_img.save(out_path)
                            print(f"[✓] Saved {out_filename}")
                
                pipe.unet.train()

            global_step += 1

    pipe.unet.save_pretrained(os.path.join(config.OUTPUT_DIR, "final"))
    print("Training Complete.")

if __name__ == "__main__":
    main()