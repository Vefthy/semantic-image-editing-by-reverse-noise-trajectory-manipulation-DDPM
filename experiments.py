import os
import torch
import numpy as np

from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image


# ---------------- CHANGE THESE ----------------
recipient_seed = 38
donor_seed = 92
mask_path = "masks/mask_mouth.png"
# ----------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recipient_dir = f"samples/{recipient_seed}/trajectory_raw"
donor_dir = f"samples/{donor_seed}/trajectory_raw"
base_output_dir = f"experiments/{recipient_seed}-{donor_seed}"

# Load mask
x_t_ref = torch.load(
    f"{recipient_dir}/x_t_tensor_0000.pt"
).unsqueeze(0).to(device)

mask = Image.open(mask_path).convert("L")

mask_tensor = torch.from_numpy(
    np.array(mask, dtype=np.float32) / 255.0
).unsqueeze(0).unsqueeze(0).to(device)

mask_tensor = (mask_tensor >= 0.5).float()
mask_tensor = mask_tensor.expand_as(x_t_ref)

# Load model and scheduler
model_id = "google/ddpm-celebahq-256"
model = UNet2DModel.from_pretrained(
    model_id,
    use_safetensors=False,
).to(device)
model.eval()

scheduler = DDPMScheduler.from_pretrained(model_id)
scheduler.set_timesteps(scheduler.config.num_train_timesteps)


def save_tensor_image(tensor, filename):
    image = tensor.clamp(-1, 1) * 0.5 + 0.5
    image = image.mul(255).byte()
    image = image.permute(1, 2, 0).cpu().numpy()

    Image.fromarray(image).save(filename)


def exp_to_alpha(experiment_number):
    return (experiment_number + 1) / 10.0


# 10 methods: alpha 0.1, 0.2, ..., 1.0
experiment_numbers_to_run = list(range(10))

torch.manual_seed(recipient_seed)
os.makedirs(base_output_dir, exist_ok=True)

for experiment_number in experiment_numbers_to_run:
    alpha = exp_to_alpha(experiment_number)
    output_dir = f"{base_output_dir}/{experiment_number}"
    os.makedirs(output_dir, exist_ok=True)

# ---------------- CHANGE THIS ----------------
    for start_timestep_index in range(100, 900, 20):
# ----------------------------------------------
        timestep_str = f"{start_timestep_index:04d}"

        # A = donor, B = recipient
        x_t_A = torch.load(
            f"{donor_dir}/x_t_tensor_{timestep_str}.pt"
        ).unsqueeze(0).to(device)

        x_t_B = torch.load(
            f"{recipient_dir}/x_t_tensor_{timestep_str}.pt"
        ).unsqueeze(0).to(device)

        # Masked alpha blend
        x_t_B = (
            x_t_B * (1 - mask_tensor)
            + (alpha * x_t_A + (1 - alpha) * x_t_B) * mask_tensor
        )

        # Resume reverse diffusion
        for timestep in scheduler.timesteps[start_timestep_index:]:
            with torch.no_grad():
                noise_pred = model(x_t_B, timestep).sample
            x_t_B = scheduler.step(
                noise_pred,
                timestep,
                x_t_B,
            ).prev_sample

        save_tensor_image(
            x_t_B[0].detach().cpu(),
            os.path.join(
                output_dir,
                f"final_sample_from_{timestep_str}.png",
            ),
        )

    print(
        f"Done experiment {experiment_number} "
        f"(alpha={alpha:.1f}). Saved to {output_dir}"
    )
