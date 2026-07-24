import argparse
import json
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torchvision.transforms import ToPILImage



def save_tensor_image(tensor: torch.Tensor, filename: Path, to_pil: ToPILImage) -> None:
    """
    Save a tensor in [-1, 1] as a PNG image in [0, 1].
    """
    image = to_pil(tensor.clamp(-1, 1) * 0.5 + 0.5)
    image.save(filename)



def make_generator(device: torch.device, seed: int) -> torch.Generator:
    """
    Create a torch.Generator on the correct device and seed it.
    """
    if device.type == "cuda":
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator



def run_inference_for_seed(
    seed: int,
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
    output_root: Path,
    num_inference_steps: int,
    save_every: int,
) -> None:
    """
    Generate one image from a given seed and save the full reverse trajectory.

    Output structure:
        output_root/
            <seed>/
                final_sample<seed>.png
                final_sample<seed>.pt
                metadata.json
                trajectory_raw/
                    x_t_tensor_0000.pt
                    x_t_tensor_0001.pt
                    ...
                    x_t_tensor_0999.pt
                    x_t_tensor_1000.pt   # final x_0
    """
    seed_dir = output_root / str(seed)
    trajectory_dir = seed_dir / "trajectory_raw"
    seed_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    to_pil = ToPILImage()
    generator = make_generator(device, seed)

    image_size = model.config.sample_size
    channels = model.config.in_channels

    scheduler.set_timesteps(num_inference_steps)

    x_t = torch.randn(
        (1, channels, image_size, image_size),
        generator=generator,
        device=device,
    )

    saved_steps = []

    model.eval()
    with torch.inference_mode():
        for step_index, timestep in enumerate(scheduler.timesteps):
            if step_index % save_every == 0:
                checkpoint_path = trajectory_dir / f"x_t_tensor_{step_index:04d}.pt"
                torch.save(x_t[0].detach().cpu(), checkpoint_path)
                saved_steps.append(
                    {
                        "step_index": int(step_index),
                        "scheduler_timestep": int(timestep),
                        "file": checkpoint_path.name,
                    }
                )

            noise_pred = model(x_t, timestep).sample
            x_t = scheduler.step(
                noise_pred,
                timestep,
                x_t,
                generator=generator,
            ).prev_sample

    # Save final denoised sample x_0.
    final_tensor_path = seed_dir / f"final_sample{seed}.pt"
    final_image_path = seed_dir / f"final_sample{seed}.png"
    final_checkpoint_path = trajectory_dir / f"x_t_tensor_{num_inference_steps:04d}.pt"

    torch.save(x_t[0].detach().cpu(), final_tensor_path)
    torch.save(x_t[0].detach().cpu(), final_checkpoint_path)
    save_tensor_image(x_t[0].detach().cpu(), final_image_path, to_pil)

    metadata = {
        "seed": int(seed),
        "model_id": model.config._name_or_path if hasattr(model.config, "_name_or_path") else "unknown",
        "num_inference_steps": int(num_inference_steps),
        "save_every": int(save_every),
        "image_size": int(image_size),
        "channels": int(channels),
        "final_image": final_image_path.name,
        "final_tensor": final_tensor_path.name,
        "trajectory_directory": trajectory_dir.name,
        "saved_step_count": len(saved_steps),
        "saved_steps": saved_steps,
    }

    with open(seed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] seed={seed} -> saved to {seed_dir}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DDPM inference for one or more seeds and save the reverse trajectory."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="One or more seeds, e.g. --seeds 38 92 110 200",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/ddpm-celebahq-256",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="samples",
        help="Root directory where seed folders will be created.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of reverse diffusion steps.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help=(
            "Save one x_t tensor every N steps. "
            "Use 1 to save the full trajectory, 20 to save every 20 steps."
        ),
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model: {args.model_id}")

    model = UNet2DModel.from_pretrained(args.model_id).to(device)
    scheduler = DDPMScheduler.from_pretrained(args.model_id)

    for seed in args.seeds:
        run_inference_for_seed(
            seed=seed,
            model=model,
            scheduler=scheduler,
            device=device,
            output_root=output_root,
            num_inference_steps=args.num_inference_steps,
            save_every=args.save_every,
        )

    print("\nAll requested seeds have finished successfully.")


if __name__ == "__main__":
    main()
