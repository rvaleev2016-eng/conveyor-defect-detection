import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from src.clearml_utils import init_task, report_image, upload_if_exists
from src.visualization import ensure_dir, save_prediction_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net inference on custom images")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_root", default="results")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = get_device()
    task = init_task(
        project_name="Conveyor Defect Detection",
        task_name=f"Inference U-Net ({args.category})",
        tags=["unet", "segmentation", "inference", args.category],
        params=vars(args),
    )

    model_path = Path("models") / f"unet_{args.category}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    output_dir = Path(args.output_root) / args.category / "inference"
    ensure_dir(output_dir)

    image_paths = sorted(Path(args.images_dir).glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No .png images found in {args.images_dir}")

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            probability = torch.sigmoid(model(tensor))
            prediction = (probability > args.threshold).float()

            save_prediction_figure(
                image=tensor.squeeze(0).cpu(),
                target_mask=None,
                predicted_mask=prediction.squeeze(0).cpu(),
                path=output_dir / f"{image_path.stem}_prediction.png",
                title=f"Inference: {image_path.name}",
            )
            output_path = output_dir / f"{image_path.stem}_prediction.png"
            report_image(task, "Inference Examples", image_path.stem, output_path, iteration=index - 1)
            upload_if_exists(task, f"{image_path.stem}_prediction_png", output_path)

            print(f"[{index}/{len(image_paths)}] Saved {image_path.stem}_prediction.png")

    task.close()


if __name__ == "__main__":
    main()
