import logging

import click
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from histopatseg.fewshot.protonet import ProtoNet
from histopatseg.models.foundation_models import load_model
from histopatseg.utils import get_device

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler()],  # Output logs to the console
)

logger = logging.getLogger(__name__)

# label_map = {
#     "Lepidic adenocarcinoma": 0,
#     "Acinar adenocarcinoma": 1,
#     "Papillary adenocarcinoma": 2,
#     "Solid adenocarcinoma": 3,
#     "Micropapillary adenocarcinoma": 5,
# }


def get_fitted_protonet(embeddings_train, labels_train, label_map=None):
    protonet = ProtoNet(label_map=label_map)
    protonet.fit(
        torch.tensor(embeddings_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long),
    )

    return protonet


def compute_embeddings(dataloader, model):
    embeddings_all = []
    labels_all = []
    device = next(model.parameters())[0].device

    for batch_idx, (batch, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.inference_mode():
            embeddings = model(batch.to(device, non_blocking=True))
            embeddings = embeddings.cpu().numpy()
            embeddings_all.append(embeddings)
            labels_all.append(target.cpu().numpy())

    return np.concatenate(embeddings_all, axis=0), np.concatenate(labels_all, axis=0)


@click.command()
@click.option(
    "--tiles-dir",
    type=click.Path(exists=True),
    help="Path to the directory containing tiles.",
)
@click.option("--output-path", required=True, help="Path to save the output ProtoNet")
@click.option("--model-name", default="UNI2", help="Model name to use for embeddings")
@click.option("--gpu-id", default=0, help="GPU ID to use.")
@click.option("--batch-size", default=256, help="Batch size for embedding computation")
@click.option("--num-workers", default=8, help="Number of workers for data loading.")
def main(tiles_dir, output_path, model_name, gpu_id, batch_size, num_workers):
    """Simple CLI program to greet someone"""

    device = get_device(gpu_id=gpu_id)
    model, preprocess, _, _ = load_model(model_name, device)

    logger.info(f"Using model {model_name} for embedding computation")
    logger.info(f"Using preprocess:\n {preprocess}")

    dataset = ImageFolder(tiles_dir, transform=preprocess)
    label_map = dataset.class_to_idx
    logger.info(f"Label map: {label_map}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    embeddings, labels = compute_embeddings(dataloader, model)

    protonet = get_fitted_protonet(embeddings, labels, label_map=label_map)
    protonet.save(output_path)


if __name__ == "__main__":
    main()
