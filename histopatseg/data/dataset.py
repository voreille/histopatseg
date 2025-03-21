from PIL import Image
from torch.utils.data import Dataset


class TileDataset(Dataset):

    def __init__(self, tile_paths, transform=None):
        """
        Tile-level dataset that returns individual tile images from a list of paths.

        Args:
            tile_paths (list): List of paths to tile images for a WSI.
            transform (callable, optional): Transform to apply to each tile image.
        """
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        image = Image.open(tile_path).convert("RGB")  # Load as PIL image

        if self.transform:
            image = self.transform(image)  # Apply augmentation

        return image, self.tile_paths[idx].stem


class EmbeddingDataset(Dataset):

    def __init__(self, embeddings, labels):
        """
        Dataset to store precomputed embeddings and tile IDs.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            tile_ids (list): List of tile IDs.
        """
        self.embeddings = embeddings
        self.tile_ids = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.tile_ids[idx]
