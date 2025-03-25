import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from histopatseg.constants import CLASS_MAPPING, SUBCLASS_MAPPING, SUPERCLASS_MAPPING


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


class LabeledTileDataset(Dataset):

    def __init__(self, tile_paths, metadata, transform=None):
        """
        Wrapper dataset that adds labels from metadata.
        Args:
            base_dataset (Dataset): Instance of TileDataset.
            metadata (pd.DataFrame): Metadata DataFrame with a "class_name" column indexed by tile_id.
        """
        self.base_dataset = TileDataset(tile_paths, transform=transform)
        self.metadata = metadata

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, tile_id = self.base_dataset[idx]
        # Look up the label for this tile_id
        label = CLASS_MAPPING[self.metadata.loc[tile_id, "class_name"]]
        return image, label


class EmbeddingDataset(Dataset):

    def __init__(self, embeddings, labels):
        """
        Dataset to store precomputed embeddings and tile IDs.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            tile_ids (list): List of tile IDs.
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class HierarchicalEmbeddingDataset(Dataset):

    def __init__(self, embeddings, tile_ids, metadata):
        """
        Dataset to store precomputed embeddings and tile IDs.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            tile_ids (list): List of tile IDs.
        """
        self.embeddings = embeddings
        self.tile_ids = tile_ids
        self.metadata = metadata

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        super_class = SUPERCLASS_MAPPING.get(self.metadata.loc[tile_id,
                                                               "superclass"])
        sub_class = SUBCLASS_MAPPING.get(
            self.metadata.loc[tile_id, "subclass"], -1)
        return self.embeddings[idx], super_class, sub_class


class EmbeddingDatasetMIL(Dataset):

    def __init__(self, embeddings, metadata):
        """
        Dataset to store precomputed embeddings and tile IDs.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            tile_ids (list): List of tile IDs.
        """
        self.embeddings = pd.DataFrame(embeddings)
        self.embeddings.index = metadata.index
        self.image_ids = metadata["original_filename"].values
        self.metadata = metadata

    @staticmethod
    def get_collate_fn_ragged():

        def collate_fn_ragged(batch):
            wsi_ids, embeddings, labels = zip(*batch)
            return list(wsi_ids), list(embeddings), torch.stack(labels)

        return collate_fn_ragged

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
