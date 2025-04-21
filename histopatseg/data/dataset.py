import math

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

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
        super_class = SUPERCLASS_MAPPING.get(self.metadata.loc[tile_id, "superclass"])
        sub_class = SUBCLASS_MAPPING.get(self.metadata.loc[tile_id, "subclass"], -1)
        return self.embeddings[idx], super_class, sub_class


class EmbeddingDatasetMIL(Dataset):
    def __init__(self, embeddings, tile_ids, metadata, class_mapping=CLASS_MAPPING):
        """
        Dataset to store precomputed embeddings and tile IDs.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            tile_ids (list): List of tile IDs.
        """
        self.embeddings = pd.DataFrame(embeddings, index=tile_ids)
        self.metadata = metadata.loc[tile_ids].copy()
        self.tile_ids = tile_ids
        df_tmp = self.metadata.groupby("original_filename").agg({"class_name": "first"})
        self.image_ids = df_tmp.index.values
        labels = df_tmp["class_name"].values
        self.labels = [class_mapping[label] for label in labels]
        self.class_mapping = class_mapping

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            wsi_ids, embeddings, labels = zip(*batch)
            return list(wsi_ids), list(embeddings), torch.stack(labels)

        return collate_fn_ragged

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        tile_ids = self.metadata[self.metadata["original_filename"] == self.image_ids[idx]].index
        embeddings = self.embeddings.loc[tile_ids, :].values
        label = self.labels[idx]

        return (
            image_id,
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


class TileDatasetMILAlbumentation(Dataset):
    def __init__(
        self,
        image_ids,
        metadata,
        transform=None,
        class_mapping=CLASS_MAPPING,
        cache_data=False,
    ):
        """
        Dataset to store precomputed embeddings and tile IDs.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            tile_ids (list): List of tile IDs.
        """
        self.image_ids = image_ids
        self.metadata = metadata.copy()
        df_tmp = self.metadata.groupby("original_filename").agg({"class_name": "first"})
        labels = df_tmp.loc[image_ids, "class_name"].values
        self.labels = [class_mapping[label] for label in labels]
        self.class_mapping = class_mapping
        self.transform = transform
        self.replay_transform = isinstance(transform, A.ReplayCompose)

        self.cached_tiles = None
        if cache_data:
            self.create_cache()

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            wsi_ids, embeddings, labels = zip(*batch)
            return list(wsi_ids), list(embeddings), torch.stack(labels)

        return collate_fn_ragged

    def __len__(self):
        return len(self.image_ids)

    def create_cache(self):
        self.cached_tiles = []
        for image_id in self.image_ids:
            self.cached_tiles.append(self._load_tiles_without_transform(image_id))

    def _load_tiles_without_transform(self, image_id):
        tile_ids = self.metadata[self.metadata["original_filename"] == image_id].index

        tiles = []
        for tile_id in tile_ids:
            tile_path = self.metadata.loc[tile_id, "tile_path"]
            image = Image.open(tile_path).convert("RGB")
            tiles.append(np.array(image))
        return tiles

    def _apply_transform(self, tiles):
        output = []
        for tile in tiles:
            output.append(self.transform(tile))
        return torch.stack(output)

    def _load_tiles(self, image_id):
        tile_ids = self.metadata[self.metadata["original_filename"] == image_id].index

        tiles = []
        for tile_id in tile_ids:
            tile_path = self.metadata.loc[tile_id, "tile_path"]
            image = Image.open(tile_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            tiles.append(image)
        return torch.stack(tiles)

    def _get_item_with_replay(self, idx):
        image_id = self.image_ids[idx]
        if self.cached_tiles:
            tiles = self.cached_tiles[idx]
        else:
            tiles = self._load_tiles_without_transform(image_id)

        # Apply the replayable transform on the first tile
        first_transformed_tile = self.transform(image=tiles[0])
        transformed_tiles = [first_transformed_tile["image"]]

        # Replay the exact same transformation on the remaining tiles
        for tile in tiles[1:]:
            transformed = A.ReplayCompose.replay(first_transformed_tile["replay"], image=tile)[
                "image"
            ]
            transformed_tiles.append(transformed)

        tiles_tensor = torch.stack(transformed_tiles)
        label = self.labels[idx]
        return image_id, tiles_tensor, torch.tensor(label, dtype=torch.long)

    def _get_item_without_replay(self, idx):
        image_id = self.image_ids[idx]
        if self.cached_tiles:
            tiles = self.cached_tiles[idx]
        else:
            tiles = self._load_tiles_without_transform(image_id)

        # Replay the exact same transformation on the remaining tiles
        transformed_tiles = []
        for tile in tiles:
            transformed_tiles.append(self.transform(image=tile)["image"])

        tiles_tensor = torch.stack(transformed_tiles)
        label = self.labels[idx]
        return image_id, tiles_tensor, torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        if self.replay_transform:
            return self._get_item_with_replay(idx)

        return self._get_item_without_replay(idx)


class TileDatasetMIL(Dataset):
    def __init__(
        self,
        image_ids,
        metadata,
        transform=None,
        class_mapping=CLASS_MAPPING,
        cache_data=False,
    ):
        """
        Dataset to store precomputed embeddings and tile IDs.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            tile_ids (list): List of tile IDs.
        """
        self.image_ids = image_ids
        self.metadata = metadata.copy()
        df_tmp = self.metadata.groupby("original_filename").agg({"class_name": "first"})
        labels = df_tmp.loc[image_ids, "class_name"].values
        self.labels = [class_mapping[label] for label in labels]
        self.class_mapping = class_mapping
        self.transform = transform

        self.cached_tiles = None
        if cache_data:
            self.create_cache()

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            wsi_ids, embeddings, labels = zip(*batch)
            return list(wsi_ids), list(embeddings), torch.stack(labels)

        return collate_fn_ragged

    def __len__(self):
        return len(self.image_ids)

    def create_cache(self):
        self.cached_tiles = []
        for image_id in self.image_ids:
            self.cached_tiles.append(self._load_tiles_without_transform(image_id))

    def _load_tiles_without_transform(self, image_id):
        tile_ids = self.metadata[self.metadata["original_filename"] == image_id].index

        tiles = []
        for tile_id in tile_ids:
            tile_path = self.metadata.loc[tile_id, "tile_path"]
            image = Image.open(tile_path).convert("RGB")
            tiles.append(image)
        return tiles

    def _apply_transform(self, tiles):
        output = []
        for tile in tiles:
            output.append(self.transform(tile))
        return torch.stack(output)

    def _load_tiles(self, image_id):
        tile_ids = self.metadata[self.metadata["original_filename"] == image_id].index

        tiles = []
        for tile_id in tile_ids:
            tile_path = self.metadata.loc[tile_id, "tile_path"]
            image = Image.open(tile_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            tiles.append(image)
        return torch.stack(tiles)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.cached_tiles:
            tiles = self.cached_tiles[idx]
        else:
            tiles = self._load_tiles_without_transform(image_id)

        # Replay the exact same transformation on the remaining tiles
        transformed_tiles = []
        for tile in tiles:
            transformed_tiles.append(self.transform(tile))

        tiles_tensor = torch.stack(transformed_tiles)
        label = self.labels[idx]
        return image_id, tiles_tensor, torch.tensor(label, dtype=torch.long)


class LungHist700ImageDataset(Dataset):
    def __init__(self, image_paths, metadata, transform=None, class_mapping=CLASS_MAPPING):
        """
        Tile-level dataset that returns individual tile images from a list of paths.

        Args:
            tile_paths (list): List of paths to tile images for a WSI.
            transform (callable, optional): Transform to apply to each tile image.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.metadata = metadata
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            images, labels = zip(*batch)
            return list(images), torch.stack(labels)

        return collate_fn_ragged

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = image_path.stem
        image = Image.open(image_path).convert("RGB")  # Load as PIL image

        if self.transform:
            image = self.transform(image)  # Apply augmentation

        label = self.class_mapping[self.metadata.loc[image_id]["class_name"]]

        return image, torch.tensor(label, dtype=torch.long)


class MILDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=None, tile_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.tile_size = tile_size

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            tile_bags, labels = zip(*batch)
            return list(tile_bags), torch.stack(labels)

        return collate_fn_ragged

    def calculate_stride(self, image_dim):
        """Calculate stride to align tiles symmetrically with borders."""
        if image_dim <= self.tile_size:
            raise ValueError("Image dimension must be larger than tile size")

        n_tiles = math.ceil(image_dim / self.tile_size)
        if n_tiles == 1:
            return 0  # Single tile, no stride needed
        total_stride_space = image_dim - self.tile_size * n_tiles
        stride = self.tile_size + total_stride_space // (n_tiles - 1)
        return int(stride)

    def tile_image(self, image: Image.Image) -> torch.Tensor:
        """
        Tile a PIL image into patches of shape (N, C, tile_size, tile_size).

        Args:
            image (PIL.Image): Input image of shape (H, W, 3)

        Returns:
            torch.Tensor: Tensor of tiles (num_tiles, C, tile_size, tile_size)
        """
        # Convert PIL image to tensor: [C, H, W]
        image_tensor = TF.to_tensor(image)  # range [0, 1]
        C, H, W = image_tensor.shape

        # Compute symmetric stride
        stride_y = self.calculate_stride(H)
        stride_x = self.calculate_stride(W)

        # Tiling with unfold
        tiles = image_tensor.unfold(1, self.tile_size, stride_y).unfold(
            2, self.tile_size, stride_x
        )
        # shape: [C, n_tiles_y, n_tiles_x, tile_size, tile_size]

        tiles = tiles.permute(1, 2, 0, 3, 4).contiguous()
        # shape: [n_tiles_y, n_tiles_x, C, tile_size, tile_size]

        tiles = tiles.view(-1, C, self.tile_size, self.tile_size)
        # shape: [num_tiles, C, tile_size, tile_size]

        return tiles

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load image as Tensor [C, H, W]
        image = Image.open(image_path).convert("RGB")  # Load as PIL image

        if self.augment:
            image = self.augment(image)

        tiles = self.tile_image(image)
        if self.transform:
            tiles = self.transform(tiles)

        return tiles, label


class TileOnTheFlyDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None, augment=None, tile_size=224):
        self.image_paths = image_paths
        self.transform = transform
        self.augment = augment
        self.tile_size = tile_size

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            tile_bags = zip(*batch)
            return list(tile_bags)

        return collate_fn_ragged

    def calculate_stride(self, image_dim):
        """Calculate stride to align tiles symmetrically with borders."""
        if image_dim <= self.tile_size:
            raise ValueError("Image dimension must be larger than tile size")

        n_tiles = math.ceil(image_dim / self.tile_size)
        if n_tiles == 1:
            return 0  # Single tile, no stride needed
        total_stride_space = image_dim - self.tile_size * n_tiles
        stride = self.tile_size + total_stride_space // (n_tiles - 1)
        return int(stride)

    def tile_image(self, image: Image.Image) -> torch.Tensor:
        """
        Tile a PIL image into patches of shape (N, C, tile_size, tile_size).

        Args:
            image (PIL.Image): Input image of shape (H, W, 3)

        Returns:
            torch.Tensor: Tensor of tiles (num_tiles, C, tile_size, tile_size)
        """
        # Convert PIL image to tensor: [C, H, W]
        image_tensor = TF.to_tensor(image)  # range [0, 1]
        C, H, W = image_tensor.shape

        # Compute symmetric stride
        stride_y = self.calculate_stride(H)
        stride_x = self.calculate_stride(W)

        # Tiling with unfold
        tiles = image_tensor.unfold(1, self.tile_size, stride_y).unfold(
            2, self.tile_size, stride_x
        )
        # shape: [C, n_tiles_y, n_tiles_x, tile_size, tile_size]

        tiles = tiles.permute(1, 2, 0, 3, 4).contiguous()
        # shape: [n_tiles_y, n_tiles_x, C, tile_size, tile_size]

        tiles = tiles.view(-1, C, self.tile_size, self.tile_size)
        # shape: [num_tiles, C, tile_size, tile_size]

        return tiles

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Load image as Tensor [C, H, W]
        image = Image.open(image_path).convert("RGB")  # Load as PIL image

        if self.augment:
            image = self.augment(image)

        tiles = self.tile_image(image)
        if self.transform:
            tiles = self.transform(tiles)

        return tiles
