from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class TileDatasetFromImage(Dataset):
    def __init__(self, image: Image.Image, tile_size: int = 256):
        self.tile_size = tile_size
        self.image = image
        self.tiles, self.coords = self._tile_image()
        self.transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _center_crop_to_multiple(self, image: Image.Image) -> Image.Image:
        W, H = image.size
        new_W = (W // self.tile_size) * self.tile_size
        new_H = (H // self.tile_size) * self.tile_size
        left = (W - new_W) // 2
        top = (H - new_H) // 2
        right = left + new_W
        bottom = top + new_H
        return image.crop((left, top, right, bottom))

    def _tile_image(self):
        image = self._center_crop_to_multiple(self.image)
        W, H = image.size
        tiles = []
        coords = []
        for y in range(0, H - self.tile_size + 1, self.tile_size):
            for x in range(0, W - self.tile_size + 1, self.tile_size):
                crop = image.crop((x, y, x + self.tile_size, y + self.tile_size))
                tiles.append(crop)
                coords.append([x, y])
        return tiles, coords

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        # Make sure coordinates are in the right format - should be [x, y] in pixels
        coord = torch.tensor(self.coords[idx], dtype=torch.float)
        return self.transform(tile), coord