import os

from dotenv import load_dotenv
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torchvision import transforms

load_dotenv()


def load_model(model_name, device):
    """Load the model dynamically based on the model name."""

    if model_name == "bioptimus":
        login(token=os.getenv("HUGGING_FACE_TOKEN"))
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0",
                                  pretrained=True,
                                  init_values=1e-5,
                                  dynamic_img_size=False)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.707223, 0.578729, 0.703617),
                                 std=(0.211883, 0.230117, 0.177517)),
        ])
        embedding_dim = 1536

    elif model_name == "UNI2":
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h",
                                  pretrained=True,
                                  **timm_kwargs)
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model))
        embedding_dim = 1536
    else:
        raise ValueError(f"No model for {model_name}.")
        # # Load your custom model from local weights
        # model = load_local_model(weights_path,
        #                          device)  # Your existing function
        # preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        # embedding_dim = 2048

    model.to(device)
    model.eval()
    return model, transform, embedding_dim
