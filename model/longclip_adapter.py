import torch
import torch.nn as nn
from model.adapter import MMadapter, DCShareAdapter
from typing import Tuple, Union
from model.model_longclip import ResidualAttentionBlock,Transformer,VisionTransformer,CLIP,convert_weights
import math

class ResidualAttentionBlock_DCAdapter(ResidualAttentionBlock):
    def __init__(self,d_model: int, n_head: int, attn_mask: torch.Tensor = None, Bi_adapter = None):
        super().__init__(d_model, n_head, attn_mask)
        self.adapter1 = MMadapter(None,hidden_size=d_model)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.adapter2 = MMadapter(Bi_adapter,hidden_size=d_model)
        self.gate2 = nn.Parameter(torch.tensor(0.6), requires_grad=True)

    def forward(self, x: torch.Tensor):
        xattn = self.attention(self.ln_1(x))
        alpha = torch.sigmoid(self.gate1)
        xattn =  alpha *self.adapter1(xattn) + (1 - alpha) * xattn
        x = x + xattn
        
        xmlp = self.mlp(self.ln_2(x)) 
        alpha = torch.sigmoid(self.gate2)
        xmlp =  alpha *self.adapter2(xmlp) + (1 - alpha) * xmlp
        x = x + xmlp
        return x
    
class Transformer_DCAdapter(Transformer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,Bi_adapter = None):
        super().__init__(width,layers,heads,attn_mask)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) if layer < 0
                                         else ResidualAttentionBlock_DCAdapter(width, heads, attn_mask,Bi_adapter[layer])
                                           for layer in range(layers)])
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer_DCAdapter(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,Bi_adapter = None):
        super().__init__(input_resolution,patch_size,width,layers,heads,output_dim)
        self.transformer = Transformer_DCAdapter(width, layers, heads,Bi_adapter=Bi_adapter)

class CLIP_DCAdapter(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 load_from_clip: bool
                 ):
        super().__init__(embed_dim,
                 # vision
                 image_resolution,
                 vision_layers,
                 vision_width,
                 vision_patch_size,
                 # text
                 context_length,
                 vocab_size,
                 transformer_width,
                 transformer_heads,
                 transformer_layers,
                 load_from_clip)
        vision_heads = vision_width // 64
        DCShareAdapters = nn.ModuleList([
                DCShareAdapter(128, 8)
                for _ in range(12)
            ])
        self.transformer = Transformer_DCAdapter(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            Bi_adapter=DCShareAdapters
        )
        self.visual = VisionTransformer_DCAdapter(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            Bi_adapter=DCShareAdapters
        )
        print("text encoder is ", self.transformer)
        print("img is ------------------", self.visual)
        

def build_adapter_model(state_dict: dict, load_from_clip: bool, adapter_type="conv"):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    
    if adapter_type == "DC":
        model = CLIP_DCAdapter(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, load_from_clip
        )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    load_result = model.load_state_dict(state_dict,strict=False)
    return model.eval()

def load_adapter_model(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         download_root: str = None, adapter_type ="conv"):

    model_path = name

    state_dict = torch.load(model_path, map_location="cpu")

    model = build_adapter_model(state_dict or model.state_dict(), load_from_clip=False,adapter_type=adapter_type).to(device)

    if str(device) == "cpu":
        model.float()

    return model
