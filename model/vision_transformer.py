""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013
The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision
Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020, Ross Wightman
"""

"""I modified original code and add comments to make easy to review the paper and code
So I removed all features about regularization and additional analyses
Copyright 2023, Minsik Yoon
"""

from functools import partial

import torch
from torch import nn
from torch import _assert

from .util import to_2tuple, trunc_normal_, named_apply, init_weights_vit_timm


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768, # embed_dim == D
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size # patch_size == P
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        """Splitting an image into patches and linear embedding are implemented by convolution
        Unlike the paper, flatten later instead of flattening first
        """
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x) # Size([B, D, H//P, W//P])
        if self.flatten:
            x = x.flatten(2) # Size([B, D, N']), N' == (H//P)(W//P)
            x = x.transpose(1, 2) # Size([B, N', D])
        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads # num_heads == h
        head_dim = dim // num_heads # head_dim == d
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x) # Size([B, N, 3*D])
        qkv = qkv.reshape(B, N, 3, self.num_heads, D // self.num_heads) # Size([B, N, 3, h, d])
        # B and h are dimensions to be broadcasted
        qkv = qkv.permute(2, 0, 3, 1, 4) # Size([3, B, h, N, d])
        # q, k, v have Size([B, h, N, d]) respectively
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # Size([B, h, N, N])
        attn = attn.softmax(dim=-1)

        x = (attn @ v) # Size([B, h, N, d])
        x = x.transpose(1, 2).reshape(B, N, D) # Size([B, N, D])
        x = self.proj(x) # Size([B, N, D])
        return x


class MLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias = True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerEncorder(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    """In the paper, there are no explicit mention about bias in linear transformation (expressed by matrix multiplication in the paper)
    the official implementation use bias in linear transformation
        - https://github.com/google-research/vision_transformer
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            embed_layer=PatchEmbed,
            block_fn=TransformerEncorder,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            embed_layer (nn.Module): patch embedding layer
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        embed_len = num_patches + 1 # patches and class token
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier Head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    def reset_classifier(self, num_classes: int):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) # Size([B, N, D]), N = N' + 1
        x = x + self.pos_embed
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = x[:, 0] # class token
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
