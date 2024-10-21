# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from lib.vision_transformer import VisionTransformer, _cfg

__all__ = [
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
    "deit_tiny_distilled_patch16_224",
    "deit_small_distilled_patch16_224",
    "deit_base_distilled_patch16_224",
    "deit_base_patch16_384",
    "deit_base_distilled_patch16_384",
]


class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dim)
        )  # torch.Size([1, 197, 384])

    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]

        x = self.patch_embed(x)  # torch.Size([32, 196, 384])
        pe = self.pos_embed  # torch.Size([1, 196, 384])
        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


# @register_model
# def deit_tiny_patch16_224(pretrained=False, **kwargs):
#     model = DeiT(
#         patch_size=16, embed_dim=192, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         ckpt = torch.load('deit_tiny_patch16_224-a1311bcf.pth')
#         model.load_state_dict(ckpt['model'], strict=False)

#     pe = model.pos_embed[:, 1:, :].detach()  # torch.Size([1, 196, 384])
#     pe = pe.transpose(-1, -2)  # torch.Size([1, 384, 196])
#     pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
#     # torch.Size([1, 384, 14, 14])
#     # pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)
#     # pe = F.interpolate(pe, size=(14, 14), mode='bilinear', align_corners=True)  # size = trainsize / patch_size
#     pe = F.interpolate(pe, size=(22, 22), mode='bilinear', align_corners=True) #追加10/05
#     # torch.Size([1, 384, 14, 14])
#     pe = pe.flatten(2)  # torch.Size([1, 384, 196])
#     pe = pe.transpose(-1, -2)  # torch.Size([1, 196, 384])
#     model.pos_embed = nn.Parameter(pe)
#     model.head = nn.Identity()
#     return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load("deit_tiny_patch16_224-a1311bcf.pth")
        model.load_state_dict(ckpt["model"])
        pe = model.pos_embed[:, 1:, :].detach()
    else:
        pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(
        pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2]))
    )
    pe = F.interpolate(pe, size=(22, 22), mode="bilinear", align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load("deit_small_patch16_224-cd65a155.pth")
        model.load_state_dict(ckpt["model"], strict=False)

    pe = model.pos_embed[:, 1:, :].detach()  # torch.Size([1, 196, 384])
    pe = pe.transpose(-1, -2)  # torch.Size([1, 384, 196])
    pe = pe.view(
        pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2]))
    )
    # torch.Size([1, 384, 14, 14])
    # pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)
    # pe = F.interpolate(pe, size=(14, 14), mode='bilinear', align_corners=True)  # size = trainsize / patch_size
    pe = F.interpolate(
        pe, size=(22, 22), mode="bilinear", align_corners=True
    )  # 追加10/05
    # torch.Size([1, 384, 14, 14])
    pe = pe.flatten(2)  # torch.Size([1, 384, 196])
    pe = pe.transpose(-1, -2)  # torch.Size([1, 196, 384])
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ############################################################################################
        checkpoint = torch.load("deit_base_patch16_224-b5f2ef4d.pth")
        checkpoint_model = checkpoint["model"]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # load_state_dict()のstrictをFalseにすると，ロード側のパラメータを設定する際，ロードされる側にしか存在しないものを無視する

        # print(checkpoint_model.keys()) # pthファイルのすべての重みkeyを表示

        print(msg)
        # IncompatibleKeys(missing_keys=[], unexpected_keys=[])はパラメータをロードした際に、
        # missing_keysは読み込んだpthファイルのモデルには存在しないが、ロード先のモデルには存在するパラメータ名
        # unexpected_keysは読み込んだpthファイルのモデルには存在するが、ロード先のモデルには存在しないパラメータ名
        # missing_keysもunexpected_keysも [ ] の場合、空リストなので、対応できなかったパラメータはなく、
        # 「きちんと学習済みモデルが読み込め、ロードできましたよ」ということを意味しています。
        # エラーメッセージではなく、情報の出力、そして中身も空リストなので、何の対応も必要ありません。
        ############################################################################################

    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(
        pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2]))
    )
    pe = F.interpolate(
        pe, size=(14, 14), mode="bilinear", align_corners=True
    )  # size = trainsize / patch_size
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = DeiT(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ############################################################################################
        # checkpoint = torch.load('deit_base_patch16_384-8de9b5d1.pth')
        checkpoint = torch.load("checkpoint-399_deit_l_352.pth")
        # checkpoint = torch.load('checkpoint-399l_224.pth')

        checkpoint_model = checkpoint["model"]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # load_state_dict()のstrictをFalseにすると，ロード側のパラメータを設定する際，ロードされる側にしか存在しないものを無視する

        # print(checkpoint_model.keys()) # pthファイルのすべての重みkeyを表示

        print(msg)
        # IncompatibleKeys(missing_keys=[], unexpected_keys=[])はパラメータをロードした際に、
        # missing_keysは読み込んだpthファイルのモデルには存在しないが、ロード先のモデルには存在するパラメータ名
        # unexpected_keysは読み込んだpthファイルのモデルには存在するが、ロード先のモデルには存在しないパラメータ名
        # missing_keysもunexpected_keysも [ ] の場合、空リストなので、対応できなかったパラメータはなく、
        # 「きちんと学習済みモデルが読み込め、ロードできましたよ」ということを意味しています。
        # エラーメッセージではなく、情報の出力、そして中身も空リストなので、何の対応も必要ありません。
        ############################################################################################

    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(
        pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2]))
    )

    #################################### # size = trainsize / patch_size################################
    # pe = F.interpolate(pe, size=(24, 24), mode='bilinear', align_corners=True)
    pe = F.interpolate(pe, size=(22, 22), mode="bilinear", align_corners=True)
    ###################################################################################################

    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model
