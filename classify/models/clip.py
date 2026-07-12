"""
Code mainly from https://github.com/vturrisi/disef/blob/main/fine-tune/src/model.py
"""

import types
import torch
import torch.nn as nn
import clip
from timm.models._manipulate import checkpoint_seq

from models.lora import lora_replace_attention_layers
from util_data import SUBSET_NAMES, TEMPLATES_SMALL


# 手部实验固定使用“整体软组织 + 第一骨间肌”两类可见征象提示词。
# 每个类别的两条文本特征取平均；zero-shot 与 LoRA 共用同一组文本原型。
# HAND_CLASS_PROMPTS = {
#     "malnourished_hand": [
#         "a clinical photograph of thin and bony hands with reduced soft tissue",
#         "a clinical photograph of hands showing first dorsal interosseous muscle wasting between the thumb and index finger",
#     ],
#     "normal_hand": [
#         "a clinical photograph of healthy hands with normal soft tissue",
#         "a clinical photograph of hands showing preserved first dorsal interosseous muscle bulk between the thumb and index finger",
#     ],
# }
# HAND_CLASS_PROMPTS = {
#     "malnourished_hand": [
#         "a clinical photograph of thin and bony hands with reduced soft tissue",
#         "a clinical photograph of hands showing reduced fullness in the web space between the thumb and index finger",
#     ],
#     "normal_hand": [
#         "a clinical photograph of healthy hands with normal soft tissue",
#         "a clinical photograph of hands showing preserved fullness in the web space between the thumb and index finger",
#     ],
# }

HAND_CLASS_PROMPTS = {
    "malnourished_hand": [
        "a clinical photograph of thin and bony hands with reduced soft tissue",
        "a close-up clinical image of wasted hands with reduced muscle and fat",
        "a clinical photograph of hands showing reduced fullness in the web space between the thumb and index finger",
    ],
    "normal_hand": [
        "a clinical photograph of healthy hands with normal soft tissue",
        "a close-up clinical image of healthy hands with normal muscle and fat",
        "a clinical photograph of hands showing preserved fullness in the web space between the thumb and index finger",
    ],
}


def get_class_texts(dataset, classname, dataset_name, templates):
    """手部返回固定提示词，其他数据集保持原有模板构造方式。"""
    if dataset == "hand_nutrition" and classname in HAND_CLASS_PROMPTS:
        return list(HAND_CLASS_PROMPTS[classname])
    return [template.format(dataset_name, classname) for template in templates]

def get_dataset_name_for_template(dataset):
    dataset_name = {
        "imagenet_100": "",
        "imagenet": "",
        "std10": "",
        "pets": "pet ",
        "fgvc_aircraft": "aircraft ",
        "cars": "car ",
        "eurosat": "satellite ",
        "dtd": "texture ",
        "flowers102": "flower ",
        "food101": "food ",
        "sun397": "scene ",
        "caltech101": "",
        # 原始手部映射："my_dataset": "human ",
        "hand_nutrition": "human ",
    }[dataset]
    return dataset_name


class CLIP(nn.Module):
    def __init__(
        self, 
        dataset,
        is_lora_image,
        is_lora_text,
        clip_download_dir="model_clip",
        clip_version="ViT-B/16",
        use_roi_aux_head: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = get_dataset_name_for_template(dataset)
        self.is_lora_image = is_lora_image
        self.is_lora_text = is_lora_text
        self.clip_version = clip_version
        self.use_roi_aux_head = use_roi_aux_head

        # TODO: change the number of templates
        self.templates = TEMPLATES_SMALL[:1]

        self.clip, _ = clip.load(clip_version, device="cpu", download_root=clip_download_dir)

        # visual model
        if is_lora_image:
            if self.clip_version != "RN50":
                self.clip.visual.transformer = lora_replace_attention_layers(
                    self.clip.visual.transformer,
                    lora_r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    start_block=0,
                )

        # text model
        if is_lora_text:
            self.clip.transformer = lora_replace_attention_layers(
                self.clip.transformer,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                start_block=0,
            )

        self.register_buffer("tokenized_text", self.tokenize_text())

        if self.use_roi_aux_head:
            # ROI 辅助头预测 [temporal, orbital, malar, jawline] 4 维描述符。
            image_feature_dim = int(getattr(self.clip.visual, "output_dim", 512))
            self.roi_head = nn.Sequential(
                nn.Linear(image_feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
            )

        # enable checkpointing for text transformer
        # datasets with more classes simply go OOM if we don't do this
        def checkpoint_forward(self, x):
            x.requires_grad = True
            x = checkpoint_seq(self.resblocks, x)
            return x

        self.clip.transformer.forward = types.MethodType(
            checkpoint_forward, self.clip.transformer)

        # configure all learnable parameters
        self.set_learnable_params()

#     @staticmethod
    def tokenize_text(self):
        print("Tokenizing text...")

        texts = []
        for classname in SUBSET_NAMES[self.dataset]:

            # 原始提示构造逻辑保留如下：
            # class_texts = []
            # for template in self.templates:
            #     class_texts.append(template.format(self.dataset_name, classname))
            # 手部类别使用固定且可读的自然语言提示，其他类别仍由原模板生成。
            class_texts = get_class_texts(
                self.dataset, classname, self.dataset_name, self.templates
            )
            print(f"类别 {classname} 的CLIP提示: {class_texts}")

            class_texts = clip.tokenize(class_texts)

            texts.append(class_texts)

        texts = torch.stack(texts)
        return texts

#   原代码！！ 
    def set_learnable_params(self):
        # turn off all parameters
        for p in self.clip.parameters():
            p.requires_grad = False

        # learnable parameters for the visual model
        if self.is_lora_image:
            if self.clip_version != "RN50":
                for name, p in self.clip.visual.named_parameters():
                    if "lora_" in name:
                        p.requires_grad = True
            else:
                for name, p in self.clip.visual.named_parameters():
                    p.requires_grad = True
                
#         elif not self.cfg.freeze_visual:
#             for p in self.clip.visual.parameters():
#                 p.requires_grad = True

        # learnable parameters for the text model
        if self.is_lora_text:
            for name, p in self.clip.transformer.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

#   新代码！！ --对应以下这个命令.对CLIP不使用Lora微调。
#   PARAM="--is_synth_train=False --is_lora_image=False --is_lora_text=False" bash 04_classify.sh
    # def set_learnable_params(self):
    #     # turn off all parameters
    #     for p in self.clip.parameters():
    #         p.requires_grad = False
    #     # learnable parameters for the visual model
    #     if self.is_lora_image:
    #         if self.clip_version != "RN50":
    #             for name, p in self.clip.visual.named_parameters():
    #                 if "lora_" in name:
    #                     p.requires_grad = True
    #         else:
    #             for name, p in self.clip.visual.named_parameters():
    #                 p.requires_grad = True
    #     else:
    #         # 不用LoRA时解冻整个视觉编码器进行全参数微调
    #         for p in self.clip.visual.parameters():
    #             p.requires_grad = True
    #     # learnable parameters for the text model
    #     if self.is_lora_text:
    #         for name, p in self.clip.transformer.named_parameters():
    #             if "lora_" in name:
    #                 p.requires_grad = True
    #     else:
    #         # 不用LoRA时解冻整个文本编码器进行全参数微调
    #         for p in self.clip.transformer.parameters():
    #             p.requires_grad = True


    def learnable_params(self):
#         return [{"name": "all", "params": [p for p in self.clip.parameters() if p.requires_grad]}]
        params = [p for p in self.clip.parameters() if p.requires_grad]
        if self.use_roi_aux_head:
            params += list(self.roi_head.parameters())
        return params

    def forward_image(
        self,
        x: torch.Tensor,
    ):
        image_feats = self.clip.visual(x)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        return image_feats

    def forward_text(self, tokenized_text):
        n_classes, n_prompts, n_token = tokenized_text.size()
#         tokenized_text = einops.rearrange(tokenized_text, "c p d -> (c p) d")
        tokenized_text = tokenized_text.view(-1, n_token)
        with torch.set_grad_enabled(self.is_lora_text):
            text_feats = self.clip.encode_text(tokenized_text)

        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # average across multiple prompt templates and re-norm
#         text_feats = einops.rearrange(text_feats, "(c p) d -> c p d", c=n_classes, p=n_prompts)
        text_feats = text_feats.view(n_classes, n_prompts, -1)
        text_feats = text_feats.mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        return text_feats

    def forward(
        self,
        x: torch.Tensor,
        tokenized_text: torch.Tensor = None,
        output_features: bool = False,
        **kwargs,
    ):
        if tokenized_text is None:
            tokenized_text = self.tokenized_text

        image_feats = self.forward_image(x)
        text_feats = self.forward_text(tokenized_text)

        logit_scale = self.clip.logit_scale.exp()

        # no instance-specific text feats
        if len(text_feats.shape) == 2:
            # cosine similarity as logits
            logits_per_image = logit_scale * image_feats @ text_feats.t()
        else:
            logits_per_image = logit_scale * torch.stack(
                [image_feats[i] @ text_feats[i].t() for i in range(image_feats.shape[0])]
            )

        if self.use_roi_aux_head:
            roi_pred = self.roi_head(image_feats)
            if output_features:
                return {
                    "logits": logits_per_image,
                    "roi_pred": roi_pred,
                    "image_feats": image_feats,
                    "text_feats": text_feats,
                }
            return logits_per_image, roi_pred

        if output_features:
            return {
                "logits": logits_per_image,
                "image_feats": image_feats,
                "text_feats": text_feats,
            }

        return logits_per_image

