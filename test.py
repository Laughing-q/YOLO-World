from yolo_world.models.backbones import HuggingCLIPLanguageBackbone
import torch
import clip

text = [["person"]]
huggingclip = HuggingCLIPLanguageBackbone(model_name="./third_party/clip-vit-base-patch32", frozen_modules=["all"])
print(huggingclip(text).dtype)

text = clip.tokenize(["person"]).cuda()
model, preprocess = clip.load("ViT-B/32", device="cuda")
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features.reshape(-1, 1, text_features.shape[-1])
    print(text_features.dtype)
