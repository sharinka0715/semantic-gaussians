import cv2
import clip
import torch
import torchvision
import numpy as np
import time
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.automask import SamAutomaticMaskGenerator as MultiScaleMaskGenerator


class SAMCLIP:
    embedding_dim = 768
    
    def __init__(self, sam_path, clip_model_name):
        if sam_path is not None:
            print("Load SAM model...")
            sam = sam_model_registry["vit_h"](checkpoint=sam_path)
            sam.cuda()
            self.sam = SamPredictor(sam)
            self.mask_generator = MultiScaleMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.7,
                box_nms_thresh=0.7,
                stability_score_thresh=0.85,
                # crop_n_layers=1,
                # crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )

        if clip_model_name is not None:
            print("Loading CLIP {} model...".format(clip_model_name))
            self.clip_model, self.preprocess = clip.load(clip_model_name, device="cuda", jit=False)

    def set_predefined_cls(self, cls):
        self.classes = ".".join(cls)
        print(self.classes)

    def set_predefined_part(self, cls, parts):
        self.classes = ".".join([f"{cls}:{e}" for e in parts])
        print(self.classes)

    def get_text(self, vocabulary, prefix_prompt="a "):
        vocabulary = vocabulary.split(".")
        texts = [prefix_prompt + x.lower().replace(":", " ").replace("_", " ") for x in vocabulary]
        return texts

    def extract_image_feature(self, img_dir, img_size=None):
        """Extract per-pixel OpenSeg features.
        Only receives image path as input.
        """
        image = cv2.imread(str(img_dir))
        image = cv2.resize(image, (img_size[1], img_size[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks, masks_s, masks_m, masks_l = self.mask_generator.generate(image)

        sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        pad_imgs = []
        segs = []
        scores = []
        for mask in sorted_masks:
            bbox = mask["bbox"]
            seg_mask = mask["segmentation"]
            score = mask["stability_score"] * mask["predicted_iou"]
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            # h_thresh = int(image.shape[0] * 0.1)
            # w_thresh = int(image.shape[1] * 0.1)
            # if x2 < w_thresh or x1 > image.shape[1] - w_thresh or y2 < h_thresh or y1 > image.shape[0] - h_thresh:
            #     continue

            crop = (image * seg_mask[:, :, np.newaxis])[y1:y2, x1:x2]
            h, w, _ = crop.shape

            l = max(h, w)
            pad = np.zeros((l, l, 3), dtype=np.uint8)
            if h > w:
                pad[:, (h - w) // 2 : (h - w) // 2 + w, :] = crop
            else:
                pad[(w - h) // 2 : (w - h) // 2 + h, :, :] = crop
            pad_imgs.append(cv2.resize(pad, (336, 336)))
            segs.append(seg_mask)
            scores.append(score)

        if len(pad_imgs) == 0:
            print("Error: no mask detected!")
            return torch.zeros((768, image.shape[0], image.shape[1]), dtype=torch.half)

        pad_imgs = np.stack(pad_imgs, axis=0)  # B, H, W, 3
        pad_imgs = torch.from_numpy(pad_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0
        pad_imgs = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )(pad_imgs).cuda()

        crop_features = self.clip_model.encode_image(pad_imgs).cpu()
        features = torch.zeros((768, image.shape[0], image.shape[1]), dtype=torch.half)
        for idx, seg_mask in enumerate(segs):
            features[:, seg_mask] += crop_features[idx].unsqueeze(1)  # * scores[idx]

        features = features / (features.norm(dim=0, keepdim=True) + 1e-8)

        return features

    def extract_text_feature(self, labelset):
        # "ViT-L/14@336px" # the big model that OpenSeg uses
        if isinstance(labelset, str):
            lines = labelset.split(",")
        elif isinstance(labelset, list):
            lines = labelset
        else:
            raise NotImplementedError

        labels = []
        for line in lines:
            label = line
            labels.append(label)
        text = clip.tokenize(labels)
        text = text.cuda()
        text_features = self.clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features
