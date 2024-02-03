import os
import cv2
import clip
import numpy as np
import supervision as sv
from tqdm import tqdm
from skimage.transform import resize

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from dataset.scannet.scannet_constants import SCANNET_CLASS_LABELS_200, SCANNET_CLASS_LABELS_20


class GroundingSAM:
    def __init__(
        self,
        grounding_dino_path,
        sam_path,
        text_model_name,
        box_threshold=0.25,
        text_threshold=0.25,
        nms_threshold=0.25,
        predefined_classes=list(SCANNET_CLASS_LABELS_20),
    ):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.classes = predefined_classes

        if grounding_dino_path is not None:
            print("Load GroundingDINO model...")
            self.grounding_dino = Model(
                model_config_path="./model/config/groundingdino_config.py", model_checkpoint_path=grounding_dino_path
            )

        if sam_path is not None:
            print("Load SAM model...")
            sam = sam_model_registry["vit_h"](checkpoint=sam_path)
            sam.cuda()
            self.sam = SamPredictor(sam)

        if text_model_name is not None:
            print("Loading CLIP {} model...".format(text_model_name))
            self.text_model, _ = clip.load(text_model_name, device="cuda", jit=False)

        self.text_features = self.extract_text_feature(self.classes).float()

    def extract_image_feature(self, img_dir, img_size=None):
        """Extract per-pixel GroundingSAM features."""
        image = cv2.imread(str(img_dir))
        detections = self.grounding_dino.predict_with_classes(
            image=image,
            classes=self.classes[:180],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        if detections.class_id.shape[0] == 0:
            return torch.zeros((768, img_size[1], img_size[0])).cuda()

        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence), self.nms_threshold
            )
            .numpy()
            .tolist()
        )
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        self.sam.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result_masks = []
        for box in detections.xyxy:
            masks, scores, logits = self.sam.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        detections.mask = np.array(result_masks)

        if img_size is not None:
            mask = resize(detections.mask.transpose(1, 2, 0), img_size, order=0, preserve_range=True).transpose(2, 0, 1)
        else:
            mask = detections.mask

        class_sem = self.text_features[detections.class_id]  # [num_boxes, 768]
        class_sem = class_sem * torch.from_numpy(detections.confidence).cuda().unsqueeze(-1)
        sem_map = torch.einsum("nc,nhw->chw", class_sem, torch.from_numpy(mask).float().cuda())
        sem_map = sem_map / (sem_map.norm(dim=0, keepdim=True) + 1e-8)

        return sem_map

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
        text_features = self.text_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features
