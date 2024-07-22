import os
import cv2
import clip
import numpy as np
from skimage.transform import resize

import torch

from model.vlpart.vlpart import build_vlpart
import detectron2.data.transforms as T
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.amg import remove_small_regions
from model.vlpart.vocab import LVIS_PACO_VOCAB, PASCAL_PART_VOCAB


class VLPart:
    embedding_dim = 768
    
    def __init__(
        self,
        vlpart_path,
        sam_path,
        text_model_name,
        box_threshold=0.3,
        predefined_classes=list(LVIS_PACO_VOCAB) + list(PASCAL_PART_VOCAB),
    ):
        self.box_threshold = box_threshold
        self.classes = ".".join(predefined_classes)

        if vlpart_path is not None:
            print("Load VLPart model...")
            self.vlpart = build_vlpart(vlpart_path)

        if sam_path is not None:
            print("Load SAM model...")
            sam = build_sam(checkpoint=sam_path).to("cuda")
            self.sam = SamPredictor(sam)

        if text_model_name is not None:
            print("Loading CLIP {} model...".format(text_model_name))
            self.text_model, _ = clip.load(text_model_name, device="cuda", jit=False)
            self.text_model.eval()

        self.text_features = self.extract_text_feature(self.get_text(self.classes)).float()

    def set_predefined_cls(self, cls):
        self.classes = ".".join(cls)
        print(self.classes)
        self.text_features = self.extract_text_feature(self.get_text(self.classes)).float()

    def set_predefined_part(self, cls, parts):
        self.classes = ".".join([f"{cls}:{e}" for e in parts])
        print(self.classes)
        self.text_features = self.extract_text_feature(self.get_text(self.classes)).float()

    def get_text(self, vocabulary, prefix_prompt="a "):
        vocabulary = vocabulary.split(".")
        texts = [prefix_prompt + x.lower().replace(":", " ").replace("_", " ") for x in vocabulary]
        texts_aug = texts + ["background"]
        return texts_aug

    def extract_image_feature(self, img_dir, img_size=None):
        """Extract per-pixel GroundingSAM features."""
        image = cv2.imread(str(img_dir))
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # vlpart model inference
        preprocess = T.ResizeShortestEdge([800, 800], 1333)
        height, width = original_image.shape[:2]
        image = preprocess.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        with torch.no_grad():
            predictions = self.vlpart.inference([inputs], text_prompt=self.classes)[0]

        boxes, masks = None, None
        filter_scores, filter_boxes, filter_classes = [], [], []

        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

            num_obj = len(scores)
            for obj_ind in range(num_obj):
                category_score = scores[obj_ind]
                if category_score < self.box_threshold:
                    continue
                filter_scores.append(category_score)
                filter_boxes.append(boxes[obj_ind])
                filter_classes.append(classes[obj_ind])

        if len(filter_boxes) > 0:
            # sam model inference
            self.sam.set_image(original_image)

            boxes_filter = torch.stack(filter_boxes)
            transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_filter, original_image.shape[:2])
            masks, _, _ = self.sam.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to("cuda"),
                multimask_output=False,
            )

            # remove small disconnected regions and holes
            fine_masks = []
            for mask in masks.to("cpu").numpy():  # masks: [num_masks, 1, h, w]
                fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
            masks = np.stack(fine_masks, axis=0)
        else:
            return torch.zeros((768, img_size[0], img_size[1])).cuda()

        if img_size is not None:
            masks = resize(masks.transpose(1, 2, 0), img_size, order=0, preserve_range=True).transpose(2, 0, 1)

        filter_classes = torch.tensor(filter_classes).cuda()
        filter_scores = torch.tensor(filter_scores).cuda()

        class_sem = self.text_features[filter_classes]  # [num_boxes, 768]
        class_sem = class_sem * filter_scores.unsqueeze(-1)
        sem_map = torch.einsum("nc,nhw->chw", class_sem, torch.from_numpy(masks).float().cuda())
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

        bs = (len(labels) + 99) // 100

        with torch.no_grad():
            text_features = []
            for i in range(bs):
                input_text = text[i * 100 : (i + 1) * 100].cuda()
                feature = self.text_model.encode_text(input_text)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                text_features.append(feature)
            text_features = torch.cat(text_features, dim=0)

        return text_features
