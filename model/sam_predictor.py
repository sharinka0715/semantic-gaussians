import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class SAMAutoMask:
    def __init__(self, sam_path):
        if sam_path is not None:
            print("Load SAM model...")
            sam = sam_model_registry["vit_h"](checkpoint=sam_path)
            sam.cuda()
            self.sam = SamPredictor(sam)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.7,
                box_nms_thresh=0.7,
                stability_score_thresh=0.85,
                # crop_n_layers=1,
                # crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )

    def extract_image_feature(self, img_dir, img_size=None):
        """Extract per-pixel OpenSeg features.
        Only receives image path as input.
        """
        image = cv2.imread(str(img_dir))
        image = cv2.resize(image, (img_size[1], img_size[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)

        segs = []
        sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        for mask in sorted_masks:
            segs.append(mask["segmentation"])

        return np.stack(segs, axis=0)
