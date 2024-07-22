import torch
import clip
import cv2
from torch.nn import functional as F
from model.lseg.modules.lseg_module import LSegModule
from model.lseg.additional_utils.models import LSeg_MultiEvalModule
from torchvision import transforms


class LSeg:
    embedding_dim = 512

    def __init__(self, weight_path=None):
        # set memory growth to avoid out of memory
        if weight_path is not None:
            module = LSegModule.load_from_checkpoint(
                checkpoint_path=weight_path,
                backbone="clip_vitl16_384",
                data_path=None,
                num_features=256,
                batch_size=1,
                base_lr=1e-3,
                max_epochs=100,
                augment=False,
                aux=True,
                aux_weight=0,
                ignore_index=255,
                dataset="ade20k",
                se_loss=False,
                se_weight=0,
                arch_option=0,
                block_depth=0,
                activation="lrelu",
            )
            self.transform = transforms.Compose(module.val_transform.transforms)
            net = module.net.cuda()
            scales = [1.0]
            self.evaluator = LSeg_MultiEvalModule(module, scales=scales, flip=False).cuda().eval()
            self.text_model = module.net.clip_pretrained.to(torch.float32).cuda()
        else:
            self.text_model, _ = clip.load("ViT-B/32", device='cuda', jit=False)
            self.text_model = self.text_model.to(torch.float32).cuda()

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

    def extract_image_feature(self, img_dir, img_size=None, regional_pool=True):
        """Extract per-pixel LSeg features.
        Only receives image path as input.
        """

        # load RGB image
        image = cv2.imread(str(img_dir))
        image = cv2.resize(image, (img_size[1], img_size[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # run LSeg
        x = self.transform(image).cuda()
        feat_2d = self.evaluator.compute_features(x.unsqueeze(0))

        feat_2d = feat_2d[0].cpu()  # [512, h, w]

        return feat_2d

    def extract_text_feature(self, labelset):
        # "ViT-B/32" # the model that LSeg uses
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
