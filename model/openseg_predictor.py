import torch
import clip
from tensorflow import io
import tensorflow as tf2
import tensorflow.compat.v1 as tf


def read_bytes(path):
    """Read bytes for OpenSeg model running."""

    with io.gfile.GFile(path, "rb") as f:
        file_bytes = f.read()
    return file_bytes


class OpenSeg:
    embedding_dim = 768
    
    def __init__(self, weight_path, text_model_name, set_memory_growth=True):
        # set memory growth to avoid out of memory
        if weight_path is not None:
            print("Load Tensorflow OpenSeg model...")
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf2.config.experimental.set_memory_growth(gpu, set_memory_growth)
            self.model = tf2.saved_model.load(
                weight_path,
                tags=[tf.saved_model.tag_constants.SERVING],
            )
            self.text_emb = tf.zeros([1, 1, 768])

        if text_model_name is not None:
            print("Loading CLIP {} model...".format(text_model_name))
            self.text_model, _ = clip.load(text_model_name, device="cuda", jit=False)

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
        """Extract per-pixel OpenSeg features.
        Only receives image path as input.
        """

        # load RGB image
        np_image_string = read_bytes(img_dir)
        # run OpenSeg
        results = self.model.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(np_image_string), inp_text_emb=self.text_emb
        )
        img_info = results["image_info"]
        crop_sz = [
            int(img_info[0, 0] * img_info[2, 0]),
            int(img_info[0, 1] * img_info[2, 1]),
        ]
        if regional_pool:
            image_embedding_feat = results["ppixel_ave_feat"][:, : crop_sz[0], : crop_sz[1]]
        else:
            image_embedding_feat = results["image_embedding_feat"][:, : crop_sz[0], : crop_sz[1]]
        if img_size is not None:
            feat_2d = tf.cast(
                tf.image.resize_nearest_neighbor(image_embedding_feat, img_size, align_corners=True)[0],
                dtype=tf.float16,
            ).numpy()
        else:
            feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()

        feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)

        return feat_2d

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
