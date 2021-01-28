import numpy as np
from ofa.model_zoo import ofa_specialized
from torchvision.transforms.functional import pil_to_tensor, normalize
from ofa.imagenet_classification.networks.mobilenet_v3 import MobileNetV3

from .abc import Embedding, RequestIgnored
from ..images import getSmallImage, ImageError


class ImageEmbedding(Embedding):

    def __init__(self):
        self.__net_id = "flops@389M_top1@79.1_finetune@75"
        self.__model, self.__image_size = ofa_specialized(self.__net_id, pretrained=True)
        mobi = self.__model
        assert isinstance(mobi, MobileNetV3)

        def _forward__fv(x):
            """Patch of bound method MobileNetV3.forward to extract the feature vectors without classification layer"""
            x = mobi.first_conv(x)
            for block in mobi.blocks:
                x = block(x)
            x = mobi.final_expand_layer(x)
            x = mobi.global_avg_pool(x)  # global average pooling
            x = mobi.feature_mix_layer(x)
            x = x.view(x.size(0), -1)
            # x = mobi.classifier(x)
            return x

        self.__model.forward = _forward__fv
        self.__model.eval()
        self.__dim = self.__model.feature_mix_layer.conv.out_channels

        # to normalize according to model trained on imagenet
        self.__mean = [0.485, 0.456, 0.406]
        self.__std = [0.229, 0.224, 0.225]

    def get_dim(self) -> int:
        return self.__dim

    def get_version(self) -> str:
        return 'ofa/ofa_specialized/' + self.__net_id + '/fv'

    def transform(self, *, url: str = None, data: bytes = None) -> np.ndarray:

        assert url is not None or data is not None
        fp = data if data is not None else url

        if url is not None and url.startswith('data:,'):
            raise RequestIgnored('Not an image')

        try:
            img = getSmallImage(fp, self.__image_size, self.__image_size)
        except ImageError as ex:
            raise RequestIgnored(ex) from ex

        _input = pil_to_tensor(img.convert('RGB'))
        assert 'uint8' in str(_input.dtype)
        _input = _input / 255.
        assert 'float32' in str(_input.dtype)
        _input = normalize(_input, self.__mean, self.__std, inplace=True)
        assert len(_input.shape) == 3, _input.shape
        _input = _input.unsqueeze(0)
        assert len(_input.shape) == 4, _input.shape

        v = self.__model(_input).detach().numpy()[0]
        
        assert len(v.shape) == 1 and v.shape[0] == self.__dim, v.shape
        
        v /= np.sum(v ** 2) ** 0.5  # L2 normalization
        return v

    # TODO: reimplement transform_batch
