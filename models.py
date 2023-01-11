import ray
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.data import transforms as T
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

import compressai.zoo as ca_zoo

import utils
import codec_ops


class EndToEndNetwork(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.is_filtering = hasattr(self.cfg.setting, 'filtering_network')
        self.is_vision = hasattr(self.cfg.setting, 'vision_network')
        self.is_estimator = hasattr(self.cfg.setting, 'rate_estimator')

        # Codec
        if self.cfg.setting.codec.name == 'surrogate':
            self.codec = ca_zoo.mbt2018(self.cfg.setting.codec.quality, pretrained=True)
        else:
            self.codec = StandardCodec(codec=self.cfg.setting.codec.name)

        # Networks
        if self.is_filtering:
            filtering_cfg = self.cfg.setting.filtering_network
            self.feature_modulation = filtering_cfg.architecture.feature_modulation
            self.filtering_network = FilteringNetwork(
                filtering_cfg.architecture.feature_modulation,
                filtering_cfg.architecture.normalization)

        if self.is_vision:
            vision_cfg = self.cfg.setting.vision_network
            od_cfg = utils.get_od_cfg(vision_cfg.task, vision_cfg.model)
            self.vision_network = VisionNetwork(od_cfg)
            self.inference_aug = T.ResizeShortestEdge(
                [od_cfg.INPUT.MIN_SIZE_TEST, od_cfg.INPUT.MIN_SIZE_TEST], od_cfg.INPUT.MAX_SIZE_TEST)

        if self.is_estimator:
            estimator_cfg = self.cfg.setting.rate_estimator
            self.rate_estimator = BitrateEstimator(
                estimator_cfg.architecture.feature_modulation,
                estimator_cfg.architecture.normalization)
        

    def forward(self, inputs, control_input=None, eval_codec=None, eval_quality=None, eval_downscale=None):
        """ Forward method.
            x --|filtering|--> x_dot --|codec|--> x_hat --|vision_network|--> results(score, bitrate)

        Args:
            inputs (_type_): a batched tensor (training) or a single numpy image (inference).
            control_input (_type_, optional): an additional input for feature modulation. Defaults to None.
            eval_codec (str, optional): evaluation codec type. Defaults to None.
            eval_quality (int, optional): evaluation codec quality parameter. Defaults to None.
            eval_downscale (int, optional): evaluation downscaling level. Defaults to None.
            eval_filtering (bool, optional): whether to apply filtering or not. Defaults to False.

        Returns:
            dict: a dictionary containing losses and outputs
        """

        if not self.training:
            return self.inference(
                inputs, eval_codec, eval_quality, eval_downscale, control_input=control_input)


        codec_cfg = self.cfg.setting.codec
        
        if self.cfg.setting.control_input:
            fm_layer_input = control_input * 2.0 - 1.0
            fm_layer_input = torch.as_tensor(fm_layer_input, dtype=torch.float32, device=self.device)
            fm_layer_input = fm_layer_input.reshape(len(fm_layer_input), 1)

            # Lambda
            if hasattr(self.cfg.setting, 'lmbda'):
                if self.cfg.setting.lmbda.mode == 'parametrization':
                    log2_lmbda_range = self.cfg.setting.lmbda.max_log2_lmbda - self.cfg.setting.lmbda.min_log2_lmbda
                    log2_lmbdas = control_input * log2_lmbda_range + self.cfg.setting.lmbda.min_log2_lmbda
                    lmbdas = 2 ** log2_lmbdas
                elif self.cfg.setting.lmbda.mode == 'single':
                    lmbdas = 2 ** self.cfg.setting.lmbda.log2_lmbda
                else:
                    raise NotImplementedError("Only 'parametrization' and 'single' are supported.")
            
            # (optional) Quantization parameter
            if codec_cfg.name != 'surrogate':
                qp_range = codec_cfg.max_qp - codec_cfg.min_qp
                qps = control_input * qp_range + codec_cfg.min_qp
                qps = qps.round().astype('int32')
        else:
            fm_layer_input = None
            # Lambda
            assert self.cfg.setting.lmbda.mode == 'single'
            lmbdas = 2 ** self.cfg.setting.lmbda.log2_lmbda        

        outs = dict()

        # Convert input format to RGB & batch images after applying padding.
        images = self.preprocess_train_images_for_od(inputs)

        x = images.tensor / 255.

        # 1. (optional) Filtering
        if self.is_filtering:
            x_padded, (h, w) = self.filtering_network.preprocess(x)
            if self.cfg.setting.filtering_network.architecture.feature_modulation:
                x_dot_padded = self.filtering_network(x_padded, fm_layer_input)
            else:
                x_dot_padded = self.filtering_network(x_padded)
            x_dot = self.filtering_network.postprocess(x_dot_padded, (h, w))
        else:
            x_dot = x

        # 2. Codec
        if codec_cfg.name == 'surrogate':
            surrogate_codec_out = self.codec(x_dot_padded)
            x_hat_padded = surrogate_codec_out['x_hat']
            x_hat = self.filtering_network.postprocess(x_hat_padded, (h, w))
            bpp_pred = self.compute_bpp_from_likelihoods(surrogate_codec_out)
        else:
            x_hat, bpp = self.codec(x_dot, qps)
            if self.cfg.setting.rate_estimator.architecture.feature_modulation:
                bpp_pred = self.rate_estimator(x_hat, fm_layer_input)
            else:
                bpp_pred = self.rate_estimator(x_hat)
            loss_aux = torch.mean((bpp - bpp_pred) ** 2)
            outs['bpp'] = torch.mean(bpp)
            outs['loss_aux'] = loss_aux
        outs['bpp_pred'] = torch.mean(bpp_pred)

        if hasattr(self.cfg.setting, 'lmbda'):
            loss_r = bpp_pred
            lmbdas = torch.as_tensor(lmbdas, dtype=torch.float32, device=loss_r.device)
            loss_r = torch.mean(lmbdas * loss_r)
            outs['loss_r'] = loss_r

        # 3. (optional) Vision task
        if self.is_vision:
            # Re-order color channel from RGB to BGR & denormalize.
            x_hat_BGR = x_hat[:, [2, 1, 0], :, :] * 255.
            images.tensor = x_hat_BGR

            if 'instances' in inputs[0]:
                gt_instances = [x['instances'].to(self.device) for x in inputs]
            else:
                gt_instances = None            
            if 'proposals' in inputs[0]:
                proposals = [x['proposals'].to(self.device) for x in inputs]
            else:
                proposals = None

            # Internally normalize input & compute losses for object detection.
            losses_d = self.vision_network(images, gt_instances, proposals)
            outs['loss_d'] = sum(losses_d.values())
        return outs


    def inference(self, x, codec, quality, downscale, control_input=None):
        """ Proceed inference on a sinle image (not batched!).

        Args:
            x (np.ndarray): a single input image.
            codec (str): a type of codec.
            quality (int): a quantization parameter setting for codec.
            downscale (int): a resolution of image processed by codec.
            filtering (bool): whether to apply filtering network during inference. Defaults to False.
            fm_layer_input (_type_, optional): an additional input for feature modulation. Defaults to None.

        Returns:
            dict: a dictionary containing outputs.
        """        

        assert not self.training
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 3

        outs = dict()

        with torch.no_grad():
            # Re-order a color channel from 'BGR' to 'RGB'.
            x = x[:, :, ::-1]
            
            # Convert dtype to 'float32' & normalize to [0, 1].
            x = x.astype('float32') / 255.

            # Change image format from (H, W, C) to (C, H, W).
            # which is a canonical input format of Pytorch framework.
            x = x.transpose(2, 0, 1)

            # Convert to torch tensor.
            x = torch.as_tensor(x, device=self.device)

            if control_input:
                fm_layer_input = control_input * 2.0 - 1.0
                fm_layer_input = torch.as_tensor(fm_layer_input, dtype=torch.float32, device=self.device)
                fm_layer_input = fm_layer_input.reshape((1, 1))

            # 1. (optional) Filtering
            if self.is_filtering:
                x_padded, (h, w) = self.filtering_network.preprocess(x)
                if self.cfg.setting.filtering_network.architecture.feature_modulation:
                    x_dot_padded = self.filtering_network(x_padded[None, ...], fm_layer_input)[0]
                else:
                    x_dot_padded = self.filtering_network(x_padded[None, ...])[0]
                x_dot = self.filtering_network.postprocess(x_dot_padded, (h, w))
            else:
                x_dot = x
            # Convert torch tensor to numpy array.
            x_dot_numpy = x_dot.detach().cpu().numpy()

            # 2. Codec
            if codec == 'none':
                # (a). Without codec.
                x_hat_numpy, bpp = x_dot_numpy, 24.
            elif codec == 'surrogate':
                # (b). Surrogate codec.
                x_dot = torch.as_tensor(x_dot_numpy, device=self.device)
                x_dot, (h, w) = self.filtering_network.preprocess(x_dot)
                surrogate_codec_out = self.surrogate_network(x_dot[None, ...])
                # Remove padding & calculate bpp.
                x_hat, bpp = (
                    self.filtering_network.postprocess(surrogate_codec_out['x_hat'][0], (h, w)),
                    self.compute_bpp_from_likelihoods(surrogate_codec_out).item())
                x_hat_numpy = x_hat.detach().cpu().numpy()
            else:
                # (c). Standard codec.
                x_hat_numpy, bpp = ray.get(codec_ops.ray_codec_fn.remote(
                    x_dot_numpy,
                    codec=codec,
                    quality=quality,
                    downscale=downscale))
                if self.is_estimator:
                    x_hat = torch.as_tensor(x_hat_numpy, dtype=torch.float32, device=self.device)
                    x_hat = x_hat[None, ...]
                    if self.cfg.setting.rate_estimator.architecture.feature_modulation:
                        bpp_pred = self.rate_estimator(x_hat, fm_layer_input)[0]
                    else:
                        bpp_pred = self.rate_estimator(x_hat)[0]
                    bpp_mse = torch.mean((bpp_pred - bpp))
                    outs['bpp_pred'] = bpp_pred.item()
                    outs['bpp_mse'] = bpp_mse.item()
                outs['bpp'] = bpp
                    

            # 3. (optional) Vision task
            if self.is_vision:
                # Change reconstructed image format to (H, W, C) & denormalize.
                od_input_image = x_hat_numpy.transpose(1, 2, 0) * 255.

                # Re-order color channel from 'RGB' to 'BGR'.
                od_input_image = od_input_image[:, :, ::-1]

                # Convert dtype to 'uint8'
                od_input_image = od_input_image.round().astype('uint8')

                # Backup original size.
                height, width = od_input_image.shape[:2]

                # Augment reconstructed image for detection network.
                od_input_image = self.inference_aug.get_transform(od_input_image).apply_image(od_input_image)

                # Convert numpy array to torch tensor & change image format to (C, H, W)
                od_input_image = torch.as_tensor(
                    od_input_image.astype('float32').transpose(2, 0, 1), device=self.device)

                # Detector takes 'BGR' format image of range [0, 255].
                vision_inputs = {'image': od_input_image, 'height': height, 'width': width}
                vision_results = self.vision_network([vision_inputs])[0]
                outs.update(vision_results)

            outs.update({
                # Image array format: (H, W, C)
                # Pixel value range: [0, 1]
                'x_dot': x_dot_numpy.transpose(1, 2, 0),
                'x_hat': x_hat_numpy.transpose(1, 2, 0),})
        return outs

    def preprocess_train_images_for_od(self, batched_inputs):
        """ Batch the images after padding. """
        images = [x['image'] for x in batched_inputs]

        # BGR -> RGB
        images = [x[[2, 1, 0], :, :] for x in images]

        images = [x.to(self.device) for x in images]

        if self.is_vision:
            images = ImageList.from_tensors(images, self.vision_network.size_divisibility)
        else:
            images = ImageList.from_tensors(images, 64)
        return images

    def compute_bpp_from_likelihoods(self, out):
        size = out['x_hat'].size()
        num_pixels = size[-2] * size[-1]
        return sum(-torch.log2(likelihoods).sum(axis=(1, 2, 3)) / num_pixels
                for likelihoods in out['likelihoods'].values())

    @property
    def device(self):
        return next(self.parameters()).device


class VisionNetwork(nn.Module):
    def __init__(self, od_cfg):
        super().__init__()
        self.od_cfg = od_cfg
        self._event_storage = EventStorage(0)
        self.model = build_object_detection_model(od_cfg)
    
    def forward(self, images, gt_instances=None, proposals=None):
        if not self.training:
            return self.inference(images)
        
        with self._event_storage:
            images.tensor = self.preprocess(images.tensor)
            features = self.model.backbone(images.tensor)

            if proposals is None:
                proposals, proposal_losses = self.model.proposal_generator(images, features, gt_instances)
            else:
                proposal_losses = {}
            
            _, detector_losses = self.model.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training

        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [self.preprocess(x) for x in images]
        images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)

        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _  = self.model.roi_heads(images, features, proposals, None)
        results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def preprocess(self, x):
        x = (x - self.model.pixel_mean) / self.model.pixel_std
        return x

    @property
    def device(self):
        return self.model.device

    @property
    def size_divisibility(self):
        return self.model.backbone.size_divisibility

    def train(self, mode=True):
        super().train(mode)
        self.model.backbone.train(False)


class FilteringNetwork(nn.Module):
    def __init__(self, feature_modulation=False, normalization='cn'):
        super().__init__()
        self.feature_modulation = feature_modulation
        self.normalization = normalization

        conv_channel_config = [
            ( 3, 16), (16, 32), (32, 64),
            (64, 32), (32, 16)
        ]
        self.filtering_blocks = nn.ModuleList([
            FilteringBlock(in_channels, out_channels, normalization)
            for in_channels, out_channels in conv_channel_config
        ])
        self.last_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, stride=1),
            nn.Tanh()
        )

        if self.feature_modulation:
            fc_channel_config = [
                ( 1, 16), ( 1, 32), ( 1, 64),
                ( 1, 32), ( 1, 16)
            ]
            self.modulators = nn.ModuleList([
                FeatureModulator(in_channels, out_channels)
                for in_channels, out_channels in fc_channel_config
            ])

    def forward(self, x, fm_in=None):
        out = x
        if self.feature_modulation:
            for m, fb in zip(self.modulators, self.filtering_blocks):
                gamma, beta = m(fm_in)
                out = fb(out, gamma=gamma, beta=beta)
        else:
            for fb in self.filtering_blocks:
                out = fb(out)
        out = self.last_conv(out)
        out = out + x
        out = torch.clip(out, 0., 1.)
        return out

    def preprocess(self, x):
        # Pad.
        def _check_for_padding(x):
            remainder = x % 64
            if remainder:
                return 64 - remainder
            return remainder
        h, w = x.shape[-2:]
        h_pad, w_pad = map(_check_for_padding, (h, w))
        x = F.pad(x, [0, w_pad, 0, h_pad])
        return x, (h, w)
        
    def postprocess(self, x, size):
        # Unpad.
        h, w = size
        x = x[..., :h, :w]

        # Clip to [0, 1].
        x = x.clip(0., 1.)
        return x

    @property
    def device(self):
        return next(self.parameters()).device
    

class FilteringBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization):
        super().__init__()
        self.conv = FMConv2dBlock(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding='same', normalization=normalization
        )
        self.se_layer = SELayer(out_channels, reduction_ratio=8)
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, gamma=None, beta=None):
        out = self.conv(x, gamma, beta)
        out = self.se_layer(out)
        if self.proj:
            x = self.proj(x)
        out = out + x
        out = self.relu(out)
        return out


class ChannelNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-3):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.parameter.Parameter(
            torch.empty((num_features,), dtype=torch.float32), requires_grad=True
        )
        self.beta = nn.parameter.Parameter(
            torch.empty((num_features,), dtype=torch.float32), requires_grad=True
        )
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def _get_moments(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = torch.sum((x - mean.detach()) ** 2, dim=1, keepdim=True)
        # Divide by N - 1
        variance /= (self.num_features - 1)
        return mean, variance

    def forward(self, x):
        mean, variance = self._get_moments(x)
        x = (x - mean) / (torch.sqrt(variance) + self.eps)
        x = x * self.gamma[None, :, None, None] + self.beta[None, :, None, None]
        return x


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.se_module = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_pooled = torch.mean(x, dim=(2, 3), keepdim=False)
        scale = self.se_module(x_pooled)[:, :, None, None]
        out = x * scale
        return out


class FeatureModulator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc_g = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Softplus()
        )
        self.fc_b = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        gamma = self.fc_g(x)
        beta  = self.fc_b(x)
        return gamma, beta


class FMConv2dBlock(nn.Module):
    """ Feature Modulated 2D Convolutional Block """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, normalization='cn'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.normalization = normalization

        assert normalization in ['bn', 'cn']

        self.conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            ),
            ChannelNorm2d(out_channels) if self.normalization == 'cn' else nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, gamma=None, beta=None):
        x = self.conv(x)
        if (gamma is not None) and (beta is not None):
            x = x * gamma[:, :, None, None] + beta[:, :, None, None]
        return x


class BitrateEstimator(nn.Module):
    """ Bitrate estimator module that predicts bit-per-pixel from the reconstructed image. """
    def __init__(self, feature_modulation=False, normalization='cn'):
        super().__init__()
        self.feature_modulation = feature_modulation
        self.normalization = normalization

        conv_channel_config = [
            (3, 32), (32, 64), (64, 128)
        ]
        self.conv_blocks = nn.ModuleList([
            FMConv2dBlock(i, o, (3, 3), 2) for i, o in conv_channel_config
        ])
        self.last_conv = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), 2),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1),
        )

        if self.feature_modulation:
            fc_channel_config = [
                ( 1, 32), (1, 64), (1, 128)
            ]
            self.modulators = nn.ModuleList([
                FeatureModulator(i, o) for i, o in fc_channel_config
            ])
    
    def forward(self, x, fm_in=None):
        out = x
        if self.feature_modulation:
            for m, cb in zip(self.modulators, self.conv_blocks):
                gamma, beta = m(fm_in)
                out = cb(out, gamma=gamma, beta=beta)
        else:
            for cb in self.conv_blocks:
                out = cb(out)
        out = self.last_conv(out)
        out = torch.mean(out, dim=(1, 2, 3)) * 24.0
        return out

    @property
    def device(self):
        return next(self.parameters()).device


class StandardCodec(nn.Module):
    def __init__(self, codec='vvenc'):
        super().__init__()
        self.connect_gradient = GradientConnector.apply
        self.codec = codec
    
    def forward(self, x, qp):
        device = x.device
        x_numpy = x.detach().cpu().numpy()
        n = len(x_numpy)
        x_hat, bpp = zip(*ray.get([codec_ops.ray_codec_fn.remote(x_numpy[i], self.codec, qp[i]) for i in range(n)]))
        x_hat = np.stack(x_hat, axis=0)
        bpp = np.stack(bpp, axis=0)
        x_hat = torch.as_tensor(x_hat, dtype=torch.float32, device=device)
        bpp = torch.as_tensor(bpp, dtype=torch.float32, device=device)
        _, x_hat = self.connect_gradient(x, x_hat)
        return x_hat, bpp


class GradientConnector(torch.autograd.Function):
    @staticmethod
    def forward(_, input1, input2):
        return input1, input2
    
    @staticmethod
    def backward(_, grad_out1, grad_out2):
        return grad_out1, grad_out1


def build_object_detection_model(cfg):
    cfg = cfg.clone()
    model = GeneralizedRCNN(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model