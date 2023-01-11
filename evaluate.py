import logging
import warnings
import itertools
from pathlib import Path

import ray
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from object_detection.utils import object_detection_evaluation
from object_detection.metrics import oid_challenge_evaluation_utils as oid_utils

import utils
import models
import checkpoint
import oid_mask_encoding


class Evaluator:
    def __init__(self, eval_cfg):
        warnings.filterwarnings('ignore', category=UserWarning)
        self.eval_cfg = eval_cfg

        output_path = Path(eval_cfg.output_dir)

        # Manage configs
        train_path = output_path.parent / 'train'

        # Build end-to-end network.
        if train_path.exists():
            _cfg = OmegaConf.load(train_path / '.hydra/config.yaml')
            _cfg.setting.vision_network = self.eval_cfg.setting.vision_network
            self.end2end_network = models.EndToEndNetwork(_cfg)
            if hasattr(self.eval_cfg.setting, 'filtering_network'):
                net = self.end2end_network.filtering_network
                ckpt = checkpoint.Checkpoint(output_path.parent / 'train' / 'filtering_network')
                ckpt.load(net, self.eval_cfg.setting.filtering_network.step)
            if hasattr(self.eval_cfg.setting, 'rate_estimator'):
                net = self.end2end_network.rate_estimator
                ckpt = checkpoint.Checkpoint(output_path.parent / 'train' / 'rate_estimator')
                ckpt.load(net, self.eval_cfg.setting.rate_estimator.step)
        else:
            self.end2end_network = models.EndToEndNetwork(eval_cfg)

        # Set evaluation mode & load on GPU.
        self.end2end_network.eval()
        self.end2end_network.cuda()

        self._is_vision = hasattr(eval_cfg.setting, 'vision_network')

        # Read coco classes file.
        if self._is_vision:
            self.coco_classes = open(eval_cfg.data_path.coco_class, 'r').read().splitlines()
            self._is_segmentation = (eval_cfg.setting.vision_network.task == 'segmentation')
    
    def step(self, input_file, codec, quality, downscale, control_input=None):
        image = cv2.imread(str(input_file))

        out = self.end2end_network(
            image,
            control_input=control_input,
            eval_codec=codec,
            eval_quality=quality,
            eval_downscale=downscale)
        
        bpps = {k: v for k, v in out.items() if 'bpp' in k}

        if not self._is_vision:
            return bpps

        imageId = input_file.stem
        classes = out['instances'].pred_classes.to('cpu').numpy()
        scores = out['instances'].scores.to('cpu').numpy()
        bboxes = out['instances'].pred_boxes.tensor.to('cpu').numpy()
        H, W = out['instances'].image_size

        # Convert bboxes to 0-1
        bboxes = bboxes / [W, H, W, H]

        # OpenImage output x1, x2, y1, y2 in percentage
        bboxes = bboxes[:, [0, 2, 1, 3]]

        if self._is_segmentation:
            masks = out['instances'].pred_masks.to('cpu').numpy()

        od_outs = []
        for i, coco_cnt_id in enumerate(classes):
            class_name = self.coco_classes[coco_cnt_id]
            od_out = [imageId, class_name, scores[i]] + bboxes[i].tolist()
            if self.eval_cfg.setting.vision_network.task == 'segmentation':
                assert all(map(lambda a, b: a == b, masks[i].shape[:2], [H, W])), \
                    f"Size of resulting mask does not match the input size: {imageId}"
                od_out += [
                    masks[i].shape[1],
                    masks[i].shape[0],
                    oid_mask_encoding.encode_binary_mask(masks[i]).decode('ascii')]
            od_outs.append(od_out)
        return bpps, od_outs


def evaluate(eval_cfg):
    ray.init()

    is_vision = hasattr(eval_cfg.setting, 'vision_network')
    is_estimator = hasattr(eval_cfg.setting, 'rate_estimator')
    if is_vision:
        is_segmentation = (eval_cfg.setting.vision_network.task == 'segmentation')

    # Required.
    eval_cfg.output_dir = eval_cfg.output_dir

    output_path = Path(eval_cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate evaluation settings.
    eval_downscales = eval_cfg.setting.ds_levels
    eval_qualities = eval_cfg.setting.codec.qualities

    result_path = output_path / 'result.csv'

    eval_settings = list(itertools.product(eval_downscales, eval_qualities))
    # Create or load result dataframe.
    if result_path.exists():
        result_df = pd.read_csv(result_path)
        subset_df = result_df.copy()
        # Delete already evaluated settings.
        subset_df = subset_df[subset_df.task == eval_cfg.setting.vision_network.task]
        subset_df = subset_df[subset_df.model == eval_cfg.setting.vision_network.model]
        subset_df = subset_df[subset_df.codec == eval_cfg.setting.codec.name]
        subset_df = subset_df[subset_df.step == eval_cfg.setting.step]
        evaluated_settings = itertools.product(subset_df.downscale, subset_df.quality)
        for _setting in evaluated_settings:
            if _setting in eval_settings:
                eval_settings.remove(_setting)
    else:
        result_df = pd.DataFrame(
            columns=[
                'task', 'model', 'downscale', 'codec', 'quality',
                'filtering_step', 'estimator_step',
                'rate[bpp]', 'rate_pred[bpp]', 'score[mAP]'])

    # Check input image files.
    input_files = utils.get_input_files(eval_cfg.data_path.eval_img_list, eval_cfg.data_path.eval_img_dir)
    logging.info(f"Number of total images: {len(input_files)}")

    # Create evaluators.
    n_gpu = len(eval_cfg.cuda.indices)
    n_eval = eval_cfg.num_parallel_eval_per_gpu * n_gpu
    eval_builder = ray.remote(num_gpus=(1 / eval_cfg.num_parallel_eval_per_gpu))(Evaluator)

    #TODO. Memory growth issue (~ 174GB).
    logging.info("Start evaluation loop.")
    total = len(input_files) * len(eval_settings)
    with tqdm(total=total, dynamic_ncols=True, smoothing=0.1) as pbar:
        for ds, q in eval_settings:

            if eval_cfg.setting.control_input == 'quality':
                min_q = min(eval_cfg.setting.codec.qualities)
                max_q = max(eval_cfg.setting.codec.qualities)
                control_input = 1.0 * (q - min_q) / (max_q - min_q)
            else:
                control_input = eval_cfg.setting.control_input
                assert control_input > 0. and control_input < 1.

            # Make/set evaluators and their inputs.
            evaluators = [eval_builder.remote(eval_cfg) for _ in range(n_eval)]
            input_iter = iter(input_files)

            eval_step_kwargs = {
                'codec': eval_cfg.setting.codec.name,
                'quality': q,
                'downscale': ds,
                'control_input': control_input,
            }

            # Run evaluators.
            bpps_list, od_outs_list = [], []

            work_info = dict()
            while True:
                # Put inputs.
                try:
                    if evaluators:
                        file = next(input_iter)
                        eval = evaluators.pop()
                        work_id = eval.step.remote(file, **eval_step_kwargs)
                        work_info.update({work_id: eval})
                        end_flag = False
                except StopIteration:
                    end_flag = True
                
                # Get detection result & bpp.
                if (not evaluators) or end_flag:
                    done_ids, _ = ray.wait(list(work_info.keys()), timeout=1)
                    if done_ids:
                        for done_id in done_ids:
                            if hasattr(eval_cfg.setting, 'vision_network'):
                                bpps, od_outs = ray.get(done_id)
                                od_outs_list.extend(od_outs)
                            else:
                                bpps = ray.get(done_id)
                            bpps_list.append(bpps)

                            eval = work_info.pop(done_id)
                            if end_flag:
                                ray.kill(eval)
                                del eval
                            else:
                                evaluators.append(eval)
                            pbar.update(1)
                # End loop for one setting.
                if not work_info:
                    break

            bpps_df = pd.DataFrame(bpps_list).to_dict(orient='list')

            if is_vision:
                # Read input label map.
                class_label_map, categories = utils.read_label_map(eval_cfg.data_path.coco_labelmap)
                selected_classes = list(class_label_map.keys())

                # Read annotation files.
                all_location_annotations = pd.read_csv(eval_cfg.data_path.eval_annot_boxes)
                all_label_annotations = pd.read_csv(eval_cfg.data_path.eval_annot_labels)
                all_label_annotations.rename(columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)

                # Postprocess: Convert coco to oid.
                columns = ['ImageID', 'LabelName', 'Score', 'XMin', 'XMax', 'YMin', 'YMax']
                if is_segmentation:
                    columns += ['ImageWidth', 'ImageHeight', 'Mask']
                od_output_df = pd.DataFrame(od_outs_list, columns=columns)

                # Fix & filter the image label.
                od_output_df['LabelName'] = od_output_df['LabelName'].replace(' ', '_', regex=True)
                od_output_df = od_output_df[od_output_df['LabelName'].isin(selected_classes)]

                # Resize GT segmentation labels.
                if is_segmentation:
                    all_segm_annotations = pd.read_csv(eval_cfg.data_path.eval_annot_masks)
                    for idx, row in all_segm_annotations.iterrows():
                        pred_rslt = od_output_df.loc[od_output_df['ImageID'] == row['ImageID']]
                        if not len(pred_rslt):
                            logging.info(f"Image not in prediction: {row['ImageID']}")
                            continue

                        W, H = pred_rslt['ImageWidth'].iloc[0], pred_rslt['ImageHeight'].iloc[0]

                        mask_img = Image.open(Path(eval_cfg.data_path.gt_segm_mask_dir) / row['MaskPath'])

                        if any(map(lambda a, b: a != b, mask_img.size, [W, H])):
                            mask_img = mask_img.resize((W, H))
                            mask = np.asarray(mask_img)
                            mask_str = oid_mask_encoding.encode_binary_mask(mask).decode('ascii')
                            all_segm_annotations.at[idx, 'Mask'] = mask_str
                            all_segm_annotations.at[idx, 'ImageWidth'] = W
                            all_segm_annotations.at[idx, 'ImageHeight'] = H

                    all_location_annotations = oid_utils.merge_boxes_and_masks(
                        all_location_annotations, all_segm_annotations)
                
                all_annotations = pd.concat([all_location_annotations, all_label_annotations])

                # Open images challenge evaluation.
                # Generate open image challenge evaluator.
                challenge_evaluator = (
                    object_detection_evaluation.OpenImagesChallengeEvaluator(
                        categories, evaluate_masks=is_segmentation))
                        
                # Ready for evaluation.
                with tqdm(all_annotations.groupby('ImageID')) as tbar:
                    for image_id, image_groundtruth in tbar:
                        groundtruth_dictionary = oid_utils.build_groundtruth_dictionary(image_groundtruth, class_label_map)
                        challenge_evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dictionary)
                        prediction_dictionary = oid_utils.build_predictions_dictionary(
                            od_output_df.loc[od_output_df['ImageID'] == image_id], class_label_map)
                        challenge_evaluator.add_single_detected_image_info(image_id, prediction_dictionary)

                # Evaluate. class-wise evaluation result is produced.
                metrics = challenge_evaluator.evaluate()
                
            result = {
                'task'           : eval_cfg.setting.vision_network.task if is_vision else None,
                'model'          : eval_cfg.setting.vision_network.model if is_vision else None,
                'downscale'      : ds,
                'codec'          : eval_cfg.setting.codec.name,
                'quality'        : q,
                'filtering_step' : eval_cfg.setting.filtering_network.step if is_vision else None,
                'estimator_step' : eval_cfg.setting.rate_estimator.step if is_estimator else None,
                'rate[bpp]'      : sum(bpps_df['bpp']) / len(bpps_df['bpp']),
                'rate_pred[bpp]' : sum(bpps_df['bpp_pred']) / len(bpps_df['bpp_pred']) if is_estimator else None,
                'score[mAP]'     : list(metrics.values())[0] if is_vision else None,
            }
            result_df = pd.concat([result_df, pd.DataFrame([result])], ignore_index=True)
            result_df.sort_values(
                by=['task', 'model', 'downscale', 'codec', 'quality', 'filtering_step', 'estimator_step'],
                inplace=True)
            result_df.to_csv(result_path, index=False, float_format='%.8f')
        



