import logging
import warnings
import itertools
from pathlib import Path

import ray
import cv2
import pandas as pd
from tqdm import tqdm

from object_detection.utils import object_detection_evaluation
from object_detection.metrics import oid_challenge_evaluation_utils as oid_utils

import utils
import models
import checkpoint
import oid_mask_encoding


class Evaluator:
    def __init__(self, cfg):
        warnings.filterwarnings('ignore', category=UserWarning)
        self.cfg = cfg

        target_path = Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)
        if surrogate_quality:
            _, self.is_saved_session, norm_layer = utils.inspect_target_path(target_path)
        else:
            surrogate_quality, self.is_saved_session, norm_layer = utils.inspect_target_path(target_path)

        # dummy
        if surrogate_quality is None:
            surrogate_quality = 1

        # Build end-to-end network.
        self.end2end_network = models.EndToEndNetwork(cfg)

        # Restore weights.
        if self.is_saved_session:
            ckpt = checkpoint.Checkpoint(target_path)
            ckpt.load(self.end2end_network.filtering_network, step=session_step)

        # Set evaluation mode & load on GPU.
        self.end2end_network.eval()
        self.end2end_network.cuda()
    
    def step(self, input_file, codec, quality, downscale, control_input=None):
        image = cv2.imread(str(input_file))
        fm_layer_input = control_input * 2.0 - 1.0
        outs = self.end2end_network(
            image,
            eval_codec=codec,
            eval_quality=quality,
            eval_downscale=downscale,
            eval_filtering=self.is_saved_session,
            fm_layer_input=fm_layer_input)
        imageId = input_file.stem
        classes = outs['instances'].pred_classes.to('cpu').numpy()
        scores = outs['instances'].scores.to('cpu').numpy()
        bboxes = outs['instances'].pred_boxes.tensor.to('cpu').numpy()
        H, W = outs['instances'].image_size

        # Bit per pixel.
        bpp = outs['bpp']

        # convert bboxes to 0-1
        bboxes = bboxes / [W, H, W, H]

        # OpenImage output x1, x2, y1, y2 in percentage
        bboxes = bboxes[:, [0, 2, 1, 3]]

        if self.vision_task == 'segmentation':
            masks = outs['instances'].pred_masks.to('cpu').numpy()

        od_outs = []
        for i, coco_cnt_id in enumerate(classes):
            class_name = self.coco_classes[coco_cnt_id]
            od_out = [imageId, class_name, scores[i]] + bboxes[i].tolist()
            if self.vision_task == 'segmentation':
                od_out += [
                    masks[i].shape[1],
                    masks[i].shape[0],
                    oid_mask_encoding.encode_binary_mask(masks[i]).decode('ascii')]
            od_outs.append(od_out)
        return od_outs, bpp


def evaluate_for_object_detection(cfg):
    ray.init()

    target_path = Path(cfg.command.target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    result_path = target_path / 'result.csv'

    # Generate evaluation settings.
    eval_downscales = cfg.command.codec.ds_levels
    eval_qualities = cfg.command.codec.qualities
    eval_settings = list(itertools.product(eval_downscales, eval_qualities))

    # Create or load result dataframe.
    if result_path.exists():
        result_df = pd.read_csv(result_path)
        subset_df = result_df.copy()
        # Delete already evaluated settings.
        subset_df = subset_df[subset_df.step == cfg.target_step]
        subset_df = subset_df[subset_df.task == cfg.vision_task]
        subset_df = subset_df[subset_df.codec == cfg.command.codec.name]
        evaluated_settings = itertools.product(subset_df.downscale, subset_df.quality)
        for _setting in evaluated_settings:
            if _setting in eval_settings:
                eval_settings.remove(_setting)
    else:
        result_df = pd.DataFrame(
            columns=['task', 'codec', 'downscale', 'quality', 'bpp', 'metric', 'step'])

    # Check input image files.
    input_files = utils.get_input_files(cfg.data_path.eval_img_list, cfg.data_path.eval_img_dir)
    logging.info(f"Number of total images: {len(input_files)}")


    # Create evaluators.
    n_gpu = len(cfg.cuda.indices)
    n_eval = cfg.num_parallel_eval_per_gpu * n_gpu
    eval_builder = ray.remote(num_gpus=(1 / cfg.num_parallel_eval_per_gpu))(Evaluator)

    #TODO. Memory growth issue (~ 174GB).
    logging.info("Start evaluation loop.")
    total = len(input_files) * len(eval_settings)
    with tqdm(total=total, dynamic_ncols=True, smoothing=0.1) as pbar:
        for downscale, quality in eval_settings:

            # Read coco classes file.
            coco_classes = open(cfg.data_path.coco_class, 'r').read().splitlines()

            # Read input label map.
            class_label_map, categories = utils.read_label_map(cfg.data_path.coco_labelmap)
            selected_classes = list(class_label_map.keys())

            # Read annotation files.
            all_location_annotations = pd.read_csv(cfg.data_path.eval_annot_boxes)
            all_label_annotations = pd.read_csv(cfg.data_path.eval_annot_labels)
            all_label_annotations.rename(columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)
            is_instance_segmentation_eval = False
            if cfg.vision_task == 'segmentation':
                annot_gt = pd.read_csv(cfg.data_path.eval_annot_masks)
                is_instance_segmentation_eval = True

            all_annotations = pd.concat([all_location_annotations, all_label_annotations])
                
            # Make/set evaluators and their inputs.
            evaluators = [eval_builder.remote(cfg) for _ in range(n_eval)]
            input_iter = iter(input_files)
            codec_args = (cfg.command.codec.name, quality, downscale)
            # Run evaluators.
            od_outputs, bpps = [], []
            work_info = dict()
            while True:
                # Put inputs.
                try:
                    if evaluators:
                        file = next(input_iter)
                        eval = evaluators.pop()
                        work_id = eval.step.remote(file, config.control_param, *codec_args)
                        work_info.update({work_id: eval})
                        end_flag = False
                except StopIteration:
                    end_flag = True
                
                # Get detection result & bpp.
                if (not evaluators) or end_flag:
                    done_ids, _ = ray.wait(list(work_info.keys()), timeout=1)
                    if done_ids:
                        for done_id in done_ids:
                            # Store outputs.
                            od_outputs_, bpp = ray.get(done_id)
                            od_outputs.extend(od_outputs_)
                            bpps.append(bpp)
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

            # Postprocess: Convert coco to oid.
            columns = ['ImageID', 'LabelName', 'Score', 'XMin', 'XMax', 'YMin', 'YMax']
            if cfg.vision_task == 'segmentation':
                columns += ['ImageWidth', 'ImageHeight', 'Mask']
            od_output_df = pd.DataFrame(od_outputs, columns=columns)

            # Fix & filter the image label.
            od_output_df['LabelName'] = od_output_df['LabelName'].replace(' ', '_', regex=True)
            od_output_df = od_output_df[od_output_df['LabelName'].isin(selected_classes)]

            # Resize GT segmentation labels.
            if cfg.vision_task == 'segmentation':
                all_segm_annotations = pd.read_csv(input_annot_masks)
                for idx, row in anno_gt.iterrows():
                    pred_rslt = od_output_df.loc[od_output_df['ImageID'] == row['ImageID']]
                    if not len(pred_rslt):
                        logger.info(f"Image not in prediction: {row['ImageID']}")
                        continue

                    W, H = pred_rslt['ImageWidth'].iloc[0], pred_rslt['ImageHeight'].iloc[0]

                    mask_img = Image.open(gt_segm_mask_dir / row['MaskPath'])

                    if any(map(lambda a, b: a != b, mask_img.size, [W, H])):
                        mask_img = mask_img.resize((W, H))
                        mask = np.asarray

            # Open images challenge evaluation.
            if cfg.vision_task == 'detection':
                # Generate open image challenge evaluator.
                challenge_evaluator = (
                    object_detection_evaluation.OpenImagesChallengeEvaluator(
                        categories, evaluate_masks=is_instance_segmentation_eval))
                # Ready for evaluation.
                for image_id, image_groundtruth in all_annotations.groupby('ImageID'):
                    groundtruth_dictionary = oid_utils.build_groundtruth_dictionary(image_groundtruth, class_label_map)
                    challenge_evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dictionary)
                    prediction_dictionary = oid_utils.build_predictions_dictionary(
                        od_output_df.loc[od_output_df['ImageID'] == image_id], class_label_map)
                    challenge_evaluator.add_single_detected_image_info(image_id, prediction_dictionary)

                # Evaluate. class-wise evaluation result is produced.
                metrics = challenge_evaluator.evaluate()
                mean_map = list(metrics.values())[0]
                mean_bpp = sum(bpps) / len(bpps)
            else:
                pass
            result = {
                'task'     : config.vision_task,
                'codec'    : config.eval_codec,
                'downscale': downscale,
                'quality'  : quality,
                'bpp'      : mean_bpp,
                'metric'   : mean_map,
                'step'     : config.session_step,
                'control_param' : config.control_param,
            }
            result_df = pd.concat([result_df, pd.DataFrame([result])], ignore_index=True)
            result_df.sort_values(
                by=['task', 'codec', 'downscale', 'step', 'bpp'], inplace=True)
            result_df.to_csv(result_path, index=False)



