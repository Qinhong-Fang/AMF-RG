import sys
import json
import argparse
import glob
import os.path as osp
import time
import warnings
import csv

import numpy as np
import torch

from config import make_cfg
from geotransformer.engine import Logger
from geotransformer.modules.registration import weighted_procrustes
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)

# Ignore PyTorch UserWarnings about tensor resizing
warnings.filterwarnings('ignore', category=UserWarning)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd'], required=True, help='registration method')
    parser.add_argument(
        '--num_corr',
        default='all',
        help='number of correspondences for registration; use "all" to keep all correspondences',
    )
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument(
        '--output_csv',
        default=None,
        help='path to save per-case evaluation csv; default: <results_dir>/eval_<method>_<num_corr>_cases.csv',
    )
    return parser


def parse_num_corr(num_corr):
    """Convert --num_corr into int or None. None means using all correspondences."""
    if num_corr is None:
        return None, 'all'
    if isinstance(num_corr, int):
        return num_corr, str(num_corr)

    num_corr = str(num_corr).strip().lower()
    if num_corr in ('all', 'none', '-1'):
        return None, 'all'

    value = int(num_corr)
    if value <= 0:
        raise ValueError('--num_corr must be a positive integer or "all".')
    return value, str(value)


def get_default_output_csv(features_root, method, num_corr_tag):
    # cfg.feature_dir is usually ".../output/results/features".
    # Save csv to ".../output/results/" by default.
    results_root = osp.dirname(features_root.rstrip('/'))
    return osp.join(results_root, f'eval_{method}_{num_corr_tag}_cases.csv')


def eval_one_epoch(args, cfg, logger):
    features_root = cfg.feature_dir
    num_corr_limit, num_corr_tag = parse_num_corr(args.num_corr)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = get_default_output_csv(features_root, args.method, num_corr_tag)

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter('precision')
    coarse_matching_meter.register_meter('PMR>0')
    coarse_matching_meter.register_meter('PMR>=0.1')
    coarse_matching_meter.register_meter('PMR>=0.3')
    coarse_matching_meter.register_meter('PMR>=0.5')

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter('recall')
    fine_matching_meter.register_meter('inlier_ratio')
    fine_matching_meter.register_meter('overlap')

    registration_meter = SummaryBoard()
    registration_meter.register_meter('recall')
    registration_meter.register_meter('rre')
    registration_meter.register_meter('rte')

    file_names = sorted(
        glob.glob(osp.join(features_root, '*.npz')),
        key=lambda x: [int(i) for i in osp.splitext(osp.basename(x))[0].split('_')],
    )

    num_test_pairs = len(file_names)
    if num_test_pairs == 0:
        raise RuntimeError(f'No npz files found in feature directory: {features_root}')

    case_rows = []

    for i, file_name in enumerate(file_names):
        seq_id, src_frame, ref_frame = [int(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]

        data_dict = np.load(file_name)

        ref_nodes = data_dict['ref_points_c']
        src_nodes = data_dict['src_points_c']
        ref_node_corr_indices = data_dict['ref_node_corr_indices']
        src_node_corr_indices = data_dict['src_node_corr_indices']

        ref_corr_points = data_dict['ref_corr_points']
        src_corr_points = data_dict['src_corr_points']
        corr_scores = data_dict['corr_scores']

        gt_node_corr_indices = data_dict['gt_node_corr_indices']
        gt_transform = data_dict['transform']

        if num_corr_limit is not None and corr_scores.shape[0] > num_corr_limit:
            sel_indices = np.argsort(-corr_scores)[:num_corr_limit]
            ref_corr_points = ref_corr_points[sel_indices]
            src_corr_points = src_corr_points[sel_indices]
            corr_scores = corr_scores[sel_indices]

        n_corr_used = int(corr_scores.shape[0])

        # Keep log format consistent with test.py / baseline logs:
        # id0 is ref_frame and id1 is src_frame.
        message = '{}/{}, seq_id: {}, id0: {}, id1: {}'.format(
            i + 1, num_test_pairs, seq_id, ref_frame, src_frame
        )

        # 1. evaluate correspondences
        # 1.1 evaluate coarse correspondences
        coarse_matching_result_dict = evaluate_sparse_correspondences(
            ref_nodes,
            src_nodes,
            ref_node_corr_indices,
            src_node_corr_indices,
            gt_node_corr_indices,
        )

        coarse_precision = coarse_matching_result_dict['precision']

        coarse_matching_meter.update('precision', coarse_precision)
        coarse_matching_meter.update('PMR>0', float(coarse_precision > 0))
        coarse_matching_meter.update('PMR>=0.1', float(coarse_precision >= 0.1))
        coarse_matching_meter.update('PMR>=0.3', float(coarse_precision >= 0.3))
        coarse_matching_meter.update('PMR>=0.5', float(coarse_precision >= 0.5))

        # 1.2 evaluate fine correspondences
        fine_matching_result_dict = evaluate_correspondences(
            ref_corr_points,
            src_corr_points,
            gt_transform,
            positive_radius=cfg.eval.acceptance_radius,
        )

        inlier_ratio = fine_matching_result_dict['inlier_ratio']
        overlap = fine_matching_result_dict['overlap']
        residual = fine_matching_result_dict['residual']
        num_corr = fine_matching_result_dict['num_corr']

        fine_matching_meter.update('inlier_ratio', inlier_ratio)
        fine_matching_meter.update('overlap', overlap)
        fine_matching_meter.update('recall', float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))

        message += ', c_PIR: {:.3f}'.format(coarse_precision)
        message += ', f_IR: {:.3f}'.format(inlier_ratio)
        message += ', f_OV: {:.3f}'.format(overlap)
        message += ', f_RS: {:.3f}'.format(residual)
        message += ', f_NU: {}'.format(num_corr)

        # 2. evaluate registration
        if args.method == 'lgr':
            est_transform = data_dict['estimated_transform']
        elif args.method == 'ransac':
            est_transform = registration_with_ransac_from_correspondences(
                src_corr_points,
                ref_corr_points,
                distance_threshold=cfg.ransac.distance_threshold,
                ransac_n=cfg.ransac.num_points,
                num_iterations=cfg.ransac.num_iterations,
            )
        elif args.method == 'svd':
            with torch.no_grad():
                ref_corr_points_t = torch.from_numpy(ref_corr_points).cuda()
                src_corr_points_t = torch.from_numpy(src_corr_points).cuda()
                corr_scores_t = torch.from_numpy(corr_scores).cuda()
                est_transform = weighted_procrustes(
                    src_corr_points_t, ref_corr_points_t, corr_scores_t, return_transform=True
                )
                est_transform = est_transform.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupported registration method: {args.method}.')

        rre, rte = compute_registration_error(gt_transform, est_transform)
        accepted = rre < cfg.eval.rre_threshold and rte < cfg.eval.rte_threshold
        rr = float(accepted)

        if accepted:
            registration_meter.update('rre', rre)
            registration_meter.update('rte', rte)
        registration_meter.update('recall', rr)

        message += ', r_RRE: {:.3f}'.format(rre)
        message += ', r_RTE: {:.3f}'.format(rte)
        message += ', RR: {:.3f}'.format(rr)

        case_rows.append(
            {
                'index': i + 1,
                'seq_id': seq_id,
                'ref_frame': ref_frame,
                'src_frame': src_frame,
                'method': args.method,
                'num_corr': num_corr_tag,
                'nCorr_used': n_corr_used,
                'c_PIR': float(coarse_precision),
                'f_IR': float(inlier_ratio),
                'f_OV': float(overlap),
                'f_RS': float(residual),
                'f_NU': int(num_corr),
                'RRE': float(rre),
                'RTE': float(rte),
                'RR': rr,
                'success': int(accepted),
                'npz_file': file_name,
            }
        )

        if args.verbose:
            logger.info(message)

    # Save per-case results for later case selection / visualization.
    fieldnames = [
        'index',
        'seq_id',
        'ref_frame',
        'src_frame',
        'method',
        'num_corr',
        'nCorr_used',
        'c_PIR',
        'f_IR',
        'f_OV',
        'f_RS',
        'f_NU',
        'RRE',
        'RTE',
        'RR',
        'success',
        'npz_file',
    ]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    logger.critical(f'  Per-case results saved to: {output_csv}')

    if args.test_epoch is not None:
        logger.critical(f'Epoch {args.test_epoch}')

    # 1. print correspondence evaluation results
    message = '  Coarse Matching'
    message += ', PIR: {:.3f}'.format(coarse_matching_meter.mean('precision'))
    message += ', PMR>0: {:.3f}'.format(coarse_matching_meter.mean('PMR>0'))
    message += ', PMR>=0.1: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.1'))
    message += ', PMR>=0.3: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.3'))
    message += ', PMR>=0.5: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.5'))
    logger.critical(message)

    message = '  Fine Matching'
    message += ', FMR: {:.3f}'.format(fine_matching_meter.mean('recall'))
    message += ', IR: {:.3f}'.format(fine_matching_meter.mean('inlier_ratio'))
    message += ', OV: {:.3f}'.format(fine_matching_meter.mean('overlap'))
    message += ', std: {:.3f}'.format(fine_matching_meter.std('recall'))
    logger.critical(message)

    # 2. print registration evaluation results
    message = '  Registration'
    message += ', RR: {:.3f}'.format(registration_meter.mean('recall'))
    message += ', RRE: {:.3f}'.format(registration_meter.mean('rre'))
    message += ', RTE: {:.3f}'.format(registration_meter.mean('rte'))
    logger.critical(message)


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.log_dir, 'eval-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file)

    message = 'Command executed: ' + ' '.join(sys.argv)
    logger.info(message)
    message = 'Configs:\n' + json.dumps(cfg, indent=4)
    logger.info(message)

    eval_one_epoch(args, cfg, logger)


if __name__ == '__main__':
    main()
