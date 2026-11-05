import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from geotransformer.modules.geotransformer.adaptive_fusion import AdaptiveFeatureFusion

from backbone import KPConvFPN


class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        # Multi-scale feature fusion parameters
        self.use_multiscale = cfg.fine_matching.use_multiscale
        self.weight_fine = cfg.fine_matching.weight_fine
        self.weight_mid = cfg.fine_matching.weight_mid

        # Adaptive fusion module
        self.use_adaptive_fusion = getattr(cfg.fine_matching, 'use_adaptive_fusion', False)
        if self.use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveFeatureFusion(
                alpha_min=0.4,
                alpha_max=0.9
            )

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
            use_residual_refinement=cfg.fine_matching.use_residual_refinement,
            num_iterative_steps=cfg.fine_matching.num_iterative_steps,
            residual_sigma=cfg.fine_matching.residual_sigma,
            geo_lambda=cfg.fine_matching.geo_lambda,
            rerun_ot=cfg.fine_matching.rerun_ot,
            geo_term_min=getattr(cfg.fine_matching, 'geo_term_min', -0.3),
            geo_term_max=getattr(cfg.fine_matching, 'geo_term_max', 0.0),
            confidence_gate=getattr(cfg.fine_matching, 'confidence_gate', True),
            topk_refine=getattr(cfg.fine_matching, 'topk_refine', True),
            refine_topk=getattr(cfg.fine_matching, 'refine_topk', 5),
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        # Projection layer for middle-level features (if using multi-scale)
        if self.use_multiscale:
            # Middle level has init_dim * 8 = 512 dimensions, project to output_dim = 256
            self.proj_mid = nn.Linear(cfg.backbone.init_dim * 8, cfg.backbone.output_dim)

    def interpolate_mid_to_fine(self, fine_points, mid_points, mid_feats, mid_masks=None, k=3):
        """
        Interpolate middle-level features to fine-level points using KNN.

        Args:
            fine_points: (B, Kf, 3) fine-level points
            mid_points:  (B, Km, 3) mid-level points
            mid_feats:   (B, Km, C) mid-level features
            mid_masks:   (B, Km) boolean mask for valid mid points (optional)
            k: number of nearest neighbors

        Returns:
            interpolated_feats: (B, Kf, C) mid features interpolated to fine points
        """
        # Compute pairwise distances
        dists = torch.cdist(fine_points, mid_points)  # (B, Kf, Km)

        # Mask out invalid mid points by setting their distances to a large value
        if mid_masks is not None:
            large_value = 1e10
            dists = dists.masked_fill(~mid_masks.unsqueeze(1), large_value)  # (B, Kf, Km)

        # Find k nearest neighbors
        k = min(k, mid_points.shape[1])
        knn_dists, knn_idx = torch.topk(dists, k=k, dim=-1, largest=False)  # (B, Kf, k)

        # Gather nearest neighbor features
        # knn_idx: (B, Kf, k), mid_feats: (B, Km, C)
        # Need to gather features for each of the k neighbors
        B, Kf, k_nn = knn_idx.shape
        _, Km, C = mid_feats.shape

        # Expand knn_idx to match feature dimension: (B, Kf, k) -> (B, Kf, k, C)
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(B, Kf, k_nn, C)

        # Expand mid_feats for gathering: (B, Km, C) -> (B, 1, Km, C) -> (B, Kf, Km, C)
        mid_feats_expanded = mid_feats.unsqueeze(1).expand(B, Kf, Km, C)

        # Gather features using knn_idx
        knn_feats = torch.gather(mid_feats_expanded, 2, knn_idx_expanded)  # (B, Kf, k, C)

        # Inverse distance weighting
        weights = 1.0 / (knn_dists + 1e-8)  # (B, Kf, k)

        # Mask out invalid neighbors (those with large distances)
        if mid_masks is not None:
            valid_mask = knn_dists < 1e9  # (B, Kf, k)
            weights = weights * valid_mask.float()

        # Normalize weights (only over valid neighbors)
        weight_sum = weights.sum(dim=-1, keepdim=True)  # (B, Kf, 1)
        valid_interp = weight_sum > 1e-6  # (B, Kf, 1) - check if any valid neighbors exist

        weights_norm = torch.where(
            valid_interp,
            weights / (weight_sum + 1e-8),
            torch.zeros_like(weights)
        )

        # Weighted sum
        interpolated_feats = (knn_feats * weights_norm.unsqueeze(-1)).sum(dim=2)  # (B, Kf, C)

        return interpolated_feats, valid_interp.squeeze(-1)  # Return (B, Kf, C) and (B, Kf)

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]
        feats_m = feats_list[1]  # Middle level features

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # Extract middle level features
        ref_length_m = data_dict['lengths'][2][0].item()
        ref_feats_m = feats_m[:ref_length_m]
        src_feats_m = feats_m[ref_length_m:]

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport with multi-scale feature fusion
        # Multi-scale feature fusion at FEATURE level (not score level)
        if self.use_multiscale:
            # Get middle level points
            points_m = data_dict['points'][2].detach()
            ref_points_m = points_m[:ref_length_m]
            src_points_m = points_m[ref_length_m:]

            # Build knn indices for middle level
            _, ref_node_masks_m, ref_node_knn_indices_m, ref_node_knn_masks_m = point_to_node_partition(
                ref_points_m, ref_points_c, self.num_points_in_patch
            )
            _, src_node_masks_m, src_node_knn_indices_m, src_node_knn_masks_m = point_to_node_partition(
                src_points_m, src_points_c, self.num_points_in_patch
            )

            # Extract middle level features for selected patches
            ref_node_corr_knn_indices_m = ref_node_knn_indices_m[ref_node_corr_indices]
            src_node_corr_knn_indices_m = src_node_knn_indices_m[src_node_corr_indices]
            ref_node_corr_knn_masks_m = ref_node_knn_masks_m[ref_node_corr_indices]
            src_node_corr_knn_masks_m = src_node_knn_masks_m[src_node_corr_indices]

            ref_padded_points_m = torch.cat([ref_points_m, torch.zeros_like(ref_points_m[:1])], dim=0)
            src_padded_points_m = torch.cat([src_points_m, torch.zeros_like(src_points_m[:1])], dim=0)
            ref_node_corr_knn_points_m = index_select(ref_padded_points_m, ref_node_corr_knn_indices_m, dim=0)
            src_node_corr_knn_points_m = index_select(src_padded_points_m, src_node_corr_knn_indices_m, dim=0)

            ref_padded_feats_m = torch.cat([ref_feats_m, torch.zeros_like(ref_feats_m[:1])], dim=0)
            src_padded_feats_m = torch.cat([src_feats_m, torch.zeros_like(src_feats_m[:1])], dim=0)
            ref_node_corr_knn_feats_m = index_select(ref_padded_feats_m, ref_node_corr_knn_indices_m, dim=0)
            src_node_corr_knn_feats_m = index_select(src_padded_feats_m, src_node_corr_knn_indices_m, dim=0)

            # Project middle features to same dimension as fine features
            ref_node_corr_knn_feats_m = self.proj_mid(ref_node_corr_knn_feats_m)
            src_node_corr_knn_feats_m = self.proj_mid(src_node_corr_knn_feats_m)

            # Interpolate mid-level features to fine-level points with masks
            mid_feats_on_ref_f, ref_valid_interp = self.interpolate_mid_to_fine(
                ref_node_corr_knn_points,
                ref_node_corr_knn_points_m,
                ref_node_corr_knn_feats_m,
                mid_masks=ref_node_corr_knn_masks_m,
                k=3
            )
            mid_feats_on_src_f, src_valid_interp = self.interpolate_mid_to_fine(
                src_node_corr_knn_points,
                src_node_corr_knn_points_m,
                src_node_corr_knn_feats_m,
                mid_masks=src_node_corr_knn_masks_m,
                k=3
            )

            # Fallback: if no valid mid neighbors, use fine features instead
            ref_valid_interp_expanded = ref_valid_interp.unsqueeze(-1)  # (P, K, 1)
            src_valid_interp_expanded = src_valid_interp.unsqueeze(-1)  # (P, K, 1)
            mid_feats_on_ref_f = torch.where(
                ref_valid_interp_expanded,
                mid_feats_on_ref_f,
                ref_node_corr_knn_feats
            )
            mid_feats_on_src_f = torch.where(
                src_valid_interp_expanded,
                mid_feats_on_src_f,
                src_node_corr_knn_feats
            )

            # Fuse features at fine-level points
            if self.use_adaptive_fusion:
                # Use adaptive fusion to compute fusion weights
                # adaptive_fusion expects (B, K, C), we have (P, K, C) where P is num patches
                # Process directly as P is treated as batch dimension
                ref_fused_feats, src_fused_feats, alpha = self.adaptive_fusion(
                    ref_node_corr_knn_points,  # (P, K, 3)
                    src_node_corr_knn_points,  # (P, K, 3)
                    ref_node_corr_knn_feats,   # (P, K, C)
                    src_node_corr_knn_feats,   # (P, K, C)
                    mid_feats_on_ref_f,        # (P, K, C)
                    mid_feats_on_src_f,        # (P, K, C)
                    ref_node_corr_knn_masks,   # (P, K)
                    src_node_corr_knn_masks,   # (P, K)
                )
            else:
                # Use static weights for feature fusion
                ref_fused_feats = self.weight_fine * ref_node_corr_knn_feats + self.weight_mid * mid_feats_on_ref_f
                src_fused_feats = self.weight_fine * src_node_corr_knn_feats + self.weight_mid * mid_feats_on_src_f

            # Compute matching scores from fused features
            matching_scores = torch.einsum('bnd,bmd->bnm', ref_fused_feats, src_fused_feats)
            matching_scores = matching_scores / ref_fused_feats.shape[2] ** 0.5
        else:
            # Single-scale: only use fine-level features
            matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)
            matching_scores = matching_scores / feats_f.shape[1] ** 0.5

        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def create_model(cfg):
    model = GeoTransformer(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
