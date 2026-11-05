import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiScaleFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128):
        """Adaptive multi-scale feature fusion module.

        Dynamically adjusts fusion weights based on local geometric complexity.
        High complexity regions (edges, corners) use more fine-level features.
        Low complexity regions (flat surfaces) use more mid-level features.

        Args:
            feature_dim: dimension of input features
            hidden_dim: hidden dimension for weight prediction network
        """
        super(AdaptiveMultiScaleFusion, self).__init__()

        # Small MLP to predict fusion weights from geometric features
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, hidden_dim),  # concat fine_feat, mid_feat, geometric_feat
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # output in [0, 1]
        )

    def compute_geometric_complexity(self, points, k=16):
        """Compute local geometric complexity (curvature-like measure).

        Args:
            points: (P, K, 3) patch points
            k: number of neighbors for complexity estimation

        Returns:
            complexity: (P, K) complexity score for each point
        """
        P, K, _ = points.shape

        # For each point, compute variance of its k-nearest neighbors
        # High variance = high complexity (edges, corners)
        # Low variance = low complexity (flat surfaces)

        complexity_scores = []
        for i in range(P):
            patch_points = points[i]  # (K, 3)

            # Compute pairwise distances
            dists = torch.cdist(patch_points, patch_points)  # (K, K)

            # For each point, find k nearest neighbors
            knn_dists, _ = torch.topk(dists, k=min(k, K), dim=1, largest=False)  # (K, k)

            # Compute variance of distances as complexity measure
            complexity = torch.var(knn_dists, dim=1)  # (K,)
            complexity_scores.append(complexity)

        complexity_scores = torch.stack(complexity_scores, dim=0)  # (P, K)

        # Normalize to [0, 1]
        complexity_scores = (complexity_scores - complexity_scores.min()) / (complexity_scores.max() - complexity_scores.min() + 1e-8)

        return complexity_scores

    def forward(self, ref_feats_fine, src_feats_fine, ref_feats_mid, src_feats_mid,
                ref_points, src_points, matching_scores_fine, matching_scores_mid):
        """Adaptive fusion of multi-scale matching scores.

        Args:
            ref_feats_fine: (P, K, C) fine-level reference features
            src_feats_fine: (P, K, C) fine-level source features
            ref_feats_mid: (P, K, C) mid-level reference features
            src_feats_mid: (P, K, C) mid-level source features
            ref_points: (P, K, 3) reference points
            src_points: (P, K, 3) source points
            matching_scores_fine: (P, K, K) fine-level matching scores
            matching_scores_mid: (P, K, K) mid-level matching scores

        Returns:
            fused_scores: (P, K, K) adaptively fused matching scores
        """
        P, K, C = ref_feats_fine.shape

        # Compute geometric complexity for both ref and src
        ref_complexity = self.compute_geometric_complexity(ref_points)  # (P, K)
        src_complexity = self.compute_geometric_complexity(src_points)  # (P, K)

        # For each pair (i, j), compute adaptive weight
        # Expand to (P, K, K) for pairwise weights
        ref_complexity_expanded = ref_complexity.unsqueeze(2).expand(P, K, K)  # (P, K, K)
        src_complexity_expanded = src_complexity.unsqueeze(1).expand(P, K, K)  # (P, K, K)

        # Average complexity of the pair
        pair_complexity = (ref_complexity_expanded + src_complexity_expanded) / 2.0  # (P, K, K)

        # Prepare features for weight prediction
        # For each pair (i, j), concat ref_feat[i], src_feat[j], and geometric info
        ref_feats_expanded = ref_feats_fine.unsqueeze(2).expand(P, K, K, C)  # (P, K, K, C)
        src_feats_expanded = src_feats_fine.unsqueeze(1).expand(P, K, K, C)  # (P, K, K, C)
        geometric_feat = pair_complexity.unsqueeze(-1)  # (P, K, K, 1)

        # Concat features - but this creates huge tensor, so we use a simpler approach:
        # Just use geometric complexity to modulate weights

        # Simple adaptive weight: high complexity -> high weight for fine features
        # Map complexity [0, 1] to weight_fine [0.5, 0.9]
        alpha = 0.5 + 0.4 * pair_complexity  # (P, K, K), range [0.5, 0.9]

        # Fuse scores
        fused_scores = alpha * matching_scores_fine + (1 - alpha) * matching_scores_mid

        return fused_scores


class AdaptiveFeatureFusion(nn.Module):
    """Adaptive multi-scale feature fusion based on geometric complexity.

    Fuses fine-level and mid-level features at the same set of fine-level points.
    Uses learnable sigmoid mapping to convert geometric complexity to fusion weights.
    """

    def __init__(self, alpha_min=0.4, alpha_max=0.9):
        """
        Args:
            alpha_min: minimum fusion weight for fine features (default 0.4)
            alpha_max: maximum fusion weight for fine features (default 0.9)
        """
        super(AdaptiveFeatureFusion, self).__init__()

        # Learnable parameters for sigmoid mapping
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.bias = nn.Parameter(torch.tensor(0.1))

        # Alpha range
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def compute_complexity(self, points, masks, k=8):
        """Compute geometric complexity only on valid points.

        Args:
            points: (B, K, 3) patch points
            masks: (B, K) boolean masks, True for valid points
            k: number of neighbors for complexity estimation

        Returns:
            complexity: (B,) raw complexity score per patch
        """
        B, K, _ = points.shape
        complexity_list = []

        for b in range(B):
            # Only use valid points
            valid_mask = masks[b]
            valid_points = points[b][valid_mask]  # (N_valid, 3)

            # If too few valid points, use default complexity
            if valid_points.shape[0] < 4:
                complexity_list.append(points.new_tensor(0.0))
                continue

            # Compute pairwise distances among valid points
            dists = torch.cdist(valid_points, valid_points)  # (N_valid, N_valid)

            # For each valid point, find k nearest neighbors
            knn_k = min(k + 1, valid_points.shape[0])  # +1 to include self
            sorted_dists, _ = torch.topk(dists, k=knn_k, dim=-1, largest=False)
            knn_dists = sorted_dists[:, 1:]  # Remove self (distance=0)

            # Compute variance of neighbor distances as complexity measure
            local_var = knn_dists.var(dim=-1).mean()
            complexity_list.append(local_var)

        # Handle empty list case
        if len(complexity_list) == 0:
            complexity = torch.zeros(B, dtype=points.dtype, device=points.device)
        else:
            complexity = torch.stack(complexity_list, dim=0)  # (B,)
        return complexity

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats_f,
        src_feats_f,
        ref_feats_mid,
        src_feats_mid,
        ref_masks,
        src_masks,
    ):
        """Adaptive feature fusion at fine-level points.

        Args:
            ref_points: (B, K, 3) reference points
            src_points: (B, K, 3) source points
            ref_feats_f: (B, K, C) fine-level reference features
            src_feats_f: (B, K, C) fine-level source features
            ref_feats_mid: (B, K, C) mid-level features interpolated to fine points
            src_feats_mid: (B, K, C) mid-level features interpolated to fine points
            ref_masks: (B, K) masks for reference points
            src_masks: (B, K) masks for source points

        Returns:
            ref_feats: (B, K, C) fused reference features
            src_feats: (B, K, C) fused source features
            alpha: (B,) fusion weights per patch
        """
        # Compute complexity for ref and src patches
        ref_c = self.compute_complexity(ref_points, ref_masks)  # (B,)
        src_c = self.compute_complexity(src_points, src_masks)  # (B,)

        # Average complexity for the correspondence pair
        c = 0.5 * (ref_c + src_c)  # (B,)

        # Sigmoid mapping: complexity -> alpha
        alpha = torch.sigmoid(self.scale * (c - self.bias))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha  # (B,)

        # Expand for broadcasting: (B,) -> (B, 1, 1)
        alpha_expanded = alpha.view(-1, 1, 1)

        # Fuse features: high complexity -> high alpha -> more fine features
        ref_feats = alpha_expanded * ref_feats_f + (1 - alpha_expanded) * ref_feats_mid
        src_feats = alpha_expanded * src_feats_f + (1 - alpha_expanded) * src_feats_mid

        return ref_feats, src_feats, alpha


# Backward compatibility alias
AdaptiveMultiScaleFusionV2 = AdaptiveFeatureFusion
