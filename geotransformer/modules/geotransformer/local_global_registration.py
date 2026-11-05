from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from geotransformer.modules.ops import apply_transform
from geotransformer.modules.registration import WeightedProcrustes


class LocalGlobalRegistration(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        correspondence_threshold: int = 3,
        correspondence_limit: Optional[int] = None,
        num_refinement_steps: int = 5,
        use_residual_refinement: bool = False,
        num_iterative_steps: int = 3,
        residual_sigma: float = 0.1,
        geo_lambda: float = 0.3,
        rerun_ot: bool = False,
        geo_term_min: float = -0.3,
        geo_term_max: float = 0.0,
        confidence_gate: bool = True,
        topk_refine: bool = True,
        refine_topk: int = 5,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            correspondence_threshold (int=3): minimal number of correspondences for each patch correspondence.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
            use_residual_refinement (bool=False): whether to use residual-guided iterative refinement.
            num_iterative_steps (int=3): number of iterative refinement steps.
            residual_sigma (float=0.1): temperature parameter for geometric consistency.
            geo_lambda (float=0.3): weight for geometric consistency update.
            rerun_ot (bool=False): whether to rerun optimal transport in each iteration.
            geo_term_min (float=-0.3): minimum clipping value for geometric term.
            geo_term_max (float=0.0): maximum clipping value for geometric term.
            confidence_gate (bool=True): whether to use confidence gating for refinement.
            topk_refine (bool=True): whether to only refine top-k candidates per ref point.
            refine_topk (int=5): number of top candidates to refine per ref point.
        """
        super(LocalGlobalRegistration, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = correspondence_threshold
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = num_refinement_steps
        self.use_residual_refinement = use_residual_refinement
        self.num_iterative_steps = num_iterative_steps
        self.residual_sigma = residual_sigma
        self.geo_lambda = geo_lambda
        self.rerun_ot = rerun_ot
        self.geo_term_min = geo_term_min
        self.geo_term_max = geo_term_max
        self.confidence_gate = confidence_gate
        self.topk_refine = topk_refine
        self.refine_topk = refine_topk
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        r"""Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        # merge results from two sides
        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)

        if self.use_dustbin:
            corr_mat = corr_mat[:, -1:, -1]

        corr_mat = torch.logical_and(corr_mat, mask_mat)

        return corr_mat

    @staticmethod
    def convert_to_batch(ref_corr_points, src_corr_points, corr_scores, chunks):
        r"""Convert stacked correspondences to batched points.

        The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
        transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
        into a batch.

        Args:
            ref_corr_points (Tensor): (C, 3)
            src_corr_points (Tensor): (C, 3)
            corr_scores (Tensor): (C,)
            chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

        Returns:
            batch_ref_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_corr_scores (Tensor): (B, K), padded with zeros.
        """
        batch_size = len(chunks)
        indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
        ref_corr_points = ref_corr_points[indices]  # (total, 3)
        src_corr_points = src_corr_points[indices]  # (total, 3)
        corr_scores = corr_scores[indices]  # (total,)

        max_corr = np.max([y - x for x, y in chunks])
        target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
        indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
        indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total,) -> (total, 3)
        indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

        batch_ref_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_ref_corr_points.index_put_([indices0, indices1], ref_corr_points)
        batch_ref_corr_points = batch_ref_corr_points.view(batch_size, max_corr, 3)

        batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
        batch_src_corr_points = batch_src_corr_points.view(batch_size, max_corr, 3)

        batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
        batch_corr_scores.index_put_([indices], corr_scores)
        batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

        return batch_ref_corr_points, batch_src_corr_points, batch_corr_scores

    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def iterative_refinement_with_residual(
        self, ref_knn_points, src_knn_points, ref_knn_masks, src_knn_masks, score_mat, initial_score_mat
    ):
        r"""Residual-guided iterative refinement of matching scores.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) current matching scores
            initial_score_mat (Tensor): (B, K, K) initial matching scores from OT

        Returns:
            refined_score_mat (Tensor): (B, K, K) refined matching scores
            estimated_transform (Tensor): (4, 4) estimated transformation
        """
        from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport

        batch_size = ref_knn_points.shape[0]
        refined_score_mat = score_mat.clone()
        estimated_transform = None

        for iter_idx in range(self.num_iterative_steps):
            # Extract correspondences from current scores
            corr_mat = self.compute_correspondence_matrix(
                torch.exp(refined_score_mat), ref_knn_masks, src_knn_masks
            )

            batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
            # Only break if we have absolutely no correspondences
            # Note: correspondence_threshold is for per-patch, not total
            if batch_indices.shape[0] == 0:
                break

            ref_corr_points = ref_knn_points[batch_indices, ref_indices]
            src_corr_points = src_knn_points[batch_indices, src_indices]
            corr_scores = torch.exp(refined_score_mat)[batch_indices, ref_indices, src_indices]

            # Estimate current transformation
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)

            # Align src patches to ref
            aligned_src_knn_points = apply_transform(src_knn_points, estimated_transform)  # (B, K, 3)

            # Compute geometric residuals (pairwise distances)
            ref_expanded = ref_knn_points.unsqueeze(2)  # (B, K_ref, 1, 3)
            src_expanded = aligned_src_knn_points.unsqueeze(1)  # (B, 1, K_src, 3)
            geometric_residuals = torch.linalg.norm(ref_expanded - src_expanded, dim=-1)  # (B, K_ref, K_src)

            # Compute geometric consistency scores
            geometric_consistency = torch.exp(-geometric_residuals / self.residual_sigma)  # (B, K, K)

            # Compute geometric term: log(g_ij), clipped to [geo_term_min, geo_term_max]
            geo_term = torch.log(geometric_consistency + 1e-8)
            geo_term = torch.clamp(geo_term, min=self.geo_term_min, max=self.geo_term_max)

            # Confidence gating: only apply refinement to low-confidence matches
            if self.confidence_gate:
                # Compute confidence from current scores (normalized to [0, 1])
                score_probs = torch.exp(refined_score_mat)  # (B, K, K)
                # Normalize per row (ref point)
                score_probs_norm = score_probs / (score_probs.sum(dim=2, keepdim=True) + 1e-8)
                # Low confidence gate: 1 - confidence
                confidence_gate = 1.0 - score_probs_norm  # (B, K, K)
                # Apply gate to geometric term
                geo_term = confidence_gate * geo_term

            # Top-k mask: only refine top-k candidates per ref point
            if self.topk_refine:
                topk_mask = torch.zeros_like(refined_score_mat, dtype=torch.bool)
                refine_topk = min(self.refine_topk, refined_score_mat.shape[2])
                topk_idx = refined_score_mat.topk(k=refine_topk, dim=2).indices  # (B, K_ref, topk)
                topk_mask.scatter_(2, topk_idx, True)  # (B, K_ref, K_src)
                # Only apply geo_term to top-k candidates
                geo_term = torch.where(topk_mask, geo_term, torch.zeros_like(geo_term))

            # Mask out invalid point pairs (padding points)
            valid_pair_mask = torch.logical_and(
                ref_knn_masks.unsqueeze(2),  # (B, K_ref, 1)
                src_knn_masks.unsqueeze(1)   # (B, 1, K_src)
            )  # (B, K_ref, K_src)
            geo_term = torch.where(valid_pair_mask, geo_term, torch.zeros_like(geo_term))

            # Iterative update: add geometric consistency term to current scores (incremental)
            refined_score_mat = refined_score_mat + self.geo_lambda * geo_term

        return refined_score_mat, estimated_transform

    def local_to_global_registration(self, ref_knn_points, src_knn_points, score_mat, corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        # build verification set
        if self.correspondence_limit is not None and global_corr_scores.shape[0] > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
        else:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            corr_scores = global_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [
            (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.correspondence_threshold
        ]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = self.convert_to_batch(
                global_ref_corr_points, global_src_corr_points, global_corr_scores, chunks
            )
            batch_transforms = self.procrustes(batch_src_corr_points, batch_ref_corr_points, batch_corr_scores)
            batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), batch_transforms)
            batch_corr_residuals = torch.linalg.norm(
                ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2
            )
            batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = batch_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)

        Returns:
            ref_corr_points: torch.LongTensor (C, 3)
            src_corr_points: torch.LongTensor (C, 3)
            corr_scores: torch.Tensor (C,)
            estimated_transform: torch.Tensor (4, 4)
        """
        initial_score_mat = score_mat.clone()

        # Apply residual-guided iterative refinement if enabled
        if self.use_residual_refinement:
            score_mat, _ = self.iterative_refinement_with_residual(
                ref_knn_points, src_knn_points, ref_knn_masks, src_knn_masks, score_mat, initial_score_mat
            )

        score_mat = torch.exp(score_mat)

        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            ref_knn_points, src_knn_points, score_mat, corr_mat
        )

        return ref_corr_points, src_corr_points, corr_scores, estimated_transform
