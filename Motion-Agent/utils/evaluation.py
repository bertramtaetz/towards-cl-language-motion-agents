
from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order
import os
import torch.nn as nn
import numpy as np
import torch
from scipy import linalg
from pathlib import Path

# Get Motion-Agent root directory for absolute paths
_MOTION_AGENT_ROOT = Path(__file__).parent.parent.resolve()

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer(str(PRETRAINED / 'glove'), 'our_vab')

from utils.motion_utils import plot_3d_motion, recover_from_ric
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


@torch.no_grad()        
def evaluation_test(out_dir, val_loader, model, eval_wrapper, draw = False, savenpy=False) : 

    device = model.device

    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    mm_dist_real = 0
    mm_dist_pred = 0

    nb_sample = 0
    
    for batch in val_loader:

        word_embeddings, pos_one_hots, caption, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(device)
        pred_len = torch.ones(bs).long()
        
        # Use batch generation for speedup
        try:
            # Check if model has generate_batch method
            if hasattr(model, 'generate_batch'):
                batch_motion_tokens = model.generate_batch(list(caption))
            else:
                # Fallback to sequential generation
                batch_motion_tokens = [model.generate(cap) for cap in caption]
        except Exception as e:
            # If batch generation fails entirely, fall back to sequential
            batch_motion_tokens = []
            for cap in caption:
                try:
                    batch_motion_tokens.append(model.generate(cap))
                except:
                    batch_motion_tokens.append(torch.ones(1, dtype=torch.long, device=device))
        
        # Process each generated motion
        for k in range(bs):
            try:
                if k < len(batch_motion_tokens):
                    index_motion = batch_motion_tokens[k]
                else:
                    index_motion = torch.ones(1, dtype=torch.long, device=device)
                pred_pose = model.net.forward_decoder(index_motion)
            except:
                index_motion = torch.ones(1, 1).to(device).long()
                pred_pose = model.net.forward_decoder(index_motion)
            
            cur_len = pred_pose.shape[1]
            # Ensure minimum length of 1 to avoid pack_padded_sequence error
            pred_len[k] = max(1, min(cur_len, seq))
            if cur_len > 0:
                pred_pose_eval[k:k+1, :min(cur_len, seq)] = pred_pose[:, :seq]

            if draw or savenpy:
                pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().to(device), num_joints)

                if savenpy:
                    np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                if draw:
                    draw_pred.append(pred_xyz)
                    draw_text_pred.append(caption[k])
                    draw_name.append(name[k])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

        pose = pose.to(device).float()
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        if draw or savenpy:
            pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().to(device), num_joints)

            if savenpy:
                for j in range(bs):
                    np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

            if draw:
                for j in range(bs):
                    draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                    draw_text.append(caption[j])

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        mm_dist_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        mm_dist_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    # Adjust diversity_times based on available samples (need at least 2 samples)
    diversity_times = min(300 if nb_sample > 300 else 100, nb_sample - 1)
    diversity_times = max(diversity_times, 2)  # Minimum 2 for meaningful diversity
    
    if nb_sample > diversity_times:
        diversity_real = calculate_diversity(motion_annotation_np, diversity_times)
        diversity = calculate_diversity(motion_pred_np, diversity_times)
    else:
        # Not enough samples for diversity calculation
        diversity_real = 0.0
        diversity = 0.0

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    mm_dist_real = mm_dist_real / nb_sample
    mm_dist_pred = mm_dist_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, mm_dist_real. {mm_dist_real}, mm_dist_pred. {mm_dist_pred}"
    print(msg)
    
    
    if draw:
        for ii in range(len(draw_org)):
            pass

    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], mm_dist_pred


def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        # print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    mm_dist = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    # Clamp top_k to available number of samples to avoid IndexError
    n_samples = argmax.shape[1]
    effective_top_k = min(top_k, n_samples)
    top_k_mat = calculate_top_k(argmax, effective_top_k)
    
    # Pad with False if effective_top_k < top_k (for small batches)
    if effective_top_k < top_k:
        padding = np.zeros((top_k_mat.shape[0], top_k - effective_top_k), dtype=bool)
        top_k_mat = np.concatenate([top_k_mat, padding], axis=1)
    
    if sum_all:
        return top_k_mat.sum(axis=0), mm_dist
    else:
        return top_k_mat, mm_dist

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # return 1000
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist
