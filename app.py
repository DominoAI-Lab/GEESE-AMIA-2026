"""
HONK: Interactive Behavioral Phenotyping Agent
"""

import os
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import pickle
import scipy.io as sio
from sklearn.decomposition import PCA
from umap import UMAP
from momentfm import MOMENTPipeline
import json
import io
import tempfile
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd
from huggingface_hub import hf_hub_download, InferenceClient
from sklearn.model_selection import GroupShuffleSplit
import requests
import traceback


# CONFIGURATION
HF_DATA_REPO = os.environ.get("HF_DATA_REPO", "TO BE FILLED after my hf grant is approved")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# otherwise falls back to free HF Inference API (Qwen-2.5-72B)

def hf_path(filename):
    if os.path.exists(filename):
        return filename
    return hf_hub_download(repo_id=HF_DATA_REPO, filename=filename, repo_type="dataset")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 32
FPS = 50

COHORT_CONFIG = {
    'CNTNAP': {
        'data_dir': '/data/datasets/sdannce/CNTNAPcohort/lone',  # local fallback only
        'result_dir': 'results/class_cntnapL_32',
        'hlac_model': 'results/class/hlac/cntnapL/trial1/best_model_hlac.pth',
        'hlac_results': 'results/class/hlac/cntnapL/trial1/results_hlac.json',
        'genotype_model': 'results/rep/gen_few/cntnapL/trial1/best_model_genotype.pth',
        'manifold_pkl': 'precomputed/cntnap_manifold.pkl',
        'n_hlac_classes': 9,
        'n_genotype_classes': 3,
    },
    'CHD8': {
        'data_dir': '/data/datasets/sdannce/CHD8cohort/lone',
        'result_dir': 'results/class_chd8L_32',
        'hlac_model': 'results/class/hlac/chd8L/trial1/best_model_hlac.pth',
        'hlac_results': 'results/class/hlac/chd8L/trial1/results_hlac.json',
        'genotype_model': 'results/rep/gen_few/chd8L/trial3/best_model_genotype.pth',
        'manifold_pkl': 'precomputed/chd8_manifold.pkl',
        'n_hlac_classes': 9,
        'n_genotype_classes': 2,
    },
    'FMR1': {
        'data_dir': '/data/datasets/sdannce/FMR1cohort/lone',
        'result_dir': 'results/class_fmr1L_32',
        'hlac_model': 'results/class/hlac/fmr1L/trial1/best_model_hlac.pth',
        'hlac_results': 'results/class/hlac/fmr1L/trial1/results_hlac.json',
        'genotype_model': 'results/rep/gen_few/fmr1L/trial3/best_model_genotype.pth',
        'manifold_pkl': 'precomputed/fmr1_manifold.pkl',
        'n_hlac_classes': 9,
        'n_genotype_classes': 2,
    }
}

GENOTYPE_NAMES = ['WT', 'HET', 'HOM']
GENOTYPE_COLORS = {'WT': '#1f77b4', 'HET': '#d62728', 'HOM': '#2ca02c'}

SEMANTIC_DESCRIPTIONS = {
    1: "idle",
    2: "sniff",
    3: "groom", 
    4: "scrunched",
    5: "reared",
    6: "active crouch",
    7: "explore",
    8: "locomotion",
    9: "fast"
}

def get_behavior_label(idx):
    """index format: 'Behavior N: semantic'"""
    hlac_num = idx + 1  # idx 0-based, HLAC 1-based
    semantic = SEMANTIC_DESCRIPTIONS.get(hlac_num, "unknown")
    return f"Behavior {hlac_num}: {semantic}"


class CohortAnalyzer:
    def __init__(self, cohort_name):
        self.cohort_name = cohort_name
        self.config = COHORT_CONFIG[cohort_name]
        self.hlac_model = None
        self.genotype_model = None
        self.embeddings = None
        self.umap_result = None
        self.hlac_centroids = None
        self.genotype_centroids = None
        self.test_windows = None
        self.test_labels = None
        self.test_metadata = None
        self.class_names = None
        self.hlac_to_idx = None
        self.idx_to_hlac = None
        
    def load_models(self):
        print(f"Loading models for {self.cohort_name}")
        
        self.hlac_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": 69,
                "num_class": self.config['n_hlac_classes']
            }
        )
        self.hlac_model.init()
        checkpoint = torch.load(hf_path(self.config['hlac_model']), map_location='cpu')
        self.hlac_model.load_state_dict(checkpoint['model_state_dict'])
        self.hlac_model = self.hlac_model.to(DEVICE)
        self.hlac_model.eval()
        print(f"  HLAC model loaded (val_acc={checkpoint['val_acc']:.4f})")
        
        self.genotype_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": 69,
                "num_class": self.config['n_genotype_classes']
            }
        )
        self.genotype_model.init()
        checkpoint = torch.load(hf_path(self.config['genotype_model']), map_location='cpu')
        self.genotype_model.load_state_dict(checkpoint['model_state_dict'])
        self.genotype_model = self.genotype_model.to(DEVICE)
        self.genotype_model.eval()
        print(f"  Genotype model loaded")
        
        with open(hf_path(self.config['hlac_results']), 'r') as f:
            hlac_results = json.load(f)
        self.class_names = hlac_results['class_names']
        self.hlac_to_idx = {int(k): v for k, v in hlac_results['hlac_to_idx'].items()}
        self.idx_to_hlac = {int(k): v for k, v in hlac_results['idx_to_hlac'].items()}
        
    def load_test_data(self):
        print(f"Loading test data for {self.cohort_name}")
        
        windows = torch.load(f"{self.config['result_dir']}/windows.pt")
        with open(f"{self.config['result_dir']}/window_metadata.pkl", 'rb') as f:
            window_metadata = pickle.load(f)
        
        m1_mask = np.array(['_m2' not in meta['filename'] for meta in window_metadata])
        windows = windows[m1_mask]
        window_metadata = [m for m, keep in zip(window_metadata, m1_mask) if keep]
        
        hlac_dict = {}
        data_dir = Path(self.config['data_dir'])
        for mat_file in sorted(data_dir.glob('*.mat')):
            try:
                mat = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
                sdannce = mat['sdannce']
                if hasattr(sdannce, 'hlac') and sdannce.hlac is not None:
                    hlac = sdannce.hlac.flatten() if sdannce.hlac.ndim > 1 else sdannce.hlac
                    hlac_dict[mat_file.name] = hlac
            except:
                pass
        
        labels = []
        for meta in window_metadata:
            filename = meta['filename']
            middle_idx = meta['frame_start'] + WINDOW_SIZE // 2
            if filename in hlac_dict and middle_idx < len(hlac_dict[filename]):
                labels.append(int(hlac_dict[filename][middle_idx]))
            else:
                labels.append(-1)
        labels = np.array(labels)
        
        valid_mask = labels != -1
        windows = windows[valid_mask]
        labels = labels[valid_mask]
        window_metadata = [m for m, v in zip(window_metadata, valid_mask) if v]
        
        labels = np.array([self.hlac_to_idx.get(l, 0) for l in labels])
        
        
        session_ids = np.array([meta['filename'] for meta in window_metadata])
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=96)
        _, test_idx = next(gss.split(windows, labels, groups=session_ids))
        
        self.test_windows = windows[test_idx]
        self.test_labels = labels[test_idx]
        self.test_metadata = [window_metadata[i] for i in test_idx]
        
        print(f"  Test samples: {len(self.test_windows)}")
        
    def compute_manifold(self, max_plot_samples=5000):
        print(f"Computing manifold for {self.cohort_name}")
        
        # > embed all test windows (no subsampling)
        all_labels = self.test_labels
        all_meta = self.test_metadata
        
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.test_windows), 64):
                batch = torch.FloatTensor(self.test_windows[i:i+64]).to(DEVICE)
                batch = batch.permute(0, 2, 1)
                #emb = self.hlac_model.embed(x_enc=batch, reduction='mean').embeddings
                emb = self.genotype_model.embed(x_enc=batch, reduction='mean').embeddings
                embeddings.append(emb.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        print(f"  Embeddings: {embeddings.shape}")
        
        # > fit UMAP on all embeddings
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, 
                       metric='cosine', random_state=42)
        umap_all = reducer.fit_transform(embeddings)
        
        # > compute centroids from all data
        hlac_centroids = {}
        hlac_centroids_umap = {}
        for i in range(len(self.class_names)):
            mask = all_labels == i
            if mask.sum() > 0:
                hlac_centroids[i] = embeddings[mask].mean(axis=0)
                hlac_centroids_umap[i] = umap_all[mask].mean(axis=0)
        
        genotypes = np.array([m['ratgen'] for m in all_meta])
        genotype_centroids = {}
        genotype_centroids_umap = {}
        for i in range(3):
            mask = genotypes == i
            if mask.sum() > 0:
                genotype_centroids[i] = embeddings[mask].mean(axis=0)
                genotype_centroids_umap[i] = umap_all[mask].mean(axis=0)
        
        # > subsample for plotting only
        if len(self.test_windows) > max_plot_samples:
            plot_idx = np.random.choice(len(self.test_windows), max_plot_samples, replace=False)
        else:
            plot_idx = np.arange(len(self.test_windows))
        
        self.embeddings = embeddings
        self.umap_result = umap_all[plot_idx]
        self.labels_sample = all_labels[plot_idx]
        self.meta_sample = [all_meta[i] for i in plot_idx]
        self.hlac_centroids = hlac_centroids
        self.hlac_centroids_umap = hlac_centroids_umap
        self.genotype_centroids = genotype_centroids
        self.genotype_centroids_umap = genotype_centroids_umap
        self.umap_reducer = reducer
        
        print(f"  Manifold fitted on {len(embeddings)} points, plotting {len(plot_idx)}")

    def save_manifold_pkl(self, path):
        data = {
            'embeddings': self.embeddings,
            'umap_result': self.umap_result,
            'labels_sample': self.labels_sample,
            'meta_sample': self.meta_sample,
            'hlac_centroids': self.hlac_centroids,
            'hlac_centroids_umap': self.hlac_centroids_umap,
            'genotype_centroids': self.genotype_centroids,
            'genotype_centroids_umap': self.genotype_centroids_umap,
            'umap_reducer': self.umap_reducer,
            'test_windows': self.test_windows,
            'test_labels': self.test_labels,
            'test_metadata': self.test_metadata,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Manifold saved to {path}")

    def load_precomputed_manifold(self):
        pkl_path = hf_path(self.config['manifold_pkl'])
        print(f"  Loading precomputed manifold from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.umap_result = data['umap_result']
        self.labels_sample = data['labels_sample']
        self.meta_sample = data['meta_sample']
        self.hlac_centroids = data['hlac_centroids']
        self.hlac_centroids_umap = data['hlac_centroids_umap']
        self.genotype_centroids = data['genotype_centroids']
        self.genotype_centroids_umap = data['genotype_centroids_umap']
        self.umap_reducer = data['umap_reducer']
        self.test_windows = data['test_windows']
        self.test_labels = data['test_labels']
        self.test_metadata = data['test_metadata']
        print(f"  Manifold loaded: {self.embeddings.shape}")
        
    def analyze_window(self, window):
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)
        window_tensor = window_tensor.permute(0, 2, 1)
        
        with torch.no_grad():
            hlac_output = self.hlac_model.classify(x_enc=window_tensor)
            hlac_logits = hlac_output.logits[0].cpu().numpy()
            hlac_probs = np.exp(hlac_logits) / np.exp(hlac_logits).sum()
            hlac_pred = np.argmax(hlac_probs)
            hlac_conf = hlac_probs[hlac_pred]
            
            gen_output = self.genotype_model.classify(x_enc=window_tensor)
            gen_logits = gen_output.logits[0].cpu().numpy()
            gen_probs = np.exp(gen_logits) / np.exp(gen_logits).sum()
            gen_pred = np.argmax(gen_probs)
            gen_conf = gen_probs[gen_pred]
            
            #embedding = self.hlac_model.embed(x_enc=window_tensor, reduction='mean').embeddings[0].cpu().numpy()
            embedding = self.genotype_model.embed(x_enc=window_tensor, reduction='mean').embeddings[0].cpu().numpy()
        
        hlac_distances = {}
        for i, centroid in self.hlac_centroids.items():
            dist = np.linalg.norm(embedding - centroid)
            hlac_distances[i] = dist
        
        genotype_distances = {}
        for i, centroid in self.genotype_centroids.items():
            dist = np.linalg.norm(embedding - centroid)
            genotype_distances[i] = dist
        
        sample_umap = self.umap_reducer.transform(embedding.reshape(1, -1))[0]
        
        return {
            'hlac_pred': hlac_pred,
            'hlac_conf': hlac_conf,
            'hlac_probs': hlac_probs,
            'hlac_distances': hlac_distances,
            'genotype_pred': gen_pred,
            'genotype_conf': gen_conf,
            'genotype_probs': gen_probs,
            'genotype_distances': genotype_distances,
            'embedding': embedding,
            'umap_pos': sample_umap
        }
    
    def analyze_time_range(self, pose_data, start_sec, end_sec):

        start_frame = int(start_sec * FPS)
        end_frame = int(end_sec * FPS)
        
        n_frames = pose_data.shape[0]
        start_frame = max(0, min(start_frame, n_frames - WINDOW_SIZE))
        end_frame = min(end_frame, n_frames)
        
        results = []
        timestamps = []
        embeddings = []
        
        # > extract windows with stride
        stride = WINDOW_SIZE // 2
        for frame_start in range(start_frame, end_frame - WINDOW_SIZE + 1, stride):
            window = pose_data[frame_start:frame_start + WINDOW_SIZE]
            result = self.analyze_window(window)
            results.append(result)
            timestamps.append((frame_start + WINDOW_SIZE // 2) / FPS)
            embeddings.append(result['embedding'])
        
        if len(results) == 0:
            return None
        
        # > all embeddings to UMAP
        embeddings = np.array(embeddings)
        umap_positions = self.umap_reducer.transform(embeddings)
        for i, result in enumerate(results):
            result['umap_pos'] = umap_positions[i]
        
        return {
            'results': results,
            'timestamps': timestamps,
            'embeddings': embeddings,
            'umap_positions': umap_positions
        }


def load_mat_file(file_path):

    try:
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        sdannce = mat['sdannce']
        
        pose = None
        if hasattr(sdannce, 'm1') and sdannce.m1 is not None and len(sdannce.m1) > 0:
            pose = sdannce.m1
        elif hasattr(sdannce, 'data') and sdannce.data is not None:
            pose = sdannce.data
        
        if pose is None:
            return None, "Cannot find pose data in .mat file"
        
        # (n_frames, n_features)
        if pose.ndim == 3:
            n_frames = pose.shape[0]
            pose = pose.reshape(n_frames, -1)
        

        n_frames = pose.shape[0]
        duration_sec = n_frames / FPS
        duration_min = duration_sec / 60
        
        info = {
            'n_frames': n_frames,
            'duration_sec': duration_sec,
            'duration_min': duration_min,
            'n_features': pose.shape[1]
        }
        
        # > get HLAC if available
        if hasattr(sdannce, 'hlac') and sdannce.hlac is not None:
            hlac = sdannce.hlac.flatten() if sdannce.hlac.ndim > 1 else sdannce.hlac
            info['hlac'] = hlac
        
        # > get genotype if available
        if hasattr(sdannce, 'ratgen'):
            info['ratgen'] = int(sdannce.ratgen)
        
        return pose, info
        
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def create_manifold_plot(analyzer, trajectory_data):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.cm.tab10

    # > Semantic manifold
    hlac_handles = []
    for i, name in enumerate(analyzer.class_names):
        mask = analyzer.labels_sample == i
        behavior_label = get_behavior_label(i)
        sc = axes[0].scatter(
            analyzer.umap_result[mask, 0], analyzer.umap_result[mask, 1],
            c=[cmap(i)], s=5, alpha=0.3, label=behavior_label
        )
        hlac_handles.append(sc)

    # > centroids: white-edged circle + text label (with offset to avoid overlap)
    for i, pos in analyzer.hlac_centroids_umap.items():
        axes[0].scatter(
            pos[0], pos[1], c=cmap(i), s=150, marker='o',
            edgecolors='white', linewidths=2, zorder=10
        )
        label_txt = get_behavior_label(i)

        axes[0].annotate(
            label_txt, (pos[0], pos[1]),
            xytext=(5, 8), textcoords='offset points',
            fontsize=6, weight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=12
        )

    # > trajectory points (light purple, unobtrusive)
    umap_pos = trajectory_data['umap_positions']
    axes[0].scatter(umap_pos[:, 0], umap_pos[:, 1], c='#B19CD9', s=25, 
                    alpha=0.6, zorder=15, edgecolors='white', linewidths=0.3)

    axes[0].set_title('Behavioral Manifold (Semantic)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Legend 1: Semantic classes
    leg1 = axes[0].legend(handles=hlac_handles, loc='upper right', fontsize=6, markerscale=2)
    axes[0].add_artist(leg1)
    
    # Legend 2: Centroid marker
    centroid_handle = Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
                             markeredgecolor='white', markersize=10, label='Centroid')
    axes[0].legend(handles=[centroid_handle], loc='upper right', bbox_to_anchor=(1.0, 0.65), fontsize=9)

    # RIGHT: Genotype manifold
    geno_handles = []
    genotypes = np.array([m['ratgen'] for m in analyzer.meta_sample])
    for i, name in enumerate(GENOTYPE_NAMES):
        mask = genotypes == i
        if mask.sum() > 0:
            sc = axes[1].scatter(
                analyzer.umap_result[mask, 0], analyzer.umap_result[mask, 1],
                c=GENOTYPE_COLORS[name], s=5, alpha=0.3, label=name
            )
            geno_handles.append(sc)

    # > genotype centroids: white-edged circle + text
    geno_color_list = ['#1f77b4', '#d62728', '#2ca02c']
    for i, pos in analyzer.genotype_centroids_umap.items():
        axes[1].scatter(
            pos[0], pos[1], c=geno_color_list[i], s=150, marker='o',
            edgecolors='white', linewidths=2, zorder=10
        )
        label_txt = GENOTYPE_NAMES[i] if i < len(GENOTYPE_NAMES) else f"GEN_{i}"
        axes[1].annotate(
            label_txt, (pos[0], pos[1]),
            xytext=(5, 8), textcoords='offset points',
            fontsize=8, weight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=12
        )



    axes[1].set_title('Behavioral Manifold (Genotype)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    leg1b = axes[1].legend(handles=geno_handles, loc='upper right', fontsize=9, markerscale=2)
    axes[1].add_artist(leg1b)
    
    centroid_handle2 = Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
                              markeredgecolor='white', markersize=10, label='Centroid')
    axes[1].legend(handles=[centroid_handle2], loc='upper right', bbox_to_anchor=(1.0, 0.7), fontsize=9)

    plt.tight_layout()
    return fig


def create_trajectory_plot(analyzer, trajectory_data, show_gt=True):
    """behavior sequence over time"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    
    timestamps = trajectory_data['timestamps']
    results = trajectory_data['results']
    
    hlac_preds = [r['hlac_pred'] for r in results]
    
    cmap = plt.cm.get_cmap('tab10', len(analyzer.class_names))
    colors = [cmap(p) for p in hlac_preds]
    
    if show_gt and 'hlac_true' in trajectory_data:
        ax.plot(timestamps, trajectory_data['hlac_true'], 
                color='gray', alpha=0.5, linewidth=4, label='GT', zorder=1)
    
    ax.plot(timestamps, hlac_preds, color='blue', alpha=0.5, linewidth=1.5, zorder=2)
    ax.scatter(timestamps, hlac_preds, c=colors, s=50, edgecolors='black', linewidths=0.5, zorder=3)
    
    behavior_labels = [get_behavior_label(i) for i in range(len(analyzer.class_names))]
    ax.set_yticks(range(len(analyzer.class_names)))
    ax.set_yticklabels(behavior_labels)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Predicted Behavior')
    ax.set_title('Behavior Sequence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if show_gt and 'hlac_true' in trajectory_data:
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_summary_plot(analyzer, trajectory_data, show_gt=True):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    results = trajectory_data['results']
    
    # > Behavior distribution: Prediction vs GT
    hlac_preds = [r['hlac_pred'] for r in results]
    pred_counts = np.bincount(hlac_preds, minlength=len(analyzer.class_names))
    pred_pct = pred_counts / len(hlac_preds) * 100
    
    cmap = plt.cm.get_cmap('tab10', len(analyzer.class_names))
    
    y_pos = np.arange(len(analyzer.class_names))
    bar_height = 0.35
    
    behavior_labels = [get_behavior_label(i) for i in range(len(analyzer.class_names))]
    
    bars1 = axes[0].barh(y_pos + bar_height/2, pred_pct, bar_height, 
                         color=[cmap(i) for i in range(len(analyzer.class_names))], 
                         edgecolor='black', label='Prediction')
    
    if show_gt and 'hlac_true' in trajectory_data:
        gt_counts = np.bincount(trajectory_data['hlac_true'], minlength=len(analyzer.class_names))
        gt_pct = gt_counts / len(trajectory_data['hlac_true']) * 100
        bars2 = axes[0].barh(y_pos - bar_height/2, gt_pct, bar_height,
                             color=[cmap(i) for i in range(len(analyzer.class_names))],
                             edgecolor='black', alpha=0.5, hatch='//', label='Ground Truth')
        axes[0].legend(loc='lower right', fontsize=9)
    
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(behavior_labels)
    axes[0].set_xlabel('Time Spent (%)')
    axes[0].set_title('Behavior Distribution', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    
    for bar, pct in zip(bars1, pred_pct):
        if pct > 0:
            axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{pct:.1f}%', va='center', fontsize=8)
    
    gen_probs_all = np.array([r['genotype_probs'] for r in results])
    gen_probs_mean = gen_probs_all.mean(axis=0)
    gen_probs_std = gen_probs_all.std(axis=0)
    
    x = np.arange(len(GENOTYPE_NAMES))
    bars_gen = axes[1].bar(x, gen_probs_mean, yerr=gen_probs_std, 
                           color=['#1f77b4', '#d62728', '#2ca02c'][:len(gen_probs_mean)],
                           edgecolor='black', capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(GENOTYPE_NAMES[:len(gen_probs_mean)])
    axes[1].set_ylabel('Mean Probability')
    axes[1].set_ylim([0, 1])
    axes[1].set_title('Genotype Prediction', fontsize=12, fontweight='bold')
    
    if show_gt and 'genotype_true' in trajectory_data:
        gt = trajectory_data['genotype_true']
        if 0 <= gt < len(gen_probs_mean):
            bars_gen[gt].set_edgecolor('gold')
            bars_gen[gt].set_linewidth(3)
            axes[1].annotate('(GT)', (gt, -0.08), ha='center', fontsize=9, 
                           fontweight='bold', annotation_clip=False)
    
    plt.tight_layout()
    return fig


def generate_trajectory_report(analyzer, trajectory_data, file_info=None):

    results = trajectory_data['results']
    timestamps = trajectory_data['timestamps']
    
    duration = timestamps[-1] - timestamps[0]
    n_windows = len(results)
    
    hlac_preds = [r['hlac_pred'] for r in results]
    counts = np.bincount(hlac_preds, minlength=len(analyzer.class_names))
    percentages = counts / len(hlac_preds) * 100
    
    dominant_idx = np.argmax(counts)
    dominant_behavior = analyzer.class_names[dominant_idx]
    
    gen_probs_all = np.array([r['genotype_probs'] for r in results])
    gen_probs_mean = gen_probs_all.mean(axis=0)
    predicted_gen = GENOTYPE_NAMES[np.argmax(gen_probs_mean)]
    gen_confidence = gen_probs_mean.max()
    
    report = f"""
# Behavioral Analysis Report

## Session Overview
Analysis Duration: {duration:.1f} seconds
Windows Analyzed: {n_windows}
Window Size: {WINDOW_SIZE} frames ({WINDOW_SIZE/FPS:.2f} sec)

## Dominant Behavior
**{dominant_behavior}** ({percentages[dominant_idx]:.1f}% of time)

## Behavior Distribution
"""
    
    sorted_idx = np.argsort(percentages)[::-1]
    for idx in sorted_idx:
        if percentages[idx] > 0:
            name = analyzer.class_names[idx]
            pct = percentages[idx]
            bar = '#' * int(pct / 5) + '.' * (20 - int(pct / 5))
            report += f"- {name}: [{bar}] {pct:.1f}%\n"
    
    report += f"""
## Genotype Prediction
**{predicted_gen}** (Mean Confidence: {gen_confidence*100:.1f}%)

Probability Distribution:
"""
    for i, name in enumerate(GENOTYPE_NAMES[:len(gen_probs_mean)]):
        prob = gen_probs_mean[i]
        std = gen_probs_all[:, i].std()
        report += f"- {name}: {prob*100:.1f}% (+/- {std*100:.1f}%)\n"
    
    # Add ground truth if available
    if file_info and 'hlac' in file_info:
        report += "\n## Ground Truth Available\nHLAC labels present in file for comparison.\n"
    if file_info and 'ratgen' in file_info:
        true_gen = GENOTYPE_NAMES[file_info['ratgen']]
        correct = "CORRECT" if file_info['ratgen'] == np.argmax(gen_probs_mean) else "INCORRECT"
        report += f"\nTrue Genotype: **{true_gen}** - Prediction {correct}\n"
    
    return report


def generate_excel_data(analyzer, trajectory_data):

    results = trajectory_data['results']
    timestamps = trajectory_data['timestamps']
    
    if len(timestamps) > 1:
        dt = timestamps[1] - timestamps[0]
    else:
        dt = WINDOW_SIZE / FPS
    
    hlac_preds = [r['hlac_pred'] for r in results]
    pred_counts = np.bincount(hlac_preds, minlength=len(analyzer.class_names))
    pred_pct = pred_counts / len(hlac_preds) * 100
    pred_time = pred_counts * dt
    
    gen_probs_all = np.array([r['genotype_probs'] for r in results])
    gen_probs_mean = gen_probs_all.mean(axis=0)
    predicted_genotype = GENOTYPE_NAMES[np.argmax(gen_probs_mean)]
    
    # > sheet 1: Summary
    behavior_labels = [get_behavior_label(i) for i in range(len(analyzer.class_names))]
    
    summary_rows = [
        {'Item': 'Predicted Genotype', 'Value': predicted_genotype, 'Percentage': ''},
        {'Item': '', 'Value': '', 'Percentage': ''},  # blank row
        {'Item': 'Behavior Distribution', 'Value': '', 'Percentage': ''},
    ]
    for i, label in enumerate(behavior_labels):
        summary_rows.append({
            'Item': label,
            'Value': f'{pred_time[i]:.1f}s',
            'Percentage': f'{pred_pct[i]:.1f}%'
        })
    df_summary = pd.DataFrame(summary_rows)
    
    # > sheet 2: Behavior Sequence (time series)
    seq_data = {
        'Time (s)': timestamps,
        'Predicted Behavior': [get_behavior_label(r['hlac_pred']) for r in results],
    }
    if 'hlac_true' in trajectory_data:
        seq_data['GT Behavior'] = [get_behavior_label(gt) for gt in trajectory_data['hlac_true']]
    df_seq = pd.DataFrame(seq_data)
    
    # > sheet 3: Genotype Probabilities
    geno_data = {
        'Genotype': GENOTYPE_NAMES[:len(gen_probs_mean)],
        'Mean Probability': [f'{p:.3f}' for p in gen_probs_mean],
        'Std': [f'{s:.3f}' for s in gen_probs_all.std(axis=0)]
    }
    df_geno = pd.DataFrame(geno_data)
    
    output_path = tempfile.mktemp(suffix='.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_seq.to_excel(writer, sheet_name='Behavior Sequence', index=False)
        df_geno.to_excel(writer, sheet_name='Genotype', index=False)
    
    return output_path



current_analysis = {'trajectory_data': None, 'analyzer': None, 'file_info': None}


def build_analysis_context():

    traj = current_analysis['trajectory_data']
    analyzer = current_analysis['analyzer']
    info = current_analysis['file_info']

    if traj is None or analyzer is None:
        return "No analysis has been performed yet. The user should first upload a .mat file or select a test sample and click Analyze."

    results = traj['results']
    timestamps = traj['timestamps']

    hlac_preds = [r['hlac_pred'] for r in results]
    counts = np.bincount(hlac_preds, minlength=len(analyzer.class_names))
    pct = counts / len(hlac_preds) * 100

    dt = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else WINDOW_SIZE / FPS
    duration_total = timestamps[-1] - timestamps[0] + dt

    behavior_lines = []
    for i in range(len(analyzer.class_names)):
        label = get_behavior_label(i)
        time_sec = counts[i] * dt
        behavior_lines.append(f"  {label}: {pct[i]:.1f}% ({time_sec:.1f}s)")

    gen_probs_all = np.array([r['genotype_probs'] for r in results])
    gen_mean = gen_probs_all.mean(axis=0)
    gen_std = gen_probs_all.std(axis=0)
    pred_gen = GENOTYPE_NAMES[np.argmax(gen_mean)]
    gen_lines = []
    for i in range(len(gen_mean)):
        name = GENOTYPE_NAMES[i] if i < len(GENOTYPE_NAMES) else f"Class_{i}"
        gen_lines.append(f"  {name}: {gen_mean[i]*100:.1f}% (std {gen_std[i]*100:.1f}%)")

    gt_info = ""
    if 'hlac_true' in traj:
        gt_counts = np.bincount(traj['hlac_true'], minlength=len(analyzer.class_names))
        gt_pct = gt_counts / len(traj['hlac_true']) * 100
        gt_lines = []
        for i in range(len(analyzer.class_names)):
            if gt_pct[i] > 0:
                gt_lines.append(f"  {get_behavior_label(i)}: {gt_pct[i]:.1f}%")
        gt_info += "\nGround truth behavior distribution:\n" + "\n".join(gt_lines)
    if 'genotype_true' in traj:
        gt_info += f"\nTrue genotype: {GENOTYPE_NAMES[traj['genotype_true']]}"

    transitions = sum(1 for i in range(1, len(hlac_preds)) if hlac_preds[i] != hlac_preds[i-1])
    transition_rate = transitions / duration_total if duration_total > 0 else 0

    ctx = f"""Current analysis results (cohort: {analyzer.cohort_name}):
Analysis duration: {duration_total:.1f} seconds ({len(results)} windows, window size {WINDOW_SIZE} frames at {FPS} fps)

Predicted behavior distribution:
{chr(10).join(behavior_lines)}

Behavior transitions: {transitions} total, rate {transition_rate:.2f}/sec

Genotype prediction: {pred_gen}
Genotype probabilities:
{chr(10).join(gen_lines)}
{gt_info}"""
    return ctx


SYSTEM_PROMPT = """You are HONK, the analysis agent for a mouse behavioral phenotyping system built on the GEESE pipeline.
You have access to analysis results from a MOMENT foundation model that processes 3D pose data.

Given the analysis context below, answer the user's question concisely and accurately.
- For quantitative questions (frequencies, durations, percentages), give exact numbers from the data.
- For genotype questions, report the prediction and confidence.
- For comparison questions (prediction vs ground truth), compare them directly.
- Keep answers short and factual. Do not speculate beyond what the data shows.
- If no analysis has been run yet, tell the user to run an analysis first.

Analysis context:
{context}
"""


def call_llm(messages):
    if OPENAI_API_KEY:
        return _call_openai(messages)
    return _call_hf_inference(messages)


def _call_openai(messages):
    
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o", "messages": messages, "max_tokens": 512},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenAI API error: {str(e)}"


def _call_hf_inference(messages):
    try:
        client = InferenceClient("Qwen/Qwen2.5-72B-Instruct")
        response = client.chat_completion(messages=messages, max_tokens=512)
        return response.choices[0].message.content
    except Exception as e:
        return f"HF Inference error: {str(e)}"


def chatbot_respond(message, history):
    if not message or not message.strip():
        return history

    context = build_analysis_context()
    system_msg = SYSTEM_PROMPT.format(context=context)

    messages = [{"role": "system", "content": system_msg}]
    for user_msg, bot_msg in (history or []):
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = call_llm(messages)
    history = history or []
    history.append((message, response))
    return history


analyzers = {}
uploaded_data = {'pose': None, 'info': None}


def initialize_analyzers():
    global analyzers
    
    for cohort in COHORT_CONFIG.keys():
        print(f"Initializing {cohort} analyzer")
        
        try:
            analyzer = CohortAnalyzer(cohort)
            analyzer.load_models()
            
            try:
                analyzer.load_precomputed_manifold()
            except Exception:
                print(f"  No precomputed manifold, computing from local data...")
                analyzer.load_test_data()
                analyzer.compute_manifold()
            
            analyzers[cohort] = analyzer
            print(f"{cohort} ready")
        except Exception as e:
            print(f"Failed to initialize {cohort}: {e}")
            traceback.print_exc()
            analyzers[cohort] = None


def get_sample_options(cohort):
    if cohort not in analyzers or analyzers[cohort] is None:
        return []
    
    analyzer = analyzers[cohort]
    options = []
    for i, meta in enumerate(analyzer.test_metadata[:50]):
        ratgen = GENOTYPE_NAMES[int(meta['ratgen'])]
        hlac = analyzer.class_names[analyzer.test_labels[i]]
        options.append(f"Sample {i}: {meta['ratid']} ({ratgen}) - {hlac}")
    return options


def on_file_upload(file, cohort):

    global uploaded_data
    
    if file is None:
        return "No file uploaded", gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1)
    
    pose, info = load_mat_file(file.name)
    
    if pose is None:
        uploaded_data = {'pose': None, 'info': None}
        return info, gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1)
    
    uploaded_data = {'pose': pose, 'info': info}
    
    duration = info['duration_sec']
    
    status = f"""File loaded:
- Frames: {info['n_frames']}
- Duration: {info['duration_min']:.1f} minutes ({info['duration_sec']:.1f} seconds)
- Features: {info['n_features']}
"""
    if 'ratgen' in info:
        status += f"- Genotype: {GENOTYPE_NAMES[info['ratgen']]}\n"
    
    return (
        status,
        gr.Slider(minimum=0, maximum=duration, value=0, step=0.1, label="Start Time (sec)"),
        gr.Slider(minimum=0, maximum=duration, value=min(10, duration), step=0.1, label="End Time (sec)")
    )


def analyze_uploaded_file(cohort, start_time, end_time, show_gt):

    global uploaded_data
    
    if uploaded_data['pose'] is None:
        return None, None, None, None
    
    if cohort not in analyzers or analyzers[cohort] is None:
        return None, None, None, None
    
    analyzer = analyzers[cohort]
    pose = uploaded_data['pose']
    info = uploaded_data['info']
    
    if end_time <= start_time:
        return None, None, None, None
    
    trajectory_data = analyzer.analyze_time_range(pose, start_time, end_time)
    
    if trajectory_data is None:
        return None, None, None, None
    
    if info and 'hlac' in info:
        hlac_raw = info['hlac']
        hlac_true = []
        for t in trajectory_data['timestamps']:
            mid = int(round(t * FPS))
            mid = max(0, min(mid, len(hlac_raw) - 1))
            hlac_true.append(analyzer.hlac_to_idx.get(int(hlac_raw[mid]), 0))
        trajectory_data['hlac_true'] = np.array(hlac_true, dtype=int)
    
    if info and 'ratgen' in info:
        trajectory_data['genotype_true'] = int(info['ratgen'])
    
    summary_fig = create_summary_plot(analyzer, trajectory_data, show_gt)
    trajectory_fig = create_trajectory_plot(analyzer, trajectory_data, show_gt)
    manifold_fig = create_manifold_plot(analyzer, trajectory_data)
    
    excel_path = generate_excel_data(analyzer, trajectory_data)
    
    current_analysis['trajectory_data'] = trajectory_data
    current_analysis['analyzer'] = analyzer
    current_analysis['file_info'] = uploaded_data.get('info')
    
    return summary_fig, trajectory_fig, manifold_fig, excel_path


def analyze_test_sample(cohort, sample_selection, show_gt):

    if cohort not in analyzers or analyzers[cohort] is None:
        return None, None, None, None
    
    analyzer = analyzers[cohort]
    
    try:
        sample_idx = int(sample_selection.split(':')[0].replace('Sample ', ''))
    except:
        return None, None, None, None
    
    window = analyzer.test_windows[sample_idx]
    
    result = analyzer.analyze_window(window)
    
    trajectory_data = {
        'results': [result],
        'timestamps': [0],
        'embeddings': result['embedding'].reshape(1, -1),
        'umap_positions': result['umap_pos'].reshape(1, -1)
    }
    
    
    hlac_pred = result['hlac_pred']
    gen_pred = result['genotype_pred']
    
    fig_summary, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    cmap = plt.cm.get_cmap('tab10', len(analyzer.class_names))
    colors = [cmap(i) for i in range(len(analyzer.class_names))]
    behavior_labels = [get_behavior_label(i) for i in range(len(analyzer.class_names))]
    axes[0].bar(range(len(analyzer.class_names)), result['hlac_probs'], color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(analyzer.class_names)))
    axes[0].set_xticklabels(behavior_labels, rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Behavior Probabilities')
    axes[0].set_ylim([0, 1])

    axes[0].get_children()[hlac_pred].set_edgecolor('gold')
    axes[0].get_children()[hlac_pred].set_linewidth(3)
    
    colors_gen = ['#1f77b4', '#d62728', '#2ca02c'][:len(result['genotype_probs'])]
    axes[1].bar(GENOTYPE_NAMES[:len(result['genotype_probs'])], result['genotype_probs'], 
               color=colors_gen, edgecolor='black')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Genotype Probabilities')
    axes[1].set_ylim([0, 1])

    axes[1].get_children()[gen_pred].set_edgecolor('gold')
    axes[1].get_children()[gen_pred].set_linewidth(3)
    
    plt.tight_layout()
    
    manifold_fig = create_manifold_plot(analyzer, trajectory_data)
    

    current_analysis['trajectory_data'] = trajectory_data
    current_analysis['analyzer'] = analyzer
    current_analysis['file_info'] = None
    
    return fig_summary, None, manifold_fig, None



def create_interface():
    with gr.Blocks(title="HONK: Behavioral Phenotyping Agent") as demo:
        gr.Markdown("""
        # HONK: Behavioral Phenotyping Agent
        
        Analyze mouse behavioral sequences using the GEESE pipeline.
        Upload a .mat file or select from test samples. Ask HONK questions about the results.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                cohort_dropdown = gr.Dropdown(
                    choices=list(COHORT_CONFIG.keys()),
                    value='CNTNAP',
                    label="Cohort"
                )
                
                gr.Markdown("### Option 1: Upload .mat File")
                file_upload = gr.File(label="Upload .mat file", file_types=[".mat"])
                file_status = gr.Textbox(label="File Status", lines=4, interactive=False)
                
                with gr.Row():
                    start_slider = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Start Time (sec)")
                    end_slider = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, label="End Time (sec)")
                
                show_gt_checkbox = gr.Checkbox(label="Show Ground Truth (if available)", value=True)
                
                analyze_upload_btn = gr.Button("Analyze Time Range", variant="primary")
                
                gr.Markdown("### Option 2: Test Sample")
                sample_dropdown = gr.Dropdown(
                    choices=get_sample_options('CNTNAP'),
                    label="Select Test Sample"
                )
                analyze_sample_btn = gr.Button("Analyze Sample", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("""
### Genotype
**WT**: Wild-type (no mutation)
**HET**: Heterozygous (one copy)
**HOM**: Homozygous (two copies)

### Semantic Behaviors
1: Idle
2: Sniff
3: Groom
4: Scrunched
5: Reared
6: Active crouch
7: Explore
8: Locomotion
9: Fast
                """)
        
        gr.Markdown("## Results")
        
        with gr.Row():
            summary_plot = gr.Plot(label="Summary: Behavior Duration & Genotype")
        
        with gr.Row():
            trajectory_plot = gr.Plot(label="Behavior Sequence")
        
        with gr.Row():
            manifold_plot = gr.Plot(label="Behavioral Manifold")
        
        with gr.Row():
            excel_download = gr.File(label="Download Results (Excel)", interactive=False)
        
        gr.Markdown("## Ask HONK")
        chatbot = gr.Chatbot(label="HONK", height=300, show_copy_button=True)
        msg_input = gr.Textbox(
            placeholder="e.g. What is the predicted genotype? / How often does grooming occur?",
            show_label=False
        )
        
        file_upload.change(
            fn=on_file_upload,
            inputs=[file_upload, cohort_dropdown],
            outputs=[file_status, start_slider, end_slider]
        )
        
        cohort_dropdown.change(
            fn=lambda c: gr.Dropdown(choices=get_sample_options(c)),
            inputs=[cohort_dropdown],
            outputs=[sample_dropdown]
        )
        
        analyze_upload_btn.click(
            fn=analyze_uploaded_file,
            inputs=[cohort_dropdown, start_slider, end_slider, show_gt_checkbox],
            outputs=[summary_plot, trajectory_plot, manifold_plot, excel_download]
        )
        
        analyze_sample_btn.click(
            fn=analyze_test_sample,
            inputs=[cohort_dropdown, sample_dropdown, show_gt_checkbox],
            outputs=[summary_plot, trajectory_plot, manifold_plot, excel_download]
        )
        
        msg_input.submit(
            fn=chatbot_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )
    
    return demo


if __name__ == "__main__":
    print("HONK: Behavioral Phenotyping Agent")
    print("Initializing analyzers")
    initialize_analyzers()
    
    print("Launching interface")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Default()
    )