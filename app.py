import os
import json
import pickle
import tempfile
import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from momentfm import MOMENTPipeline

try:
    from huggingface_hub import hf_hub_download, InferenceClient
    _HAS_HF = True
except ImportError:
    _HAS_HF = False



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 32
FPS = 50
STRIDE = WINDOW_SIZE // 2

OUTPUT_DIR = "results/class/hlac/all/2phase_normalized2"
PHASE1_MODEL = f"{OUTPUT_DIR}/phase1_behavior/best_model_hlac_slim.pth"
PHASE1_JSON = f"{OUTPUT_DIR}/phase1_behavior/results.json"
PHASE2_MODEL = f"{OUTPUT_DIR}/phase2_genotype/best_model_genotype_slim.pth"
PHASE2_JSON = f"{OUTPUT_DIR}/phase2_genotype/results.json"


EXAMPLES = {
    "CNTNAP - HOM": ("examples/CNTNAP_M4_20210309_0254_L.mat", "CNTNAP-HOM"),
    "CNTNAP - WT":  ("examples/CNTNAP_M5_20210309_0255_L.mat", "CNTNAP-WT"),
    "CHD8 - WT":    ("examples/CHD8_M1_20221119_0131_L.mat", "CHD8-WT"),
    "FMR1 - HET":   ("examples/FX_M2_20211205_0302_L.mat", "FMR1-HET"),
    "FMR1 - HET (2)": ("examples/FX_M8_20211204_0300_L.mat", "FMR1-HET"),
}

GENOTYPE_NAMES = ['WT', 'HET', 'HOM']  # ratgen 0/1/2

HLAC_SEMANTIC = {1: 'idle', 2: 'sniff', 3: 'groom', 4: 'scrunched', 5: 'reared',
                 6: 'active crouch', 7: 'explore', 8: 'locomotion', 9: 'fast'}


def behavior_display_names(behavior_names):
    """Map 'HLAC_<raw>' -> semantic name; fall back to raw label if unknown."""
    out = []
    for name in behavior_names:
        try:
            raw = int(name.split('_')[1])
            out.append(HLAC_SEMANTIC.get(raw, name))
        except (IndexError, ValueError):
            out.append(name)
    return out


BEHAVIOR_PALETTE = ['#6E8FD6', '#5FB49C', '#7C8AA0', '#E6B450', '#8C7AE6',
                    '#67B7DC', '#A06CD5', '#E08A5B', '#3F9E8E', '#D98BB0']
GENO_COLOR = {'WT': '#6E8FD6', 'HET': '#E6B450', 'HOM': '#5FB49C'}
COHORT_COLOR = '#5A6B85'
GT_GREY = '#C2C7D0'
INK = '#363B44'

HF_DATA_REPO = os.environ.get("HF_DATA_REPO", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def resolve(path):
    if os.path.exists(path):
        return path
    if _HAS_HF and HF_DATA_REPO:
        return hf_hub_download(repo_id=HF_DATA_REPO, filename=path, repo_type="dataset")
    raise FileNotFoundError(path)


def _softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class AllCohortAnalyzer:
    def __init__(self):
        self.behavior_model = None
        self.geno_model = None
        self.behavior_names = None
        self.geno_names = None
        self.cohort_names = None

    def load(self):
        with open(resolve(PHASE1_JSON)) as f:
            self.behavior_names = json.load(f)['class_names']
        self.raw_to_idx = {}
        for i, name in enumerate(self.behavior_names):
            try:
                self.raw_to_idx[int(name.split('_')[1])] = i
            except (IndexError, ValueError):
                pass
        with open(resolve(PHASE2_JSON)) as f:
            self.geno_names = json.load(f)['class_names']
        self.cohort_names = sorted(set(n.split('-')[0] for n in self.geno_names))

        n_beh = len(self.behavior_names)
        n_geno = len(self.geno_names)

        print(f"Loading behavior model ({n_beh}-class)...")
        self.behavior_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "classification", "n_channels": 69, "num_class": n_beh},
        )
        self.behavior_model.init()
        ckpt = torch.load(resolve(PHASE1_MODEL), map_location='cpu')
        self.behavior_model.load_state_dict(ckpt['model_state_dict'])
        self.behavior_model = self.behavior_model.to(DEVICE).eval()

        print(f"Loading genotype model ({n_geno}-class)...")
        self.geno_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "classification", "n_channels": 69, "num_class": n_geno},
        )
        self.geno_model.init()
        ckpt = torch.load(resolve(PHASE2_MODEL), map_location='cpu')
        self.geno_model.load_state_dict(ckpt['model_state_dict'])
        self.geno_model = self.geno_model.to(DEVICE).eval()
        print("Models ready.")

    def _classify_batch(self, windows):
        w = windows.astype(np.float32)
        for i in range(len(w)):
            m = w[i].mean()
            s = w[i].std() + 1e-8
            w[i] = (w[i] - m) / s

        beh_probs, geno_probs = [], []
        with torch.no_grad():
            for i in range(0, len(w), 64):
                batch = torch.FloatTensor(w[i:i+64]).to(DEVICE).permute(0, 2, 1)
                bl = self.behavior_model.classify(x_enc=batch).logits.cpu().numpy()
                gl = self.geno_model.classify(x_enc=batch).logits.cpu().numpy()
                beh_probs.append(_softmax(bl))
                geno_probs.append(_softmax(gl))
        return np.concatenate(beh_probs), np.concatenate(geno_probs)

    def analyze(self, pose, hlac_raw=None, true_geno=None):
        n_frames = pose.shape[0]
        starts = list(range(0, n_frames - WINDOW_SIZE + 1, STRIDE))
        if not starts:
            return None
        windows = np.stack([pose[s:s+WINDOW_SIZE] for s in starts])
        beh_probs, geno_probs = self._classify_batch(windows)
        timestamps = [(s + WINDOW_SIZE // 2) / FPS for s in starts]

        beh_true = None
        if hlac_raw is not None:
            beh_true = []
            for s in starts:
                mid = s + WINDOW_SIZE // 2
                if mid < len(hlac_raw):
                    beh_true.append(self.raw_to_idx.get(int(hlac_raw[mid]), -1))
                else:
                    beh_true.append(-1)
            beh_true = np.array(beh_true)

        return self._aggregate(beh_probs, geno_probs, timestamps,
                               beh_true=beh_true, true_geno=true_geno)

    def _aggregate(self, beh_probs, geno_probs, timestamps, beh_true=None, true_geno=None):
        geno_mean = geno_probs.mean(axis=0)
        top_geno = self.geno_names[geno_mean.argmax()]
        top_cohort, top_gen = top_geno.split('-')
        cohort_prob = {c: 0.0 for c in self.cohort_names}
        for name, p in zip(self.geno_names, geno_mean):
            cohort_prob[name.split('-')[0]] += p
        return {
            'timestamps': timestamps,
            'beh_pred': beh_probs.argmax(axis=1),
            'beh_probs': beh_probs,
            'beh_true': beh_true,
            'geno_pred': geno_probs.argmax(axis=1),
            'geno_probs': geno_probs,
            'geno_mean': geno_mean,
            'top_geno': top_geno,
            'top_cohort': top_cohort,
            'top_gen': top_gen,
            'cohort_prob': cohort_prob,
            'true_geno': true_geno,
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
        if pose.ndim == 3:
            pose = pose.reshape(pose.shape[0], -1)
        info = {'n_frames': pose.shape[0], 'n_features': pose.shape[1],
                'duration_sec': pose.shape[0] / FPS}
        if hasattr(sdannce, 'ratgen'):
            info['ratgen'] = int(sdannce.ratgen)
        if hasattr(sdannce, 'hlac') and sdannce.hlac is not None:
            hlac = sdannce.hlac.flatten() if sdannce.hlac.ndim > 1 else sdannce.hlac
            info['hlac'] = np.asarray(hlac)
        return pose, info
    except Exception as e:
        return None, f"Error loading file: {str(e)}"



def _style_axis(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#D6DAE1')
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, axis='both', color='#EDEFF3', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)


def _white_label(ax, x, y, text, **kw):
    ax.annotate(text, (x, y), color=INK, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#D6DAE1', alpha=0.9),
                **kw)


def make_plots(analyzer, res, true_label=None):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': '#D6DAE1'})
    n_beh = len(analyzer.behavior_names)
    beh_labels = behavior_display_names(analyzer.behavior_names)
    beh_colors = [BEHAVIOR_PALETTE[i % len(BEHAVIOR_PALETTE)] for i in range(n_beh)]

    fig = plt.figure(figsize=(14, 12), facecolor='white')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 0.55], hspace=0.45, wspace=0.25)

    # > Behavior distribution 
    ax = fig.add_subplot(gs[0, 0])
    counts = np.bincount(res['beh_pred'], minlength=n_beh)
    pct = counts / max(counts.sum(), 1) * 100
    y = np.arange(n_beh)
    has_beh_gt = res.get('beh_true') is not None and (res['beh_true'] >= 0).any()
    if has_beh_gt:
        gt = res['beh_true'][res['beh_true'] >= 0]
        gt_counts = np.bincount(gt, minlength=n_beh)
        gt_pct = gt_counts / max(gt_counts.sum(), 1) * 100
        ax.barh(y + 0.2, pct, 0.4, color=beh_colors, edgecolor='white', label='Predicted', zorder=3)
        ax.barh(y - 0.2, gt_pct, 0.4, color=beh_colors, edgecolor='white',
                alpha=0.45, hatch='///', label='Ground truth', zorder=3)
        ax.legend(loc='lower right', fontsize=8, frameon=False)
    else:
        ax.barh(y, pct, 0.62, color=beh_colors, edgecolor='white', zorder=3)
        for i, p in enumerate(pct):
            if p > 0:
                ax.text(p + 0.6, i, f'{p:.1f}%', va='center', fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(beh_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Time spent (%)')
    ax.set_title('Behavior distribution', fontweight='bold', loc='left')
    _style_axis(ax)

    # > Cohort x genotype 
    ax = fig.add_subplot(gs[0, 1])
    gx = np.arange(len(analyzer.geno_names))
    bar_colors = [GENO_COLOR.get(n.split('-')[1], COHORT_COLOR) for n in analyzer.geno_names]
    bars = ax.bar(gx, res['geno_mean'], color=bar_colors, edgecolor='white', zorder=3)
    true_geno = res.get('true_geno')
    if true_geno in analyzer.geno_names:
        gi = analyzer.geno_names.index(true_geno)
        bars[gi].set_edgecolor(INK)
        bars[gi].set_linewidth(2.2)
        bars[gi].set_hatch('///')
        ax.annotate('GT', (gi, res['geno_mean'][gi]), textcoords='offset points',
                    xytext=(0, 4), ha='center', fontsize=8, fontweight='bold', color=INK)
    ax.set_xticks(gx)
    ax.set_xticklabels(analyzer.geno_names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Mean probability')
    ax.set_title(f"Cohort x genotype  (pred: {res['top_geno']})", fontweight='bold', loc='left')
    _style_axis(ax)

    # > Behavior sequence
    ax = fig.add_subplot(gs[1, :])
    t = res['timestamps']
    pred = res['beh_pred']
    if has_beh_gt:
        gt_line = res['beh_true'].astype(float)
        gt_line[res['beh_true'] < 0] = np.nan
        ax.plot(t, gt_line, color=GT_GREY, lw=5, alpha=0.7, solid_capstyle='round',
                label='Ground truth', zorder=1)
    ax.plot(t, pred, color='#9AA4B2', lw=1.0, alpha=0.6, zorder=2)
    ax.scatter(t, pred, c=[beh_colors[p] for p in pred], s=42,
               edgecolors='white', linewidths=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(beh_labels, fontsize=8)
    ax.set_xlabel('Time (s)')
    ax.set_title('Behavior sequence', fontweight='bold', loc='left')
    if has_beh_gt:
        ax.legend(loc='upper right', fontsize=8, frameon=False)
    _style_axis(ax)

    # > Cohort prediction 
    ax = fig.add_subplot(gs[2, :])
    cohorts = list(res['cohort_prob'].keys())
    vals = [res['cohort_prob'][c] for c in cohorts]
    top_i = int(np.argmax(vals))
    cbar = ax.bar(cohorts, vals, color='#D8DDE6', edgecolor='white', zorder=3)
    cbar[top_i].set_color(COHORT_COLOR)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v*100:.0f}%', ha='center', fontsize=8, color=INK)
    ax.set_ylim(0, max(vals) * 1.2 + 0.02)
    ax.set_ylabel('Summed prob.')
    ax.set_title(f"Cohort  (pred: {res['top_cohort']})", fontweight='bold', loc='left')
    _style_axis(ax)

    return fig


def make_excel(analyzer, res):
    path = tempfile.mktemp(suffix='.xlsx')
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({
            'time_sec': res['timestamps'],
            'behavior': [analyzer.behavior_names[i] for i in res['beh_pred']],
            'genotype_7class': [analyzer.geno_names[i] for i in res['geno_pred']],
        }).to_excel(writer, sheet_name='Sequence', index=False)
        pd.DataFrame({
            'class': analyzer.geno_names,
            'mean_prob': res['geno_mean'],
        }).to_excel(writer, sheet_name='Genotype', index=False)
    return path



current = {'analyzer': None, 'res': None, 'true_label': None}


def build_context():
    res = current['res']
    analyzer = current['analyzer']
    if res is None:
        return "No analysis has been run yet."
    counts = np.bincount(res['beh_pred'], minlength=len(analyzer.behavior_names))
    pct = counts / counts.sum() * 100
    dt = WINDOW_SIZE / FPS * 0.5
    disp = behavior_display_names(analyzer.behavior_names)
    beh_lines = [f"  {disp[i]}: {pct[i]:.1f}% ({counts[i]*dt:.1f}s)"
                 for i in range(len(analyzer.behavior_names)) if counts[i] > 0]
    geno_lines = [f"  {analyzer.geno_names[i]}: {res['geno_mean'][i]*100:.1f}%"
                  for i in range(len(analyzer.geno_names))]
    cohort_lines = [f"  {c}: {p*100:.1f}%" for c, p in res['cohort_prob'].items()]
    duration = res['timestamps'][-1] - res['timestamps'][0] if len(res['timestamps']) > 1 else 0
    ctx = f"""Analysis results:
Duration: {duration:.1f}s, {len(res['beh_pred'])} windows (window {WINDOW_SIZE} frames at {FPS} fps).

Predicted cohort: {res['top_cohort']}
Predicted genotype: {res['top_gen']}  (full label: {res['top_geno']})
"""
    if current['true_label']:
        ctx += f"True label: {current['true_label']}\n"
    ctx += "\nBehavior distribution:\n" + "\n".join(beh_lines)
    ctx += "\n\nCohort probabilities:\n" + "\n".join(cohort_lines)
    ctx += "\n\n7-class (cohort x genotype) probabilities:\n" + "\n".join(geno_lines)
    return ctx


SYSTEM_PROMPT = """You are the analysis assistant for a mouse behavioral phenotyping demo (GEESE pipeline, MOMENT model on 3D pose data).
Answer the user's question using the analysis results below. Give the numbers from the data. Keep answers brief and factual.

{context}
"""


def call_llm(messages):
    if OPENAI_API_KEY:
        import requests
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                              json={"model": "gpt-4o", "messages": messages, "max_tokens": 512},
                              timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI error: {e}"
    if _HAS_HF:
        try:
            client = InferenceClient("Qwen/Qwen2.5-72B-Instruct")
            return client.chat_completion(messages=messages, max_tokens=512).choices[0].message.content
        except Exception as e:
            return f"HF Inference error: {e}"
    return "No LLM backend configured (set OPENAI_API_KEY or install huggingface_hub)."


def chat_respond(message, history):
    history = history or []
    if not message or not message.strip():
        return history
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=build_context())}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": message})
    resp = call_llm(messages)
    return history + [{"role": "user", "content": message},
                      {"role": "assistant", "content": resp}]



ANALYZER = None


def _run_mat(path, true_geno=None):
    pose, info = load_mat_file(path)
    if pose is None:
        return info, None, None
    hlac_raw = info.get('hlac') if isinstance(info, dict) else None
    res = ANALYZER.analyze(pose, hlac_raw=hlac_raw, true_geno=true_geno)
    if res is None:
        return "Not enough frames for one window.", None, None
    if true_geno is None and isinstance(info, dict) and 'ratgen' in info:
        rg = info['ratgen']
        true_geno = GENOTYPE_NAMES[rg] if rg < len(GENOTYPE_NAMES) else None
    current.update(analyzer=ANALYZER, res=res, true_label=true_geno)
    dur = info.get('duration_sec', 0) if isinstance(info, dict) else 0
    status = f"Duration {dur:.1f}s. Predicted: {res['top_cohort']} / {res['top_gen']}"
    if true_geno:
        status += f"  (true: {true_geno})"
    return status, make_plots(ANALYZER, res, true_geno), make_excel(ANALYZER, res)


def on_upload(file):
    if file is None:
        return "No file.", None, None
    return _run_mat(file.name)


def on_example(label):
    if not label or label not in EXAMPLES:
        return "Pick an example.", None, None
    path, true_geno = EXAMPLES[label]
    try:
        path = resolve(path)
    except Exception as e:
        return f"Could not load example: {e}", None, None
    return _run_mat(path, true_geno=true_geno)


def build_interface():
    with gr.Blocks(title="HONK: All-Cohort Phenotyping") as demo:
        gr.Markdown("# HONK - Behavioral Phenotyping\n"
                    "Upload a `.mat` file or pick a built-in example. "
                    "The model predicts behavior over time, cohort, and genotype.")

        with gr.Tab("Upload .mat"):
            up = gr.File(label=".mat file", file_types=[".mat"])
            up_btn = gr.Button("Analyze", variant="primary")

        with gr.Tab("Example"):
            ex_dd = gr.Dropdown(choices=list(EXAMPLES.keys()), label="Example recording")
            ex_btn = gr.Button("Analyze", variant="primary")

        status = gr.Textbox(label="Result", interactive=False)
        plot = gr.Plot(label="Analysis")
        excel = gr.File(label="Download (Excel)", interactive=False)

        gr.Markdown("## Ask")
        chatbot = gr.Chatbot(height=300)
        msg = gr.Textbox(placeholder="e.g. What is the predicted genotype? How often does each behavior occur?",
                         show_label=False)

        up_btn.click(on_upload, [up], [status, plot, excel])
        ex_btn.click(on_example, [ex_dd], [status, plot, excel])
        msg.submit(chat_respond, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])

    return demo


if __name__ == "__main__":
    print("Loading models...")
    ANALYZER = AllCohortAnalyzer()
    ANALYZER.load()
    demo = build_interface()
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate",
                           neutral_hue="slate", font=["system-ui", "sans-serif"])
    demo.launch(show_error=True, theme=theme, ssr_mode=False)