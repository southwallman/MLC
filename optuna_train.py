from model_config import ModelHyperParams
import torch.nn as nn
import time
import random
from tqdm import tqdm
import os
import torch
import torchvision.transforms as T
from typing import Optional, List, Dict, Any
from sklearn.metrics import roc_auc_score
import numpy as np
from tonguedx_MLC.ASL import AsymmetricLoss
from tonguedx_MLC.our_version7.版本7_resnet101_clip_Mod_ACFP import get_ablation_model
from gemini_MTI_HANet import MTI_HANet
from paper_ablation_MTI_HANet import get_paper_ablation_model
from tonguedx_MLC.datapro import load_pth_to_dataloader
import optuna
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from tonguedx_MLC.our_version7.消融backbone import get_backbone_ablation_model
# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 🚨 全局标签配置 (一处修改，全局生效) 🚨
# ==============================================================================
tonguedxlabel=['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
ITDDlabel=[
    'HealthyTongue', 'PeelingCoating', 'RedTongue', 'PurpleTongue', 'ChubbyTongue',
    'ThinTongue', 'RedDot', 'Crack', 'Toothmark', 'FurWhite', 'FurYellow',
    'FurBlack', 'SmoothCoating'
]
# ==============================================================================
# 🚨 全局标签配置 (一处修改，全局生效)
# ==============================================================================
GLOBAL_LABELS = tonguedxlabel
GLOBAL_NUM_CLASSES = len(GLOBAL_LABELS)

# 2. 锁死 Batch Size，防止 JSON 读取错乱导致显存溢出或报错
GLOBAL_BATCH_SIZE = 8

# 3. 锁死注意力头数等结构参数 (彻底杜绝 Size Mismatch)
FIXED_NUM_HEADS = 8
FIXED_FEATURE_DIM = 2048
FIXED_SEQ_LEN = 144


# ==============================================================================
# 🚨 参数清洗器 (兼容 Optuna 旧前缀) 🚨
# ==============================================================================
# ==============================================================================
# 🚨 参数清洗器 (兼容 Optuna 旧前缀) 🚨
# ==============================================================================
def _build_safe_config(params: dict, batch_size: int, device: str, num_classes: int = None) -> ModelHyperParams:
    """安全地从字典构建配置，自动过滤或重命名 Optuna 的旧前缀"""
    cleaned = {}
    raw_params = dict(params)

    # 提前把可能导致冲突的 key 踢掉
    raw_params.pop("num_classes", None)
    raw_params.pop("batch_size", None)
    raw_params.pop("device", None)
    raw_params.pop("learning_rate", None)

    for k, v in raw_params.items():
        if k.startswith("mhcsra_csra_"):
            cleaned[k.replace("mhcsra_csra_", "mhcsra_")] = v
        else:
            cleaned[k] = v

    # CSRA 模式特有的必须维度参数（如果 JSON 里没存）
    if "mhcsra_lam" in cleaned:
        if "mhcsra_input_dim" not in cleaned:
            cleaned["mhcsra_input_dim"] = 2048
        if "mhcsra_out_channel" not in cleaned:
            cleaned["mhcsra_out_channel"] = 2048

    # 动态切换分类数：Mode 5 传了就用传的，其他模式没传就用全局
    final_num_classes = num_classes if num_classes is not None else GLOBAL_NUM_CLASSES

    cfg = ModelHyperParams(
        num_classes=final_num_classes,
        batch_size=batch_size,
        device=device,
        **cleaned
    )
    # 兼容 acfp_in_channels
    cfg.acfp_in_channels = int(getattr(cfg, "mhcsra_csra_output_dim", getattr(cfg, "mhcsra_out_channel", 2048)))
    return cfg

# 选择模型
MODEL_VARIANT = "exp6"
MHCSRA_SEARCH_SPACE = "csra"
TRAIN_SEED = 3
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_BEST_PARAMS_JSON = os.path.join(
    _SCRIPT_DIR, "版本7_resnet101_clip_Mod_ACFP_matrix_best_params_exp6.json"
)


def get_experiment_tag() -> str:
    return f"{MODEL_VARIANT}_{str(MHCSRA_SEARCH_SPACE).lower()}"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


MAIN_NUM_EPOCHS = 40
model_save_path_total = (
    f"版本7_resnet101_clip_Mod_ACFP_matrix_best_model_{get_experiment_tag()}"
    f"_seed{TRAIN_SEED}_e{MAIN_NUM_EPOCHS}_{RUN_TAG}.pth"
)

json_path_total = f"版本7_resnet101_clip_Mod_ACFP_matrix_best_params_{get_experiment_tag()}_{RUN_TAG}.json"
json_path_latest = f"版本7_resnet101_clip_Mod_ACFP_matrix_best_params_{get_experiment_tag()}.json"


def get_model_save_path_for_variant(base_path: str, model_variant: str) -> str:
    if not base_path.endswith(".pth"):
        return f"{base_path}_{model_variant}.pth"
    return base_path.replace(".pth", f"_{model_variant}.pth")


OPTUNA_TRAIN_EPOCHS = 40
MAIN_N_TRIALS = 80
OPTIMIZER_WEIGHT_DECAY = 1e-4
NEW_MODULE_LR_MULTIPLIER = 10.0

PAPER_ABLATION_EPOCHS = 30
PAPER_MODE3_N_TRIALS = 15
PAPER_MODE3_OPTUNA_EPOCHS = 3

# PAPER_ABLATION_TRAIN_PTH = "../dataset/shezhenv3_train_data.pth"
# PAPER_ABLATION_TEST_PTH = "../dataset/shezhenv3_test_data.pth"
PAPER_ABLATION_TRAIN_PTH = "../dataset/train_data_juzhong.pth"
PAPER_ABLATION_TEST_PTH = "../dataset/test_data_juzhong.pth"


# def build_differential_adam(
#         model: nn.Module,
#         base_lr: float,
#         *,
#         weight_decay: float = OPTIMIZER_WEIGHT_DECAY,
#         new_module_lr_multiplier: float = NEW_MODULE_LR_MULTIPLIER,
# ) -> torch.optim.Optimizer:
#     if hasattr(model, "get_optimizer_groups") and callable(model.get_optimizer_groups):
#         param_groups = model.get_optimizer_groups(
#             base_lr=base_lr, new_module_lr_multiplier=new_module_lr_multiplier
#         )
#     else:
#         backbone_params = []
#         new_module_params = []
#         matrix_params = []  # 👈 新增：专门给标签相关性矩阵准备的池子
#
#         for name, param in model.named_parameters():
#             if not param.requires_grad:
#                 continue
#
#             # 🚨 拦截标签相关性矩阵 M
#             if "correlation_matrix" in name:
#                 matrix_params.append(param)
#             elif "cnnbackbone" in name:
#                 backbone_params.append(param)
#             else:
#                 new_module_params.append(param)
#
#         if not backbone_params:
#             return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
#
#         # 组装基础参数组
#         param_groups = [
#             {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
#             {"params": new_module_params, "lr": base_lr * new_module_lr_multiplier, "weight_decay": weight_decay},
#         ]
#
#         # 🚨 为矩阵 M 开启“皇室特权”
#         if matrix_params:
#             # 10倍放大，并设置最高 1e-3 的安全上限防止梯度爆炸
#             matrix_lr = min(base_lr * 10.0, 1e-3)
#             param_groups.append({
#                 "params": matrix_params,
#                 "lr": matrix_lr,
#                 "weight_decay": 0.0  # 绝对不能有权重衰减，让非对角线自由生长！
#             })
#             print(f"🔥 [优化器拦截] 已为 correlation_matrix 开启特权 -> LR: {matrix_lr:.6f}, Weight Decay: 0.0",
#                   flush=True)
#
#     # 顺手把 Adam 升级成 AdamW，对多标签分类的正则化效果更好
#     return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
def build_differential_adam(
        model: nn.Module,
        base_lr: float,
        *,
        weight_decay: float = OPTIMIZER_WEIGHT_DECAY,
        new_module_lr_multiplier: float = NEW_MODULE_LR_MULTIPLIER,
) -> torch.optim.Optimizer:
    if hasattr(model, "get_optimizer_groups") and callable(model.get_optimizer_groups):
        param_groups = model.get_optimizer_groups(
            base_lr=base_lr, new_module_lr_multiplier=new_module_lr_multiplier
        )
    else:
        backbone_params = []
        new_module_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "cnnbackbone" in name:
                backbone_params.append(param)
            else:
                new_module_params.append(param)
        if not backbone_params:
            return torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        param_groups = [
            {"params": backbone_params, "lr": base_lr},
            {"params": new_module_params, "lr": base_lr * new_module_lr_multiplier},
        ]
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def get_train_model(model_variant: str, label, config):
    v = str(model_variant).lower()
    if v == "mti_hanet":
        return MTI_HANet(label, config)
    if v.startswith("model_") or v in {"ours"}:
        return get_paper_ablation_model(v, label, config)
    return get_ablation_model(v, label, config)


def train_paper_ablation_suite_from_json(
        json_path: str,
        *,
        variants: Optional[List[str]] = None,
        num_epochs: int = PAPER_ABLATION_EPOCHS,
) -> List[Dict[str, Any]]:
    if variants is None:
        variants = [f"model_{i}" for i in range(1, 10)]

    with open(json_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    learning_rate = float(best_params.get("learning_rate", 1e-4))
    batch_size = int(best_params.get("batch_size", 16))
    best_params = dict(best_params)
    best_params.pop("learning_rate", None)
    best_params.pop("batch_size", None)

    # 🚨 使用安全构建
    config = _build_safe_config(best_params, batch_size, "cuda" if torch.cuda.is_available() else "cpu")

    train_loader = load_pth_to_dataloader(
        PAPER_ABLATION_TRAIN_PTH, batch_size=config.batch_size, shuffle=True
    )
    test_loader = load_pth_to_dataloader(
        PAPER_ABLATION_TEST_PTH, batch_size=config.batch_size, shuffle=False
    )

    results: List[Dict[str, Any]] = []
    for variant in variants:
        suite_tag = f"paper_{variant}_{str(MHCSRA_SEARCH_SPACE).lower()}"
        model_save_path = f"paper_ablation_best_model_{suite_tag}_seed{TRAIN_SEED}_e{num_epochs}_{RUN_TAG}.pth"

        print("\n" + "=" * 70, flush=True)
        print(f"[Paper-Ablation] 开始训练: {variant}", flush=True)
        model = get_paper_ablation_model(variant, GLOBAL_LABELS, config)
        _, best_val_auc, best_epoch = train_full_model(
            model=model, train_loader=train_loader, test_loader=test_loader,
            config=config, learning_rate=learning_rate, num_epochs=num_epochs,
            model_save_path=model_save_path,
        )
        results.append({
            "variant": variant, "best_val_auc": float(best_val_auc),
            "best_epoch": int(best_epoch), "model_save_path": model_save_path,
        })
    return results


def run_paper_mode3_optuna_then_train(
        *,
        variants: Optional[List[str]] = None,
        n_trials: int = PAPER_MODE3_N_TRIALS,
        optuna_epochs: int = PAPER_MODE3_OPTUNA_EPOCHS,
        full_epochs: int = PAPER_ABLATION_EPOCHS,
) -> List[Dict[str, Any]]:
    if variants is None:
        variants = [f"model_{i}" for i in range(1, 11)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_space = str(MHCSRA_SEARCH_SPACE).lower()

    def _build_config_from_trial(trial: optuna.Trial) -> ModelHyperParams:
        if search_space == "csra":
            mhcsra_kwargs = dict(
                mhcsra_input_dim=FIXED_FEATURE_DIM,
                mhcsra_out_channel=FIXED_FEATURE_DIM,
                mhcsra_num_heads=FIXED_NUM_HEADS, # 🚨 锁死！不搜了！
                mhcsra_lam=trial.suggest_float("mhcsra_csra_lam", 0.0, 1.0, step=0.1),
                mhcsra_fusion_method=trial.suggest_categorical("mhcsra_csra_fusion_method", ["concat", "sum", "attention"]),
                mhcsra_use_residual=trial.suggest_categorical("mhcsra_csra_use_residual", [True, False]),
            )
        else:
            mhcsra_kwargs = dict(
                mhcsra_num_heads=FIXED_NUM_HEADS, # 🚨 锁死！
                mhcsra_feature_dim=FIXED_FEATURE_DIM, # 🚨 锁死！
                mhcsra_out_channel=FIXED_FEATURE_DIM,
                mhcsra_dropout=trial.suggest_float("mhcsra_dropout", 0.0, 0.5, step=0.1),
                mhcsra_shallow_layers=trial.suggest_int("mhcsra_shallow_layers", 1, 3), # 层数虽然改变结构，但通常只在内部循环，如果要极致安全，这里也可以写死为 1或2
                mhcsra_use_shallow=True,
            )

        cfg = ModelHyperParams(
            num_classes=GLOBAL_NUM_CLASSES,
            batch_size=GLOBAL_BATCH_SIZE, # 🚨 锁死！
            device=device,
            **mhcsra_kwargs,
            acfp_mode=trial.suggest_categorical("acfp_mode", ['adaptive', 'keep']),
            acfp_target_seq_len=FIXED_SEQ_LEN, # 🚨 锁死序列长度！不要搜了！
            acfp_use_residual=trial.suggest_categorical("acfp_use_residual", [True, False]),
            mmaef_num_heads=FIXED_NUM_HEADS, # 🚨 锁死！
            mmaef_ffn_hidden_ratio=trial.suggest_float("mmaef_ffn_hidden_ratio", 2.0, 8.0, step=1.0),
            mmaef_use_dropout=trial.suggest_categorical("mmaef_use_dropout", [True, False]),
            mmaef_dropout_rate=trial.suggest_float("mmaef_dropout_rate", 0.0, 0.3, step=0.05),
            hybrid_intermediate_dim=256, # 🚨 锁死中间维度！
            hybrid_dropout_rate=trial.suggest_float("hybrid_dropout_rate", 0.0, 0.3, step=0.05),
            hybrid_activation=trial.suggest_categorical("hybrid_activation", ['relu', 'gelu', 'leaky_relu']),
            hybrid_alpha_init=trial.suggest_float("hybrid_alpha_init", 0.2, 0.8, step=0.05),
            max_iterations=trial.suggest_int("max_iterations", 1, 10),
        )
        cfg.acfp_in_channels = int(getattr(cfg, "mhcsra_csra_output_dim", getattr(cfg, "mhcsra_out_channel", 2048)))
        return cfg

    results: List[Dict[str, Any]] = []
    for variant in variants:
        print("\n" + "=" * 80, flush=True)
        print(
            f"[Mode3] 开始 {variant}: Optuna({n_trials} trials x {optuna_epochs} epochs) -> FullTrain({full_epochs} epochs)",
            flush=True)

        trial_dir = f"optuna_trial_records_paper_{variant}_{RUN_TAG}"
        os.makedirs(trial_dir, exist_ok=True)

        def objective_variant(trial: optuna.Trial) -> float:
            config = _build_config_from_trial(trial)
            try:
                train_loader = load_pth_to_dataloader(PAPER_ABLATION_TRAIN_PTH, batch_size=config.batch_size,
                                                      shuffle=True)
                test_loader = load_pth_to_dataloader(PAPER_ABLATION_TEST_PTH, batch_size=config.batch_size,
                                                     shuffle=False)
                model = get_paper_ablation_model(variant, GLOBAL_LABELS, config)
                lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)

                train_model_for_optuna(model=model, train_loader=train_loader, device=config.device,
                                       num_epochs=optuna_epochs, learning_rate=lr, use_asl=True, verbose=False)

                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for images, labels_batch in test_loader:
                        images = images.to(config.device)
                        outputs = model(images)
                        all_preds.append(torch.sigmoid(outputs).cpu())
                        all_labels.append(labels_batch.cpu())

                all_preds = torch.cat(all_preds, dim=0).numpy()
                all_labels = torch.cat(all_labels, dim=0).numpy()
                mean_auc = compute_mean_auc(all_preds, all_labels)

                trial_model_path = os.path.join(trial_dir, f"trial_{trial.number:04d}_auc_{mean_auc:.6f}.pth")
                torch.save({
                    "trial_number": trial.number, "trial_value_auc": mean_auc,
                    "model_state_dict": model.state_dict(), "config": config.__dict__,
                    "trial_params": dict(trial.params), "variant": variant,
                }, trial_model_path)

                trial_rows = compute_trial_label_metrics(all_preds, all_labels, label_names=GLOBAL_LABELS,
                                                         threshold=0.5)
                trial_log_csv = os.path.join(trial_dir, f"optuna_trial_metrics_{variant}_{RUN_TAG}.csv")
                append_trial_metrics_csv(trial_log_csv, trial_no=trial.number, trial_auc=mean_auc, rows=trial_rows)
                return mean_auc
            except Exception as e:
                print(f"[Mode3-{variant}] trial 失败: {e}", flush=True)
                return float("-inf")

        study = optuna.create_study(direction="maximize", study_name=f"paper_{variant}_optuna_{RUN_TAG}",
                                    sampler=optuna.samplers.TPESampler())
        study.optimize(objective_variant, n_trials=n_trials, show_progress_bar=True)

        best_params_all = dict(study.best_params)
        best_learning_rate = float(best_params_all.get("learning_rate", 1e-4))
        # 原代码：best_batch_size = int(best_params_all.get("batch_size", 16))
        # 替换为 👇
        best_batch_size = GLOBAL_BATCH_SIZE

        json_path_variant = f"paper_ablation_best_params_{variant}_{RUN_TAG}.json"
        json_path_variant_latest = f"paper_ablation_best_params_{variant}.json"
        with open(json_path_variant, "w", encoding="utf-8") as f:
            json.dump(best_params_all, f, indent=4)
        with open(json_path_variant_latest, "w", encoding="utf-8") as f:
            json.dump(best_params_all, f, indent=4)

        best_params_for_cfg = dict(best_params_all)
        best_params_for_cfg.pop("learning_rate", None)
        best_params_for_cfg.pop("batch_size", None)

        # 🚨 使用安全构建，过滤旧前缀
        best_config = _build_safe_config(best_params_for_cfg, best_batch_size, device)

        model_save_path = f"paper_ablation_best_model_paper_{variant}_{str(MHCSRA_SEARCH_SPACE).lower()}_seed{TRAIN_SEED}_e{full_epochs}_{RUN_TAG}.pth"
        train_loader = load_pth_to_dataloader(PAPER_ABLATION_TRAIN_PTH, batch_size=best_config.batch_size, shuffle=True)
        test_loader = load_pth_to_dataloader(PAPER_ABLATION_TEST_PTH, batch_size=best_config.batch_size, shuffle=False)
        best_model = get_paper_ablation_model(variant, GLOBAL_LABELS, best_config)
        _, best_val_auc, best_epoch = train_full_model(
            model=best_model, train_loader=train_loader, test_loader=test_loader,
            config=best_config, learning_rate=best_learning_rate, num_epochs=full_epochs,
            model_save_path=model_save_path, loss_kind="bce" if variant == "model_10" else "asl",
        )

        results.append({
            "variant": variant, "best_trial_auc": float(study.best_value),
            "best_val_auc": float(best_val_auc), "best_epoch": int(best_epoch),
            "model_save_path": model_save_path, "best_params_json": json_path_variant,
        })
    return results


def run_mode4_ours_optuna_then_train(
        n_trials: int = PAPER_MODE3_N_TRIALS,
        optuna_epochs: int = PAPER_MODE3_OPTUNA_EPOCHS,
        full_epochs: int = PAPER_ABLATION_EPOCHS,
) -> List[Dict[str, Any]]:
    """
    更新后的模式 4：专门针对 ours (MTI_HANet) 使用 ASL 损失进行搜参和训练。
    支持手动切换数据集，保存格式完全对齐 Mode 3。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_space = str(MHCSRA_SEARCH_SPACE).lower()

    # 🚨 【在这里手动切换数据集】想跑哪个就把另一个注释掉
    DATASETS_CONFIG = {
        "TongueDx": {
            "train_pth": "../dataset/train_data_juzhong.pth",
            "test_pth": "../dataset/test_data_juzhong.pth",
            "labels": tonguedxlabel
        },
        # "ITDD": {
        #     "train_pth": "../dataset/shezhenv3_train_data.pth",
        #     "test_pth": "../dataset/shezhenv3_test_data.pth",
        #     "labels": ITDDlabel
        # }
    }

    results: List[Dict[str, Any]] = []

    for dataset_name, ds_info in DATASETS_CONFIG.items():
        current_train_pth = ds_info["train_pth"]
        current_test_pth = ds_info["test_pth"]
        current_labels = ds_info["labels"]
        curr_n_cls = len(current_labels)
        variant = "ours"

        print("\n" + "=" * 80, flush=True)
        print(
            f"[Mode 4] 开始 {variant} 针对 {dataset_name} 数据集: Optuna({n_trials} trials) -> FullTrain({full_epochs} epochs)",
            flush=True)

        # 【保存格式 1：Trial 文件夹】
        trial_dir = f"optuna_trial_records_mode4_{variant}_{dataset_name}_{RUN_TAG}"
        os.makedirs(trial_dir, exist_ok=True)

        def objective_ours(trial: optuna.Trial) -> float:
            trial_params = {
                "acfp_mode": trial.suggest_categorical("acfp_mode", ['adaptive', 'keep']),
                "acfp_use_residual": trial.suggest_categorical("acfp_use_residual", [True, False]),
                "mmaef_ffn_hidden_ratio": trial.suggest_float("mmaef_ffn_hidden_ratio", 2.0, 8.0, step=1.0),
                "mmaef_use_dropout": trial.suggest_categorical("mmaef_use_dropout", [True, False]),
                "mmaef_dropout_rate": trial.suggest_float("mmaef_dropout_rate", 0.0, 0.3, step=0.05),
                "hybrid_dropout_rate": trial.suggest_float("hybrid_dropout_rate", 0.0, 0.3, step=0.05),
                "hybrid_activation": trial.suggest_categorical("hybrid_activation", ['relu', 'gelu', 'leaky_relu']),
                "hybrid_alpha_init": trial.suggest_float("hybrid_alpha_init", 0.2, 0.8, step=0.05),
                "max_iterations": trial.suggest_int("max_iterations", 1, 10),
            }

            if search_space == "csra":
                trial_params.update({
                    "mhcsra_csra_lam": trial.suggest_float("mhcsra_csra_lam", 0.0, 1.0, step=0.1),
                    "mhcsra_csra_fusion_method": trial.suggest_categorical("mhcsra_csra_fusion_method",
                                                                           ["concat", "sum", "attention"]),
                    "mhcsra_csra_use_residual": trial.suggest_categorical("mhcsra_csra_use_residual", [True, False]),
                    "mhcsra_input_dim": FIXED_FEATURE_DIM,
                    "mhcsra_out_channel": FIXED_FEATURE_DIM,
                    "mhcsra_num_heads": FIXED_NUM_HEADS,
                })
            else:
                trial_params.update({
                    "mhcsra_dropout": trial.suggest_float("mhcsra_dropout", 0.0, 0.5, step=0.1),
                    "mhcsra_shallow_layers": trial.suggest_int("mhcsra_shallow_layers", 1, 3),
                    "mhcsra_num_heads": FIXED_NUM_HEADS,
                    "mhcsra_feature_dim": FIXED_FEATURE_DIM,
                    "mhcsra_out_channel": FIXED_FEATURE_DIM,
                    "mhcsra_use_shallow": True,
                })

            trial_params.update({
                "acfp_target_seq_len": FIXED_SEQ_LEN,
                "mmaef_num_heads": FIXED_NUM_HEADS,
                "hybrid_intermediate_dim": 256,
            })

            config = _build_safe_config(trial_params, GLOBAL_BATCH_SIZE, device, num_classes=curr_n_cls)

            try:
                train_loader = load_pth_to_dataloader(current_train_pth, batch_size=config.batch_size, shuffle=True)
                test_loader = load_pth_to_dataloader(current_test_pth, batch_size=config.batch_size, shuffle=False)

                model = get_paper_ablation_model(variant, current_labels, config)
                lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)

                # 使用 Optuna 短训练
                train_model_for_optuna(model=model, train_loader=train_loader, device=config.device,
                                       num_epochs=optuna_epochs, learning_rate=lr, use_asl=True, verbose=False)

                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for images, labels_batch in test_loader:
                        outputs = model(images.to(config.device))
                        all_preds.append(torch.sigmoid(outputs).cpu())
                        all_labels.append(labels_batch.cpu())

                all_preds = torch.cat(all_preds, dim=0).numpy()
                all_labels = torch.cat(all_labels, dim=0).numpy()
                mean_auc = compute_mean_auc(all_preds, all_labels)

                # 【保存格式 2：每个 Trial 的模型和参数记录】
                trial_model_path = os.path.join(trial_dir, f"trial_{trial.number:04d}_auc_{mean_auc:.6f}.pth")
                torch.save({
                    "trial_number": trial.number, "trial_value_auc": mean_auc,
                    "model_state_dict": model.state_dict(), "config": config.__dict__,
                    "trial_params": dict(trial.params), "variant": variant, "dataset": dataset_name
                }, trial_model_path)

                # 【保存格式 3：CSV 日志追加】
                trial_rows = compute_trial_label_metrics(all_preds, all_labels, label_names=current_labels,
                                                         threshold=0.5)
                trial_log_csv = os.path.join(trial_dir,
                                             f"optuna_trial_metrics_mode4_{variant}_{dataset_name}_{RUN_TAG}.csv")
                append_trial_metrics_csv(trial_log_csv, trial_no=trial.number, trial_auc=mean_auc, rows=trial_rows)

                return mean_auc
            except Exception as e:
                print(f"[Mode4-{dataset_name}-{variant}] trial 失败: {e}", flush=True)
                traceback.print_exc()
                return float("-inf")

        study = optuna.create_study(direction="maximize", study_name=f"mode4_{variant}_{dataset_name}_optuna_{RUN_TAG}",
                                    sampler=optuna.samplers.TPESampler())
        study.optimize(objective_ours, n_trials=n_trials, show_progress_bar=True)

        best_params_all = dict(study.best_params)
        best_learning_rate = float(best_params_all.get("learning_rate", 1e-4))

        # 【保存格式 4：输出 Best JSON 参数文件】
        json_path_variant = f"paper_ablation_best_params_mode4_{variant}_{dataset_name}_{RUN_TAG}.json"
        with open(json_path_variant, "w", encoding="utf-8") as f:
            json.dump(best_params_all, f, indent=4)

        best_params_for_cfg = dict(best_params_all)
        best_params_for_cfg.pop("learning_rate", None)

        # 为了防漏补全固定参数
        if search_space == "csra":
            best_params_for_cfg.update({"mhcsra_input_dim": FIXED_FEATURE_DIM, "mhcsra_out_channel": FIXED_FEATURE_DIM,
                                        "mhcsra_num_heads": FIXED_NUM_HEADS})
        else:
            best_params_for_cfg.update({"mhcsra_num_heads": FIXED_NUM_HEADS, "mhcsra_feature_dim": FIXED_FEATURE_DIM,
                                        "mhcsra_out_channel": FIXED_FEATURE_DIM, "mhcsra_use_shallow": True})
        best_params_for_cfg.update(
            {"acfp_target_seq_len": FIXED_SEQ_LEN, "mmaef_num_heads": FIXED_NUM_HEADS, "hybrid_intermediate_dim": 256})

        best_config = _build_safe_config(best_params_for_cfg, GLOBAL_BATCH_SIZE, device, num_classes=curr_n_cls)

        # 【保存格式 5：最终的 Best Model】
        model_save_path = f"paper_ablation_best_model_mode4_{variant}_{dataset_name}_seed{TRAIN_SEED}_e{full_epochs}_{RUN_TAG}.pth"
        train_loader = load_pth_to_dataloader(current_train_pth, batch_size=best_config.batch_size, shuffle=True)
        test_loader = load_pth_to_dataloader(current_test_pth, batch_size=best_config.batch_size, shuffle=False)
        best_model = get_paper_ablation_model(variant, current_labels, best_config)

        # 统一使用 ASL 损失训练
        _, best_val_auc, best_epoch = train_full_model(
            model=best_model, train_loader=train_loader, test_loader=test_loader,
            config=best_config, learning_rate=best_learning_rate, num_epochs=full_epochs,
            model_save_path=model_save_path, loss_kind="asl"
        )

        results.append({
            "dataset": dataset_name, "variant": variant, "best_trial_auc": float(study.best_value),
            "best_val_auc": float(best_val_auc), "best_epoch": int(best_epoch),
            "model_save_path": model_save_path, "best_params_json": json_path_variant,
        })

    return results


import traceback  # 引入追踪库，报错不迷路


def run_backbone_optuna_then_train(
        *,
        n_trials: int = PAPER_MODE3_N_TRIALS,
        optuna_epochs: int = PAPER_MODE3_OPTUNA_EPOCHS,
        full_epochs: int = PAPER_ABLATION_EPOCHS,
) -> List[Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_space = str(MHCSRA_SEARCH_SPACE).lower()  # 提取搜索空间
    # backbones = ["resnet101", "resnet50", "resnet34", "densenet121", "vgg16", "googlenet"]
    backbones = ["resnet101"]
    all_results: List[Dict[str, Any]] = []

    DATASETS_CONFIG = {
        "TongueDx": {
            "train_pth": "../dataset/train_data_juzhong.pth",
            "test_pth": "../dataset/test_data_juzhong.pth",
            "labels": tonguedxlabel
        },
        "ITDD": {
            "train_pth": "../dataset/shezhenv3_train_data.pth",
            "test_pth": "../dataset/shezhenv3_test_data.pth",
            "labels": ITDDlabel
        }
    }

    print("\n" + "=" * 80, flush=True)
    print(f"[Mode 5] 开始双数据集骨干网络消融：先 Optuna 搜索 ({n_trials} trials)，后完整训练 ({full_epochs} epochs)",
          flush=True)

    for dataset_name, ds_info in DATASETS_CONFIG.items():
        current_train_pth = ds_info["train_pth"]
        current_test_pth = ds_info["test_pth"]
        current_labels = ds_info["labels"]
        curr_n_cls = len(current_labels)

        print(f"\n🚀 >>> 切换至数据集: {dataset_name} (类别: {curr_n_cls}) <<<")

        # 🚀 提取公共的配置构造器，确保 trial 和 best_params 都能打上固定补丁
        def _get_full_params_dict(base_params: dict) -> dict:
            full_params = dict(base_params)
            # 🚨 强行注入所有锁死的参数，绝不侧漏
            full_params.update({
                "acfp_target_seq_len": FIXED_SEQ_LEN,
                "mmaef_num_heads": FIXED_NUM_HEADS,
                "hybrid_intermediate_dim": 256,
            })
            if search_space == "csra":
                full_params.update({
                    "mhcsra_input_dim": FIXED_FEATURE_DIM,
                    "mhcsra_out_channel": FIXED_FEATURE_DIM,
                    "mhcsra_num_heads": FIXED_NUM_HEADS,
                })
            else:
                full_params.update({
                    "mhcsra_num_heads": FIXED_NUM_HEADS,
                    "mhcsra_feature_dim": FIXED_FEATURE_DIM,
                    "mhcsra_out_channel": FIXED_FEATURE_DIM,
                    "mhcsra_use_shallow": True,
                })
            return full_params

        for variant in backbones:
            print(f"\n{'-' * 20} Backbone: {variant} {'-' * 20}")
            trial_dir = f"optuna_trial_records_backbone_{dataset_name}_{variant}_{RUN_TAG}"
            os.makedirs(trial_dir, exist_ok=True)

            def objective_backbone(trial: optuna.Trial) -> float:
                # 1. 构建搜参字典
                trial_params = {
                    "acfp_mode": trial.suggest_categorical("acfp_mode", ['adaptive', 'keep']),
                    "acfp_use_residual": trial.suggest_categorical("acfp_use_residual", [True, False]),
                    "mmaef_ffn_hidden_ratio": trial.suggest_float("mmaef_ffn_hidden_ratio", 2.0, 8.0, step=1.0),
                    "mmaef_use_dropout": trial.suggest_categorical("mmaef_use_dropout", [True, False]),
                    "mmaef_dropout_rate": trial.suggest_float("mmaef_dropout_rate", 0.0, 0.3, step=0.05),
                    "hybrid_dropout_rate": trial.suggest_float("hybrid_dropout_rate", 0.0, 0.3, step=0.05),
                    "hybrid_activation": trial.suggest_categorical("hybrid_activation", ['relu', 'gelu', 'leaky_relu']),
                    "hybrid_alpha_init": trial.suggest_float("hybrid_alpha_init", 0.2, 0.8, step=0.05),
                    "max_iterations": trial.suggest_int("max_iterations", 1, 10),
                }

                if search_space == "csra":
                    trial_params.update({
                        "mhcsra_csra_lam": trial.suggest_float("mhcsra_csra_lam", 0.0, 1.0, step=0.1),
                        "mhcsra_csra_fusion_method": trial.suggest_categorical("mhcsra_csra_fusion_method",
                                                                               ["concat", "sum", "attention"]),
                        "mhcsra_csra_use_residual": trial.suggest_categorical("mhcsra_csra_use_residual",
                                                                              [True, False]),
                    })
                else:
                    trial_params.update({
                        "mhcsra_dropout": trial.suggest_float("mhcsra_dropout", 0.0, 0.5, step=0.1),
                        "mhcsra_shallow_layers": trial.suggest_int("mhcsra_shallow_layers", 1, 3),
                    })

                # 2. 注入固定参数并构造 config
                full_trial_params = _get_full_params_dict(trial_params)
                config = _build_safe_config(full_trial_params, GLOBAL_BATCH_SIZE, device, num_classes=curr_n_cls)

                try:
                    train_loader = load_pth_to_dataloader(current_train_pth, batch_size=config.batch_size, shuffle=True)
                    test_loader = load_pth_to_dataloader(current_test_pth, batch_size=config.batch_size, shuffle=False)

                    model = get_backbone_ablation_model(variant, current_labels, config)
                    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)

                    train_model_for_optuna(model, train_loader, device, optuna_epochs, lr, verbose=False)

                    model.eval()
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for img, label in test_loader:
                            out = model(img.to(device))
                            all_preds.append(torch.sigmoid(out).cpu())
                            all_labels.append(label.cpu())

                    all_preds_np = torch.cat(all_preds).numpy()
                    all_labels_np = torch.cat(all_labels).numpy()
                    mean_auc = compute_mean_auc(all_preds_np, all_labels_np)

                    # 🚨 恢复：保存中间模型与日志，防断电白给
                    trial_model_path = os.path.join(trial_dir, f"trial_{trial.number:04d}_auc_{mean_auc:.6f}.pth")
                    torch.save({
                        "trial_number": trial.number, "trial_value_auc": mean_auc,
                        "model_state_dict": model.state_dict(), "config": config.__dict__,
                        "trial_params": dict(trial.params), "variant": variant, "dataset": dataset_name
                    }, trial_model_path)

                    trial_rows = compute_trial_label_metrics(all_preds_np, all_labels_np, label_names=current_labels,
                                                             threshold=0.5)
                    trial_log_csv = os.path.join(trial_dir,
                                                 f"optuna_trial_metrics_backbone_{dataset_name}_{variant}_{RUN_TAG}.csv")
                    append_trial_metrics_csv(trial_log_csv, trial_no=trial.number, trial_auc=mean_auc, rows=trial_rows)

                    return mean_auc
                except Exception as e:
                    print(f"[Mode5-{dataset_name}-{variant}] trial {trial.number} 失败！错误信息：")
                    traceback.print_exc()  # 🚨 打印完整堆栈，一旦出错立刻知道是哪一行
                    return float("-inf")

            # 运行 Optuna
            study_name = f"backbone_{dataset_name}_{variant}_optuna_{RUN_TAG}"
            study = optuna.create_study(direction="maximize", study_name=study_name,
                                        sampler=optuna.samplers.TPESampler())
            study.optimize(objective_backbone, n_trials=n_trials, show_progress_bar=True)

            # 提取最佳参数并保存 JSON
            best_params_all = dict(study.best_params)
            best_lr = best_params_all.pop("learning_rate", 1e-4)

            json_path_variant = f"paper_backbone_best_params_{dataset_name}_{variant}_{RUN_TAG}.json"
            with open(json_path_variant, "w", encoding="utf-8") as f:
                json.dump(study.best_params, f, indent=4)  # 存入原始搜到的超参

            # 🚨 关键：最终训练前，必须为 best_params 补充丢失的固定参数
            full_best_params = _get_full_params_dict(best_params_all)
            best_config = _build_safe_config(full_best_params, GLOBAL_BATCH_SIZE, device, num_classes=curr_n_cls)

            # 最终完整训练
            model_save_path = f"paper_ablation_best_model_backbone_{dataset_name}_{variant}_seed{TRAIN_SEED}_e{full_epochs}_{RUN_TAG}.pth"
            train_loader = load_pth_to_dataloader(current_train_pth, batch_size=best_config.batch_size, shuffle=True)
            test_loader = load_pth_to_dataloader(current_test_pth, batch_size=best_config.batch_size, shuffle=False)

            final_model = get_backbone_ablation_model(variant, current_labels, best_config)
            _, best_val_auc, best_epoch = train_full_model(
                model=final_model,
                train_loader=train_loader,
                test_loader=test_loader,
                config=best_config,
                learning_rate=best_lr,
                num_epochs=full_epochs,
                model_save_path=model_save_path,
                loss_kind="asl"
            )

            all_results.append({
                "dataset": dataset_name, "backbone": variant, "best_trial_auc": float(study.best_value),
                "best_val_auc": float(best_val_auc), "best_epoch": int(best_epoch),
                "model_save_path": model_save_path, "best_params_json": json_path_variant
            })

    print("\n" + "=" * 70, flush=True)
    print("[Mode 5] 所有数据集、骨干网络消融实验 (先搜后训) 完成！")
    return all_results

def compute_mean_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    auc_scores = []
    for i in range(labels.shape[1]):
        try:
            auc_scores.append(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            auc_scores.append(0.5)
    return float(np.mean(auc_scores)) if auc_scores else 0.5


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d != 0 else 0.0


def compute_trial_label_metrics(probs: np.ndarray, labels: np.ndarray, label_names, threshold: float = 0.5):
    preds = (probs >= threshold).astype(np.int32)
    gts = labels.astype(np.int32)
    results = []

    for i, label in enumerate(label_names):
        y_true = gts[:, i]
        y_pred = preds[:, i]
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        acc = _safe_div(tp + tn, tp + tn + fp + fn)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        try:
            auc = float(roc_auc_score(y_true, probs[:, i]))
        except ValueError:
            auc = 0.5

        results.append({
            'Label': label, 'Accuracy': acc * 100, 'F1': f1 * 100, 'AUC': auc * 100,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'mAP(%)': np.nan, 'CP(%)': np.nan, 'CR(%)': np.nan, 'CF1(%)': np.nan,
            'OP(%)': np.nan, 'OR(%)': np.nan, 'OF1(%)': np.nan,
        })
    return results


def get_optuna_trial_dir() -> str:
    trial_dir = f"optuna_trial_records_{get_experiment_tag()}_{RUN_TAG}"
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir


def append_trial_metrics_csv(csv_path: str, trial_no: int, trial_auc: float, rows: list) -> None:
    fieldnames = [
        "Trial", "TrialAUC(%)", "Label", "Accuracy", "F1", "AUC", "TP", "FP", "TN", "FN",
        "mAP(%)", "CP(%)", "CR(%)", "CF1(%)", "OP(%)", "OR(%)", "OF1(%)",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            payload = {"Trial": trial_no, "TrialAUC(%)": trial_auc * 100.0}
            payload.update(row)
            writer.writerow(payload)


def train_model_for_optuna(
        model, train_loader, device, num_epochs=OPTUNA_TRAIN_EPOCHS,
        learning_rate=None, use_asl=True, verbose=False
):
    model.to(device)
    if learning_rate is None: learning_rate = 0.001
    optimizer = build_differential_adam(model, learning_rate)

    if use_asl:
        loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    else:
        loss_func = nn.BCEWithLogitsLoss()

    dynamic_aug = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.2, contrast=0.2)])
    all_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        with tqdm(total=total_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", mininterval=2.0,
                  maxinterval=5.0, dynamic_ncols=True) as tepoch:
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = dynamic_aug(images.to(device))
                labels = labels.to(device)

                loss = loss_func(model(images), labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.update(1)

        all_losses.append(running_loss / total_batches)

    return sum(all_losses) / len(all_losses) if all_losses else float('inf')


def objective(trial: optuna.Trial) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_space = str(MHCSRA_SEARCH_SPACE).lower()

    # 👇 同步修改这里的字典，锁死结构参数 👇
    if search_space == "csra":
        mhcsra_kwargs = dict(
            mhcsra_input_dim=FIXED_FEATURE_DIM, mhcsra_out_channel=FIXED_FEATURE_DIM,
            mhcsra_num_heads=FIXED_NUM_HEADS,  # 锁死
            mhcsra_lam=trial.suggest_float("mhcsra_csra_lam", 0.0, 1.0, step=0.1),
            mhcsra_fusion_method=trial.suggest_categorical("mhcsra_csra_fusion_method", ["concat", "sum", "attention"]),
            mhcsra_use_residual=trial.suggest_categorical("mhcsra_csra_use_residual", [True, False]),
        )
    else:
        mhcsra_kwargs = dict(
            mhcsra_num_heads=FIXED_NUM_HEADS, # 锁死
            mhcsra_feature_dim=FIXED_FEATURE_DIM, # 锁死
            mhcsra_out_channel=FIXED_FEATURE_DIM,
            mhcsra_dropout=trial.suggest_float("mhcsra_dropout", 0.0, 0.5, step=0.1),
            mhcsra_shallow_layers=trial.suggest_int("mhcsra_shallow_layers", 1, 3),
            mhcsra_use_shallow=True,
        )

    config = ModelHyperParams(
        num_classes=GLOBAL_NUM_CLASSES,
        batch_size=GLOBAL_BATCH_SIZE, # 锁死
        device=device,
        **mhcsra_kwargs,
        acfp_mode=trial.suggest_categorical("acfp_mode", ['adaptive', 'keep']),
        acfp_target_seq_len=FIXED_SEQ_LEN, # 锁死
        acfp_use_residual=trial.suggest_categorical("acfp_use_residual", [True, False]),
        mmaef_num_heads=FIXED_NUM_HEADS, # 锁死
        mmaef_ffn_hidden_ratio=trial.suggest_float("mmaef_ffn_hidden_ratio", 2.0, 8.0, step=1.0),
        mmaef_use_dropout=trial.suggest_categorical("mmaef_use_dropout", [True, False]),
        mmaef_dropout_rate=trial.suggest_float("mmaef_dropout_rate", 0.0, 0.3, step=0.05),
        hybrid_intermediate_dim=256, # 锁死
        hybrid_dropout_rate=trial.suggest_float("hybrid_dropout_rate", 0.0, 0.3, step=0.05),
        hybrid_activation=trial.suggest_categorical("hybrid_activation", ['relu', 'gelu', 'leaky_relu']),
        hybrid_alpha_init=trial.suggest_float("hybrid_alpha_init", 0.2, 0.8, step=0.05),
        max_iterations=trial.suggest_int("max_iterations", 1, 10),
    )
    # ... 剩下的保持不变 ...
    config.acfp_in_channels = int(
        getattr(config, "mhcsra_csra_output_dim", getattr(config, "mhcsra_out_channel", 2048)))

    try:
        train_loader = load_pth_to_dataloader('../dataset/train_data_juzhong.pth', batch_size=config.batch_size,
                                              shuffle=True)
        test_loader = load_pth_to_dataloader('../dataset/test_data_juzhong.pth', batch_size=config.batch_size,
                                             shuffle=False)
    except Exception as e:
        return float('inf')

    try:
        model = get_train_model(MODEL_VARIANT, GLOBAL_LABELS, config)
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        train_model_for_optuna(model=model, train_loader=train_loader, device=config.device,
                               num_epochs=OPTUNA_TRAIN_EPOCHS, learning_rate=lr, use_asl=True, verbose=False)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                all_preds.append(torch.sigmoid(model(images.to(config.device))).cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        mean_auc = compute_mean_auc(all_preds, all_labels)

        trial_dir = get_optuna_trial_dir()
        torch.save({
            "trial_number": trial.number, "trial_value_auc": mean_auc,
            "model_state_dict": model.state_dict(), "config": config.__dict__,
            "trial_params": dict(trial.params),
        }, os.path.join(trial_dir, f"trial_{trial.number:04d}_auc_{mean_auc:.6f}.pth"))

        trial_rows = compute_trial_label_metrics(all_preds, all_labels, label_names=GLOBAL_LABELS, threshold=0.5)
        append_trial_metrics_csv(os.path.join(trial_dir, f"optuna_trial_metrics_{get_experiment_tag()}.csv"),
                                 trial.number, mean_auc, trial_rows)

        return mean_auc
    except Exception as e:
        return float('-inf')


def run_mhcsra_optuna_test(n_trials=10, num_epochs=30, json_path=json_path_total):
    study = optuna.create_study(direction="maximize", study_name=f"mhcsra_hyperparam_tuning_{get_experiment_tag()}",
                                sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    with open(json_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    with open(json_path_latest, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    best_learning_rate = study.best_params.get('learning_rate', 1e-3)
    if 'learning_rate' in study.best_params: del study.best_params['learning_rate']

    if 'batch_size' in study.best_params:
        batch_size = study.best_params.pop('batch_size')
    else:
        batch_size = 8

    # 🚨 使用安全构建
    best_config = _build_safe_config(study.best_params, batch_size, "cuda" if torch.cuda.is_available() else "cpu")

    model = get_train_model(MODEL_VARIANT, GLOBAL_LABELS, best_config)
    train_loader = load_pth_to_dataloader('../dataset/train_data_juzhong.pth', batch_size=best_config.batch_size,
                                          shuffle=True)
    test_loader = load_pth_to_dataloader('../dataset/test_data_juzhong.pth', batch_size=best_config.batch_size,
                                         shuffle=False)

    trained_model, best_val_auc, best_epoch = train_full_model(
        model=model, train_loader=train_loader, test_loader=test_loader,
        config=best_config, learning_rate=best_learning_rate, num_epochs=num_epochs,
        model_save_path=get_model_save_path_for_variant(model_save_path_total, MODEL_VARIANT),
    )
    return study.best_params, best_config, trained_model


def train_with_best_params_from_json(json_path=json_path_total, num_epochs=50):
    with open(json_path, 'r') as f: best_params = json.load(f)
    learning_rate = best_params.pop('learning_rate', 1e-4)
    batch_size = best_params.pop('batch_size', 16)

    # 🚨 使用安全构建
    best_config = _build_safe_config(best_params, batch_size, "cuda" if torch.cuda.is_available() else "cpu")
    best_model = get_train_model(MODEL_VARIANT, GLOBAL_LABELS, best_config)

    train_loader = load_pth_to_dataloader('../dataset/train_data_juzhong.pth', batch_size=best_config.batch_size,
                                          shuffle=True)
    test_loader = load_pth_to_dataloader('../dataset/test_data_juzhong.pth', batch_size=best_config.batch_size,
                                         shuffle=False)

    best_params['learning_rate'] = learning_rate
    trained_model, best_val_auc, best_epoch = train_full_model(
        model=best_model, train_loader=train_loader, test_loader=test_loader,
        config=best_config, learning_rate=learning_rate, num_epochs=num_epochs,
        model_save_path=get_model_save_path_for_variant(model_save_path_total, MODEL_VARIANT),
    )
    return best_params, best_config, trained_model


def train_full_model(
        model, train_loader, test_loader, config, learning_rate=0.001,
        num_epochs=50, model_save_path=model_save_path_total, loss_kind: str = "asl",
):
    device = config.device
    model.to(device)
    optimizer = build_differential_adam(model, learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if str(loss_kind).lower().strip() == "bce":
        loss_func = nn.BCEWithLogitsLoss()
    else:
        loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    best_val_loss, best_val_auc, best_epoch = float('inf'), 0.0, 0
    train_losses, val_losses, train_auc_scores, val_auc_scores, learning_rates = [], [], [], [], []

    log_path = model_save_path.replace('.pth', '_training_log_juzhong.txt')
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"模型保存路径: {model_save_path}\n")
    log_file.write(f"设备: {device}  学习率: {learning_rate}  Epochs: {num_epochs}  Loss: {loss_kind}\n")
    log_file.write("-" * 100 + "\n")
    epoch_pbar = tqdm(total=num_epochs, desc="完整训练进度")
    dynamic_aug = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.2, contrast=0.2)])

    for epoch in range(num_epochs):
        model.train()
        train_loss, num_batches = 0.0, 0
        all_train_preds, all_train_labels = [], []

        with tqdm(train_loader, desc=f"训练阶段 [Epoch {epoch + 1}/{num_epochs}]", unit="batch", leave=True,
                  mininterval=2.0, maxinterval=5.0, dynamic_ncols=True) as tepoch:
            for images, labels in tepoch:
                optimizer.zero_grad()
                outputs = model(dynamic_aug(images.to(device)))
                loss = loss_func(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
                with torch.no_grad():
                    all_train_preds.append(torch.sigmoid(outputs).cpu())
                    all_train_labels.append(labels.cpu())

        avg_train_loss = train_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        if all_train_preds:
            tp, tl = torch.cat(all_train_preds, dim=0).numpy(), torch.cat(all_train_labels, dim=0).numpy()
            train_auc_scores.append(np.mean(
                [roc_auc_score(tl[:, i], tp[:, i]) if len(np.unique(tl[:, i])) > 1 else 0.5 for i in
                 range(tl.shape[1])]))
        else:
            train_auc_scores.append(0.5)

        model.eval()
        val_loss, num_val_batches = 0.0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device))
                val_loss += loss_func(outputs, labels.to(device)).item()
                num_val_batches += 1
                all_val_preds.append(torch.sigmoid(outputs).cpu())
                all_val_labels.append(labels.cpu())

        avg_val_loss = val_loss / max(num_val_batches, 1)
        val_losses.append(avg_val_loss)

        if all_val_preds:
            vp, vl = torch.cat(all_val_preds, dim=0).numpy(), torch.cat(all_val_labels, dim=0).numpy()
            avg_val_auc = np.mean([roc_auc_score(vl[:, i], vp[:, i]) if len(np.unique(vl[:, i])) > 1 else 0.5 for i in
                                   range(vl.shape[1])])
            val_auc_scores.append(avg_val_auc)
        else:
            val_auc_scores.append(0.5)

        scheduler.step(avg_val_loss)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        auc_improved, loss_improved = avg_val_auc > best_val_auc, avg_val_loss < best_val_loss
        if auc_improved: best_val_auc = avg_val_auc
        if loss_improved: best_val_loss = avg_val_loss

        if auc_improved or loss_improved:
            best_epoch = epoch + 1
            torch.save({'model_state_dict': model.state_dict()}, model_save_path)

        lr_now = optimizer.param_groups[0]["lr"]
        log_file.write(
            f"Epoch {epoch + 1:03d}/{num_epochs:03d} | "
            f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | "
            f"train_auc={train_auc_scores[-1]:.6f} | val_auc={avg_val_auc:.6f} | "
            f"lr={lr_now:.8g} | best_val_auc={best_val_auc:.6f} | best_epoch={best_epoch}\n"
        )
        log_file.flush()

        epoch_pbar.update(1)

    epoch_pbar.close()
    log_file.write("-" * 100 + "\n")
    log_file.write(f"训练结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Best Val AUC: {best_val_auc:.6f}  Best Epoch: {best_epoch}\n")
    log_file.close()

    plot_training_curves(train_losses, val_losses, train_auc_scores, val_auc_scores, learning_rates, model_save_path)

    if os.path.exists(model_save_path): model.load_state_dict(
        torch.load(model_save_path, map_location=device)['model_state_dict'])
    return model, best_val_auc, best_epoch


def plot_training_curves(train_losses, val_losses, train_auc_scores, val_auc_scores, learning_rates, model_save_path):
    try:
        epochs = range(1, len(train_losses) + 1)
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(epochs, train_losses, 'b-', label='训练损失');
        ax1.plot(epochs, val_losses, 'r-', label='验证损失')
        ax1.legend();
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(epochs, train_auc_scores, 'g-', label='训练AUC');
        ax2.plot(epochs, val_auc_scores, 'orange', label='验证AUC')
        ax2.legend();
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(epochs, learning_rates, 'purple', label='学习率')
        ax3.set_yscale('log');
        ax3.legend();
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(epochs, val_losses, color='tab:blue', label='验证损失')
        ax4_auc = ax4.twinx()
        ax4_auc.plot(epochs, val_auc_scores, color='tab:orange', label='验证AUC')
        ax4.legend(loc='upper left');
        ax4_auc.legend(loc='upper right')

        plt.savefig(model_save_path.replace('.pth', '_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        set_global_seed(TRAIN_SEED)
        print("\n" + "=" * 60, flush=True)
        print(f"消融模型: {MODEL_VARIANT}", flush=True)
        print(f"MHCSRA 搜索空间: {MHCSRA_SEARCH_SPACE}", flush=True)
        print(f"随机种子: {TRAIN_SEED}", flush=True)
        print(f"运行标识: {RUN_TAG}", flush=True)
        print(f"主训练轮数: {MAIN_NUM_EPOCHS}", flush=True)
        print(f"Optuna试验数: {MAIN_N_TRIALS}", flush=True)
        print(f"模式(2/4/5) 固定参数文件: {FIXED_BEST_PARAMS_JSON}", flush=True)
        print("=" * 60, flush=True)

        print("请选择运行模式：", flush=True)
        print("1. 先寻优最佳参数（Optuna），再训练", flush=True)
        print("2. 直接使用已有最佳参数训练", flush=True)
        print("3. 论文消融批量训练（model_1..model_10：先搜参再训练）", flush=True)
        print("4. Mode 4: 专门针对 Ours 使用 ASL 进行搜参和完整训练 (支持切换数据集)", flush=True)  # 👈 修改这里
        print("5. 骨干网络 (Backbone) 消融实验批量训练（寻优）", flush=True)

        choice = input("请输入选择 (1 / 2 / 3 / 4 / 5): ").strip()
        print(f"[Main] 已选择模式: {choice}\n", flush=True)

        if choice == "1":
            run_mhcsra_optuna_test(n_trials=MAIN_N_TRIALS, num_epochs=MAIN_NUM_EPOCHS, json_path=json_path_total)
        elif choice == "2":
            json_target = FIXED_BEST_PARAMS_JSON if os.path.isfile(FIXED_BEST_PARAMS_JSON) else json_path_latest
            train_with_best_params_from_json(json_path=json_target, num_epochs=MAIN_NUM_EPOCHS)
        elif choice == "3":
            run_paper_mode3_optuna_then_train(n_trials=PAPER_MODE3_N_TRIALS, optuna_epochs=PAPER_MODE3_OPTUNA_EPOCHS,
                                              full_epochs=PAPER_ABLATION_EPOCHS)
        elif choice == "4":
            # 👈 这里调用新的函数
            run_mode4_ours_optuna_then_train(n_trials=PAPER_MODE3_N_TRIALS, optuna_epochs=PAPER_MODE3_OPTUNA_EPOCHS,
                                             full_epochs=PAPER_ABLATION_EPOCHS)
        elif choice == "5":
            run_backbone_optuna_then_train(n_trials=PAPER_MODE3_N_TRIALS, optuna_epochs=PAPER_MODE3_OPTUNA_EPOCHS,
                                           full_epochs=PAPER_ABLATION_EPOCHS)
        else:
            print("无效输入，请重新运行并输入 1 / 2 / 3 / 4 / 5。")

    except Exception as e:
        print(f"[Main-Error] {e}")
        raise