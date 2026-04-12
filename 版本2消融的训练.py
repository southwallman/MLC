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
# 🚨 直接导入刚刚写好的版本2消融文件
from 版本2消融 import get_paper_ablation_model
from tonguedx_MLC.datapro import load_pth_to_dataloader
import optuna
import json
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 🚨 1. 全局标签与结构配置 (一处修改，全局生效) 🚨
# ==============================================================================
tonguedxlabel=['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
ITDDlabel=[
    'HealthyTongue', 'PeelingCoating', 'RedTongue', 'PurpleTongue', 'ChubbyTongue',
    'ThinTongue', 'RedDot', 'Crack', 'Toothmark', 'FurWhite', 'FurYellow',
    'FurBlack', 'SmoothCoating'
]
GLOBAL_LABELS = tonguedxlabel
GLOBAL_NUM_CLASSES = len(GLOBAL_LABELS)

# 2. 锁死 Batch Size，防止 JSON 读取错乱导致显存溢出或报错
GLOBAL_BATCH_SIZE = 8

# 3. 锁死结构参数 (彻底杜绝测试时 Size Mismatch)
FIXED_NUM_HEADS = 8
FIXED_FEATURE_DIM = 2048
FIXED_SEQ_LEN = 144

# 4. 🚨 锁死数据集路径为 8 分类的 TongueDx (juzhong.pth) 🚨
PAPER_ABLATION_TRAIN_PTH = "../dataset/train_data_juzhong.pth"
PAPER_ABLATION_TEST_PTH = "../dataset/test_data_juzhong.pth"


# ==============================================================================
# 🚨 参数清洗器 🚨
# ==============================================================================
# 🚨 增加 num_classes 参数，默认兜底使用 GLOBAL_NUM_CLASSES
def _build_safe_config(params: dict, batch_size: int, device: str, num_classes: int = GLOBAL_NUM_CLASSES) -> ModelHyperParams:
    cleaned = {}
    for k, v in params.items():
        if k.startswith("mhcsra_csra_"):
            cleaned[k.replace("mhcsra_csra_", "mhcsra_")] = v
        else:
            cleaned[k] = v

    if "mhcsra_lam" in cleaned:
        if "mhcsra_input_dim" not in cleaned: cleaned["mhcsra_input_dim"] = FIXED_FEATURE_DIM
        if "mhcsra_out_channel" not in cleaned: cleaned["mhcsra_out_channel"] = FIXED_FEATURE_DIM

    # 🚨 这里改为使用传入的 num_classes
    cfg = ModelHyperParams(num_classes=num_classes, batch_size=batch_size, device=device, **cleaned)
    cfg.acfp_in_channels = int(
        getattr(cfg, "mhcsra_csra_output_dim", getattr(cfg, "mhcsra_out_channel", FIXED_FEATURE_DIM)))
    return cfg
# def _build_safe_config(params: dict, batch_size: int, device: str) -> ModelHyperParams:
#     cleaned = {}
#     for k, v in params.items():
#         if k.startswith("mhcsra_csra_"):
#             cleaned[k.replace("mhcsra_csra_", "mhcsra_")] = v
#         else:
#             cleaned[k] = v
#
#     if "mhcsra_lam" in cleaned:
#         if "mhcsra_input_dim" not in cleaned: cleaned["mhcsra_input_dim"] = FIXED_FEATURE_DIM
#         if "mhcsra_out_channel" not in cleaned: cleaned["mhcsra_out_channel"] = FIXED_FEATURE_DIM
#
#     cfg = ModelHyperParams(num_classes=GLOBAL_NUM_CLASSES, batch_size=batch_size, device=device, **cleaned)
#     cfg.acfp_in_channels = int(
#         getattr(cfg, "mhcsra_csra_output_dim", getattr(cfg, "mhcsra_out_channel", FIXED_FEATURE_DIM)))
#     return cfg


# 基本配置
MODEL_VARIANT = "exp6"
MHCSRA_SEARCH_SPACE = "csra"
TRAIN_SEED = 3
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_BEST_PARAMS_JSON = os.path.join(_SCRIPT_DIR, "版本7_resnet101_clip_Mod_ACFP_matrix_best_params_exp6.json")


def get_experiment_tag() -> str: return f"{MODEL_VARIANT}_{str(MHCSRA_SEARCH_SPACE).lower()}"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


MAIN_NUM_EPOCHS = 40
OPTUNA_TRAIN_EPOCHS = 40
MAIN_N_TRIALS = 80
OPTIMIZER_WEIGHT_DECAY = 1e-4
NEW_MODULE_LR_MULTIPLIER = 10.0

PAPER_ABLATION_EPOCHS = 30
PAPER_MODE3_N_TRIALS = 15
PAPER_MODE3_OPTUNA_EPOCHS = 3

model_save_path_total = f"版本7_resnet101_clip_Mod_ACFP_matrix_best_model_{get_experiment_tag()}_seed{TRAIN_SEED}_e{MAIN_NUM_EPOCHS}_{RUN_TAG}.pth"
json_path_total = f"版本7_resnet101_clip_Mod_ACFP_matrix_best_params_{get_experiment_tag()}_{RUN_TAG}.json"
json_path_latest = f"版本7_resnet101_clip_Mod_ACFP_matrix_best_params_{get_experiment_tag()}.json"


def get_model_save_path_for_variant(base_path: str, model_variant: str) -> str:
    if not base_path.endswith(".pth"): return f"{base_path}_{model_variant}.pth"
    return base_path.replace(".pth", f"_{model_variant}.pth")


def build_differential_adam(model: nn.Module, base_lr: float, weight_decay: float = OPTIMIZER_WEIGHT_DECAY,
                            new_module_lr_multiplier: float = NEW_MODULE_LR_MULTIPLIER) -> torch.optim.Optimizer:
    backbone_params, new_module_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "cnnbackbone" in name:
            backbone_params.append(param)
        else:
            new_module_params.append(param)
    if not backbone_params: return torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    param_groups = [{"params": backbone_params, "lr": base_lr},
                    {"params": new_module_params, "lr": base_lr * new_module_lr_multiplier}]
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def get_train_model(model_variant: str, label, config):
    v = str(model_variant).lower()
    # 只要是消融模型或我们的最终模型，全走新的工厂函数
    if v.startswith("model_") or v in {"ours", "mti_hanet"}:
        return get_paper_ablation_model(v, label, config)
    # 其他历史版本的模型保留原来的入口
    return get_ablation_model(v, label, config)


def run_paper_mode3_optuna_then_train(*, variants: Optional[List[str]] = None, n_trials: int = PAPER_MODE3_N_TRIALS,
                                      optuna_epochs: int = PAPER_MODE3_OPTUNA_EPOCHS,
                                      full_epochs: int = PAPER_ABLATION_EPOCHS) -> List[Dict[str, Any]]:
    if variants is None: variants = [f"model_{i}" for i in range(1, 11)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_space = str(MHCSRA_SEARCH_SPACE).lower()

    # 🚨 锁死形状超参
    def _build_config_from_trial(trial: optuna.Trial) -> ModelHyperParams:
        if search_space == "csra":
            mhcsra_kwargs = dict(
                mhcsra_input_dim=FIXED_FEATURE_DIM, mhcsra_out_channel=FIXED_FEATURE_DIM,
                mhcsra_num_heads=FIXED_NUM_HEADS,
                mhcsra_lam=trial.suggest_float("mhcsra_csra_lam", 0.0, 1.0, step=0.1),
                mhcsra_fusion_method=trial.suggest_categorical("mhcsra_csra_fusion_method",
                                                               ["concat", "sum", "attention"]),
                mhcsra_use_residual=trial.suggest_categorical("mhcsra_csra_use_residual", [True, False]),
            )
        else:
            mhcsra_kwargs = dict(
                mhcsra_num_heads=FIXED_NUM_HEADS, mhcsra_feature_dim=FIXED_FEATURE_DIM,
                mhcsra_out_channel=FIXED_FEATURE_DIM,
                mhcsra_dropout=trial.suggest_float("mhcsra_dropout", 0.0, 0.5, step=0.1),
                mhcsra_shallow_layers=trial.suggest_int("mhcsra_shallow_layers", 1, 3), mhcsra_use_shallow=True,
            )
        cfg = ModelHyperParams(
            num_classes=GLOBAL_NUM_CLASSES, batch_size=GLOBAL_BATCH_SIZE, device=device, **mhcsra_kwargs,
            acfp_mode=trial.suggest_categorical("acfp_mode", ['adaptive', 'keep']),
            acfp_target_seq_len=FIXED_SEQ_LEN,
            acfp_use_residual=trial.suggest_categorical("acfp_use_residual", [True, False]),
            mmaef_num_heads=FIXED_NUM_HEADS,
            mmaef_ffn_hidden_ratio=trial.suggest_float("mmaef_ffn_hidden_ratio", 2.0, 8.0, step=1.0),
            mmaef_use_dropout=trial.suggest_categorical("mmaef_use_dropout", [True, False]),
            mmaef_dropout_rate=trial.suggest_float("mmaef_dropout_rate", 0.0, 0.3, step=0.05),
            hybrid_intermediate_dim=256,
            hybrid_dropout_rate=trial.suggest_float("hybrid_dropout_rate", 0.0, 0.3, step=0.05),
            hybrid_activation=trial.suggest_categorical("hybrid_activation", ['relu', 'gelu', 'leaky_relu']),
            hybrid_alpha_init=trial.suggest_float("hybrid_alpha_init", 0.2, 0.8, step=0.05),
            max_iterations=trial.suggest_int("max_iterations", 1, 10),
        )
        cfg.acfp_in_channels = int(
            getattr(cfg, "mhcsra_csra_output_dim", getattr(cfg, "mhcsra_out_channel", FIXED_FEATURE_DIM)))
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

                # ASL is the default metric for tuning unless it's explicitly Model 10 (which uses BCE per Table 2 setup)
                use_asl = (variant != "model_10")
                train_model_for_optuna(model=model, train_loader=train_loader, device=config.device,
                                       num_epochs=optuna_epochs, learning_rate=lr, use_asl=use_asl, verbose=False)

                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for images, labels_batch in test_loader:
                        outputs = model(images.to(config.device))
                        all_preds.append(torch.sigmoid(outputs).cpu())
                        all_labels.append(labels_batch.cpu())
                mean_auc = compute_mean_auc(torch.cat(all_preds, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy())

                trial_model_path = os.path.join(trial_dir, f"trial_{trial.number:04d}_auc_{mean_auc:.6f}.pth")
                torch.save(
                    {"trial_number": trial.number, "trial_value_auc": mean_auc, "model_state_dict": model.state_dict(),
                     "config": config.__dict__, "trial_params": dict(trial.params), "variant": variant},
                    trial_model_path)
                return mean_auc
            except Exception as e:
                print(f"[Mode3-{variant}] trial 失败: {e}", flush=True)
                return float("-inf")

        study = optuna.create_study(direction="maximize", study_name=f"paper_{variant}_optuna_{RUN_TAG}",
                                    sampler=optuna.samplers.TPESampler())
        study.optimize(objective_variant, n_trials=n_trials, show_progress_bar=True)

        best_params_all = dict(study.best_params)
        best_learning_rate = float(best_params_all.get("learning_rate", 1e-4))
        best_batch_size = GLOBAL_BATCH_SIZE

        json_path_variant = f"paper_ablation_best_params_{variant}_{RUN_TAG}.json"
        with open(json_path_variant, "w", encoding="utf-8") as f:
            json.dump(best_params_all, f, indent=4)
        with open(f"paper_ablation_best_params_{variant}.json", "w", encoding="utf-8") as f:
            json.dump(best_params_all, f, indent=4)

        best_params_for_cfg = dict(best_params_all)
        best_params_for_cfg.pop("learning_rate", None)
        best_params_for_cfg.pop("batch_size", None)
        best_config = _build_safe_config(best_params_for_cfg, best_batch_size, device)

        model_save_path = f"paper_ablation_best_model_paper_{variant}_{str(MHCSRA_SEARCH_SPACE).lower()}_seed{TRAIN_SEED}_e{full_epochs}_{RUN_TAG}.pth"
        train_loader = load_pth_to_dataloader(PAPER_ABLATION_TRAIN_PTH, batch_size=best_config.batch_size, shuffle=True)
        test_loader = load_pth_to_dataloader(PAPER_ABLATION_TEST_PTH, batch_size=best_config.batch_size, shuffle=False)
        best_model = get_paper_ablation_model(variant, GLOBAL_LABELS, best_config)

        # Determine Loss (Model 6 & 10 use BCE per Table 2 logic, others ASL)
        loss_k = "bce" if variant in ["model_6", "model_10"] else "asl"
        _, best_val_auc, best_epoch = train_full_model(model=best_model, train_loader=train_loader,
                                                       test_loader=test_loader, config=best_config,
                                                       learning_rate=best_learning_rate, num_epochs=full_epochs,
                                                       model_save_path=model_save_path, loss_kind=loss_k)

        results.append({"variant": variant, "best_val_auc": float(best_val_auc), "model_save_path": model_save_path})
    return results


def train_paper_model10_and_ours_from_json(json_path: str, *, num_epochs: int = PAPER_ABLATION_EPOCHS) -> List[
    Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f: best_params = json.load(f)
    learning_rate = float(best_params.get("learning_rate", 1e-4))
    best_params.pop("learning_rate", None);
    best_params.pop("batch_size", None)
    config = _build_safe_config(best_params, GLOBAL_BATCH_SIZE, "cuda" if torch.cuda.is_available() else "cpu")
    train_loader = load_pth_to_dataloader(PAPER_ABLATION_TRAIN_PTH, batch_size=config.batch_size, shuffle=True)
    test_loader = load_pth_to_dataloader(PAPER_ABLATION_TEST_PTH, batch_size=config.batch_size, shuffle=False)

    jobs = [("model_10", "bce", "Model10_BCE_M_alpha"), ("ours", "asl", "Ours_ASL_M_alpha")]
    results: List[Dict[str, Any]] = []
    for variant, loss_kind, tag in jobs:
        suite_tag = f"paper_{variant}_{tag}_{str(MHCSRA_SEARCH_SPACE).lower()}"
        model_save_path = f"paper_ablation_best_model_{suite_tag}_seed{TRAIN_SEED}_e{num_epochs}_{RUN_TAG}.pth"
        model = get_paper_ablation_model(variant, GLOBAL_LABELS, config)
        _, best_val_auc, best_epoch = train_full_model(model=model, train_loader=train_loader, test_loader=test_loader,
                                                       config=config, learning_rate=learning_rate,
                                                       num_epochs=num_epochs, model_save_path=model_save_path,
                                                       loss_kind=loss_kind)
        results.append({"variant": variant, "best_val_auc": float(best_val_auc)})
    return results


def compute_mean_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    auc_scores = []
    for i in range(labels.shape[1]):
        try:
            auc_scores.append(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            auc_scores.append(0.5)
    return float(np.mean(auc_scores)) if auc_scores else 0.5


def train_model_for_optuna(model, train_loader, device, num_epochs=OPTUNA_TRAIN_EPOCHS, learning_rate=0.001,
                           use_asl=True, verbose=False):
    model.to(device)
    optimizer = build_differential_adam(model, learning_rate)
    loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05) if use_asl else nn.BCEWithLogitsLoss()
    dynamic_aug = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.2, contrast=0.2)])
    all_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch",
                  dynamic_ncols=True) as tepoch:
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = loss_func(model(dynamic_aug(images.to(device))), labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.update(1)
        all_losses.append(running_loss / len(train_loader))
    return sum(all_losses) / len(all_losses) if all_losses else float('inf')


def objective(trial: optuna.Trial) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_space = str(MHCSRA_SEARCH_SPACE).lower()

    if search_space == "csra":
        mhcsra_kwargs = dict(mhcsra_input_dim=FIXED_FEATURE_DIM, mhcsra_out_channel=FIXED_FEATURE_DIM,
                             mhcsra_num_heads=FIXED_NUM_HEADS,
                             mhcsra_lam=trial.suggest_float("mhcsra_csra_lam", 0.0, 1.0, step=0.1),
                             mhcsra_fusion_method=trial.suggest_categorical("mhcsra_csra_fusion_method",
                                                                            ["concat", "sum", "attention"]),
                             mhcsra_use_residual=trial.suggest_categorical("mhcsra_csra_use_residual", [True, False]))
    else:
        mhcsra_kwargs = dict(mhcsra_num_heads=FIXED_NUM_HEADS, mhcsra_feature_dim=FIXED_FEATURE_DIM,
                             mhcsra_out_channel=FIXED_FEATURE_DIM,
                             mhcsra_dropout=trial.suggest_float("mhcsra_dropout", 0.0, 0.5, step=0.1),
                             mhcsra_shallow_layers=trial.suggest_int("mhcsra_shallow_layers", 1, 3),
                             mhcsra_use_shallow=True)

    config = ModelHyperParams(
        num_classes=GLOBAL_NUM_CLASSES, batch_size=GLOBAL_BATCH_SIZE, device=device, **mhcsra_kwargs,
        acfp_mode=trial.suggest_categorical("acfp_mode", ['adaptive', 'keep']), acfp_target_seq_len=FIXED_SEQ_LEN,
        acfp_use_residual=trial.suggest_categorical("acfp_use_residual", [True, False]),
        mmaef_num_heads=FIXED_NUM_HEADS,
        mmaef_ffn_hidden_ratio=trial.suggest_float("mmaef_ffn_hidden_ratio", 2.0, 8.0, step=1.0),
        mmaef_use_dropout=trial.suggest_categorical("mmaef_use_dropout", [True, False]),
        mmaef_dropout_rate=trial.suggest_float("mmaef_dropout_rate", 0.0, 0.3, step=0.05),
        hybrid_intermediate_dim=256,
        hybrid_dropout_rate=trial.suggest_float("hybrid_dropout_rate", 0.0, 0.3, step=0.05),
        hybrid_activation=trial.suggest_categorical("hybrid_activation", ['relu', 'gelu', 'leaky_relu']),
        hybrid_alpha_init=trial.suggest_float("hybrid_alpha_init", 0.2, 0.8, step=0.05),
        max_iterations=trial.suggest_int("max_iterations", 1, 10),
    )
    config.acfp_in_channels = int(
        getattr(config, "mhcsra_csra_output_dim", getattr(config, "mhcsra_out_channel", FIXED_FEATURE_DIM)))

    try:
        train_loader = load_pth_to_dataloader(PAPER_ABLATION_TRAIN_PTH, batch_size=config.batch_size, shuffle=True)
        test_loader = load_pth_to_dataloader(PAPER_ABLATION_TEST_PTH, batch_size=config.batch_size, shuffle=False)
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
        mean_auc = compute_mean_auc(torch.cat(all_preds, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy())
        return mean_auc
    except Exception as e:
        return float('-inf')


def run_mhcsra_optuna_test(n_trials=10, num_epochs=30, json_path=json_path_total):
    study = optuna.create_study(direction="maximize", study_name=f"mhcsra_hyperparam_tuning_{get_experiment_tag()}",
                                sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    with open(json_path, 'w') as f: json.dump(study.best_params, f, indent=4)

    best_lr = study.best_params.pop('learning_rate', 1e-3)
    best_config = _build_safe_config(study.best_params, GLOBAL_BATCH_SIZE,
                                     "cuda" if torch.cuda.is_available() else "cpu")
    model = get_train_model(MODEL_VARIANT, GLOBAL_LABELS, best_config)

    train_loader = load_pth_to_dataloader(PAPER_ABLATION_TRAIN_PTH, batch_size=best_config.batch_size, shuffle=True)
    test_loader = load_pth_to_dataloader(PAPER_ABLATION_TEST_PTH, batch_size=best_config.batch_size, shuffle=False)
    _, _, _ = train_full_model(model=model, train_loader=train_loader, test_loader=test_loader, config=best_config,
                               learning_rate=best_lr, num_epochs=num_epochs,
                               model_save_path=get_model_save_path_for_variant(model_save_path_total, MODEL_VARIANT))


def train_full_model(model, train_loader, test_loader, config, learning_rate=0.001, num_epochs=50,
                     model_save_path=model_save_path_total, loss_kind: str = "asl"):
    device = config.device
    model.to(device)
    optimizer = build_differential_adam(model, learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_func = nn.BCEWithLogitsLoss() if str(loss_kind).lower().strip() == "bce" else AsymmetricLoss(gamma_neg=4,
                                                                                                      gamma_pos=1,
                                                                                                      clip=0.05)

    best_val_auc, best_epoch = 0.0, 0
    dynamic_aug = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.2, contrast=0.2)])

    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", dynamic_ncols=True) as tepoch:
            for images, labels in tepoch:
                optimizer.zero_grad()
                loss = loss_func(model(dynamic_aug(images.to(device))), labels.to(device))
                loss.backward()
                optimizer.step()

        model.eval()
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                all_val_preds.append(torch.sigmoid(model(images.to(device))).cpu())
                all_val_labels.append(labels.cpu())

        vp, vl = torch.cat(all_val_preds, dim=0).numpy(), torch.cat(all_val_labels, dim=0).numpy()
        avg_val_auc = compute_mean_auc(vp, vl)
        scheduler.step(1.0 - avg_val_auc)  # 粗略地使用 1 - AUC 代替 loss 让调度器生效

        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            best_epoch = epoch + 1
            torch.save({'model_state_dict': model.state_dict()}, model_save_path)

        print(f"Epoch {epoch + 1} Val AUC: {avg_val_auc:.4f} | Best: {best_val_auc:.4f} at Epoch {best_epoch}")

    if os.path.exists(model_save_path): model.load_state_dict(
        torch.load(model_save_path, map_location=device)['model_state_dict'])
    return model, best_val_auc, best_epoch


def run_paper_mode5_dual_dataset_train(*, variants: Optional[List[str]] = None,
                                       n_trials: int = PAPER_MODE3_N_TRIALS,
                                       optuna_epochs: int = PAPER_MODE3_OPTUNA_EPOCHS,
                                       full_epochs: int = PAPER_ABLATION_EPOCHS):
    if variants is None: variants = [f"model_{i}" for i in range(1, 11)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_space = str(MHCSRA_SEARCH_SPACE).lower()

    # 📦 包装两套数据集的配置
    datasets_config = [
        {
            "name": "TongueDx",
            "train_pth": "../dataset/train_data_juzhong.pth",
            "test_pth": "../dataset/test_data_juzhong.pth",
            "labels": tonguedxlabel,
        },
        {
            "name": "ITDD",
            "train_pth": "../dataset/shezhenv3_train_data.pth",  # 🚨 请确认这个路径存在
            "test_pth": "../dataset/shezhenv3_test_data.pth",
            "labels": ITDDlabel,
        }
    ]

    all_results = []

    # 🌟 外层大循环：遍历数据集
    for ds in datasets_config:
        ds_name = ds["name"]
        current_labels = ds["labels"]
        current_num_classes = len(current_labels)
        train_path = ds["train_pth"]
        test_path = ds["test_pth"]

        print(f"\n{'=' * 80}")
        print(f"🚀 切换数据轨道 -> 开始训练数据集: {ds_name} | 类别数: {current_num_classes}")
        print(f"{'=' * 80}")

        def _build_config_from_trial_ds(trial: optuna.Trial) -> ModelHyperParams:
            # (省略部分结构超参设置，与原版一致...)
            mhcsra_kwargs = dict(
                mhcsra_input_dim=FIXED_FEATURE_DIM, mhcsra_out_channel=FIXED_FEATURE_DIM,
                mhcsra_num_heads=FIXED_NUM_HEADS,
                mhcsra_lam=trial.suggest_float("mhcsra_csra_lam", 0.0, 1.0, step=0.1),
                mhcsra_fusion_method=trial.suggest_categorical("mhcsra_csra_fusion_method",
                                                               ["concat", "sum", "attention"]),
                mhcsra_use_residual=trial.suggest_categorical("mhcsra_csra_use_residual", [True, False]),
            ) if search_space == "csra" else dict(
                mhcsra_num_heads=FIXED_NUM_HEADS, mhcsra_feature_dim=FIXED_FEATURE_DIM,
                mhcsra_out_channel=FIXED_FEATURE_DIM,
                mhcsra_dropout=trial.suggest_float("mhcsra_dropout", 0.0, 0.5, step=0.1),
                mhcsra_shallow_layers=trial.suggest_int("mhcsra_shallow_layers", 1, 3), mhcsra_use_shallow=True,
            )

            cfg = ModelHyperParams(
                num_classes=current_num_classes,  # 🚨 动态类别数
                batch_size=GLOBAL_BATCH_SIZE, device=device, **mhcsra_kwargs,
                acfp_mode=trial.suggest_categorical("acfp_mode", ['adaptive', 'keep']),
                acfp_target_seq_len=FIXED_SEQ_LEN,
                acfp_use_residual=trial.suggest_categorical("acfp_use_residual", [True, False]),
                mmaef_num_heads=FIXED_NUM_HEADS,
                mmaef_ffn_hidden_ratio=trial.suggest_float("mmaef_ffn_hidden_ratio", 2.0, 8.0, step=1.0),
                mmaef_use_dropout=trial.suggest_categorical("mmaef_use_dropout", [True, False]),
                mmaef_dropout_rate=trial.suggest_float("mmaef_dropout_rate", 0.0, 0.3, step=0.05),
                hybrid_intermediate_dim=256,
                hybrid_dropout_rate=trial.suggest_float("hybrid_dropout_rate", 0.0, 0.3, step=0.05),
                hybrid_activation=trial.suggest_categorical("hybrid_activation", ['relu', 'gelu', 'leaky_relu']),
                hybrid_alpha_init=trial.suggest_float("hybrid_alpha_init", 0.2, 0.8, step=0.05),
                max_iterations=trial.suggest_int("max_iterations", 1, 10),
            )
            cfg.acfp_in_channels = int(
                getattr(cfg, "mhcsra_csra_output_dim", getattr(cfg, "mhcsra_out_channel", FIXED_FEATURE_DIM)))
            return cfg

        # 🌟 内层循环：该数据集下的 10 个消融模型
        for variant in variants:
            print(f"\n[Mode5 - {ds_name}] 开始 {variant} ...", flush=True)

            # 🚨 文件夹名加上数据集标识，防止重名
            trial_dir = f"optuna_trial_records_paper_{variant}_{ds_name}_{RUN_TAG}"
            os.makedirs(trial_dir, exist_ok=True)

            def objective_variant(trial: optuna.Trial) -> float:
                config = _build_config_from_trial_ds(trial)
                try:
                    # 🚨 使用当前轨道的数据集路径
                    train_loader = load_pth_to_dataloader(train_path, batch_size=config.batch_size, shuffle=True)
                    test_loader = load_pth_to_dataloader(test_path, batch_size=config.batch_size, shuffle=False)
                    model = get_paper_ablation_model(variant, current_labels, config)
                    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)

                    use_asl = (variant != "model_10")
                    train_model_for_optuna(model=model, train_loader=train_loader, device=config.device,
                                           num_epochs=optuna_epochs, learning_rate=lr, use_asl=use_asl, verbose=False)

                    model.eval()
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for images, labels_batch in test_loader:
                            outputs = model(images.to(config.device))
                            all_preds.append(torch.sigmoid(outputs).cpu())
                            all_labels.append(labels_batch.cpu())
                    mean_auc = compute_mean_auc(torch.cat(all_preds, dim=0).numpy(),
                                                torch.cat(all_labels, dim=0).numpy())

                    trial_model_path = os.path.join(trial_dir, f"trial_{trial.number:04d}_auc_{mean_auc:.6f}.pth")
                    torch.save(
                        {"trial_number": trial.number, "trial_value_auc": mean_auc,
                         "model_state_dict": model.state_dict(),
                         "config": config.__dict__, "trial_params": dict(trial.params), "variant": variant},
                        trial_model_path)
                    return mean_auc
                except Exception as e:
                    print(f"[Mode5-{ds_name}-{variant}] trial 失败: {e}", flush=True)
                    return float("-inf")

            study = optuna.create_study(direction="maximize", study_name=f"paper_{variant}_{ds_name}_optuna_{RUN_TAG}",
                                        sampler=optuna.samplers.TPESampler())
            study.optimize(objective_variant, n_trials=n_trials, show_progress_bar=True)

            best_params_all = dict(study.best_params)
            best_learning_rate = float(best_params_all.get("learning_rate", 1e-4))

            # 🚨 JSON 文件名加上数据集标识
            json_path_variant = f"paper_ablation_best_params_{variant}_{ds_name}_{RUN_TAG}.json"
            with open(json_path_variant, "w", encoding="utf-8") as f:
                json.dump(best_params_all, f, indent=4)

            best_params_for_cfg = dict(best_params_all)
            best_params_for_cfg.pop("learning_rate", None)
            best_params_for_cfg.pop("batch_size", None)

            # 🚨 构建最终配置时，传入正确的 num_classes
            best_config = _build_safe_config(best_params_for_cfg, GLOBAL_BATCH_SIZE, device,
                                             num_classes=current_num_classes)

            # 🚨 PTH 文件名加上数据集标识
            model_save_path = f"paper_ablation_best_model_paper_{variant}_{ds_name}_{str(MHCSRA_SEARCH_SPACE).lower()}_seed{TRAIN_SEED}_e{full_epochs}_{RUN_TAG}.pth"

            train_loader = load_pth_to_dataloader(train_path, batch_size=best_config.batch_size, shuffle=True)
            test_loader = load_pth_to_dataloader(test_path, batch_size=best_config.batch_size, shuffle=False)
            best_model = get_paper_ablation_model(variant, current_labels, best_config)

            loss_k = "bce" if variant in ["model_6", "model_10"] else "asl"
            _, best_val_auc, best_epoch = train_full_model(model=best_model, train_loader=train_loader,
                                                           test_loader=test_loader, config=best_config,
                                                           learning_rate=best_learning_rate, num_epochs=full_epochs,
                                                           model_save_path=model_save_path, loss_kind=loss_k)

            all_results.append({"dataset": ds_name, "variant": variant, "best_val_auc": float(best_val_auc),
                                "model_save_path": model_save_path})

    return all_results
if __name__ == "__main__":
    try:
        set_global_seed(TRAIN_SEED)
        print(
            "1. 先寻优最佳参数（Optuna），再训练\n"
            "3. 论文消融批量训练（model_1..model_10）\n"
            "4. Model10(BCE) + Ours(ASL) 各跑一轮\n"
            "5. 双数据集并行消融训练 (TongueDx + ITDD 全跑一遍)",  # 🚨 加了这行
            flush=True)
        choice = input("请输入选择 (1 / 3 / 4 / 5): ").strip()
        print(f"[Main] 已选择模式: {choice}\n", flush=True)

        if choice == "1":
            run_mhcsra_optuna_test(n_trials=MAIN_N_TRIALS, num_epochs=MAIN_NUM_EPOCHS, json_path=json_path_total)
        elif choice == "3":
            run_paper_mode3_optuna_then_train(n_trials=PAPER_MODE3_N_TRIALS, optuna_epochs=PAPER_MODE3_OPTUNA_EPOCHS,
                                              full_epochs=PAPER_ABLATION_EPOCHS)
        elif choice == "4":
            json_target = FIXED_BEST_PARAMS_JSON
            if os.path.isfile(json_target):
                train_paper_model10_and_ours_from_json(json_path=json_target, num_epochs=PAPER_ABLATION_EPOCHS)
            else:
                print("缺少最优参数 JSON 文件。")
        # 🚨 加了模式5的路由
        elif choice == "5":
            run_paper_mode5_dual_dataset_train(n_trials=PAPER_MODE3_N_TRIALS,
                                               optuna_epochs=PAPER_MODE3_OPTUNA_EPOCHS,
                                               full_epochs=PAPER_ABLATION_EPOCHS)
    except Exception as e:
        print(f"[Main-Error] {e}")
        raise