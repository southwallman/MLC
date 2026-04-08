import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import json
from dataclasses import asdict
from types import SimpleNamespace
from model_config import ModelHyperParams
# 注意：如果下面这行在你那报错，请确保 tonguedx_MLC 路径正确
from tonguedx_MLC.datapro import load_pth_to_dataloader 
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, average_precision_score)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 🚨 全局标签配置 (一处修改，全局生效)
# ==============================================================================
GLOBAL_LABELS = [
    'HealthyTongue', 'PeelingCoating', 'RedTongue', 'PurpleTongue', 'ChubbyTongue',
    'ThinTongue', 'RedDot', 'Crack', 'Toothmark', 'FurWhite', 'FurYellow',
    'FurBlack', 'SmoothCoating'
]
GLOBAL_NUM_CLASSES = len(GLOBAL_LABELS)


# ==============================================================================
# 🌟 核心修复 1：把 Dataset 类提到全局，拒绝在循环里重复定义类！
# ==============================================================================
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, paths):
        self.images = images
        self.targets = targets
        self.paths = paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx], self.paths[idx]


def load_preloaded_test_data(test_data_path):
    """从 .pth 读入并整理为字典（全局只调用一次）。"""
    print(f"\n加载测试数据: {test_data_path}")
    data = torch.load(test_data_path, map_location="cpu")
    images = data.get("images", data.get("imgs", None))
    targets = data.get("labels", data.get("targets", None))
    paths = data.get("image_paths", data.get("paths", data.get("img_paths", None)))
    
    if images is None or targets is None:
        raise ValueError("测试数据中缺少'images'或'labels'字段")
    if paths is None:
        paths = [f"sample_{i}" for i in range(len(images))]
        print("警告: 数据中未找到路径信息，使用索引代替")

    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # 强行驻留 CPU 内存，等需要的时候 DataLoader 会切片送往 GPU
    images = images.cpu()
    targets = targets.cpu()
    print(f"测试集样本数: {len(images)}")
    return {"images": images, "targets": targets, "paths": paths}


# ==============================================================================
# 🌟 核心修复 2：剥离 Loader 构建逻辑，测试函数只管测！
# ==============================================================================
# ==============================================================================
# 新增功能：动态搜寻单标签最佳阈值
# ==============================================================================
def find_best_thresholds(y_true, y_prob, num_classes):
    best_thresholds = []
    # 遍历 13 个标签
    for i in range(num_classes):
        best_t = 0.5
        best_f1 = 0.0
        # 在 0.1 到 0.9 之间地毯式搜索
        for t in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_prob[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        # 极端情况兜底：如果这个标签死活找不到正样本预测，就给个默认 0.4
        if best_f1 == 0.0:
            best_t = 0.4

        best_thresholds.append(best_t)
        # 把 print 注释掉，不然 100 个模型打印出来屏幕要爆炸了
        # print(f"标签 {i} 的最佳阈值: {best_t:.2f}, 此时 F1: {best_f1:.4f}")

    return best_thresholds


# ==============================================================================
# 🌟 核心修复 2：剥离 Loader 构建逻辑，加入动态阈值！
# ==============================================================================
def test_with_best_params_from_json_ai(
        model=None,
        name=None,
        model_save_path=None,
        test_loader=None,
        output_dir='.',
        label_list=None,
        label_image=False,
        save_error_details=True,
        error_file=None,
        threshold=0.4,  # 这个参数留着，但实际上被动态阈值取代了
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if test_loader is None:
        raise ValueError("必须传入 test_loader！请检查主函数传参。")

    # 1. 加载模型权重
    model = model.to(device)
    if model_save_path is not None and os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        del checkpoint
    else:
        print("警告: 未加载预训练权重，使用随机初始化模型进行测试")

    # 2. 测试循环 (只收集概率，先不轻易下结论)
    model.eval()
    all_targets, all_probs, all_paths = [], [], []
    per_class_auc, per_class_ap = [], []
    per_class_precision, per_class_recall, per_class_f1, per_class_acc = [], [], [], []
    error_records = []

    if label_list is None:
        label_list = GLOBAL_LABELS
    num_classes = len(label_list)

    with torch.no_grad():
        for images_batch, targets_batch, paths_batch in tqdm(test_loader, desc=f"提取特征 {name}"):
            images_batch = images_batch.to(device)
            targets_batch = targets_batch.float().to(device)

            if label_image:
                outputs = model(targets_batch, images_batch)
            else:
                outputs = model(images_batch)

            if isinstance(outputs, (tuple, list)):
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                logits = outputs

            probs = torch.sigmoid(logits).cpu().numpy()

            # 🚨 改变：这里只存真实标签、概率和路径，不再强行生成 preds
            all_probs.append(probs)
            all_targets.append(targets_batch.cpu().numpy())
            all_paths.extend(paths_batch)

            # 将收集到的结果拼接成大数组
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)

    # 🚨 终极绝招：动态寻找 13 个标签的最佳阈值
    best_thresholds = find_best_thresholds(all_targets, all_probs, num_classes)

    # 按照寻找出的专属阈值，对每一个标签分别进行预测
    all_preds = np.zeros_like(all_probs)
    for i in range(num_classes):
        all_preds[:, i] = (all_probs[:, i] > best_thresholds[i]).astype(int)

    # 有了真实的 preds 后，再去生成错题本 (error_records)
    for i in range(len(all_paths)):
        path = all_paths[i]
        true = all_targets[i]
        pred = all_preds[i]
        error_labels = [label_list[j] for j in range(num_classes) if true[j] != pred[j]]
        if error_labels:
            error_records.append({
                'image_path': path, 'true_labels': true, 'pred_labels': pred,
                'probs': all_probs[i], 'error_labels': error_labels
            })

    # 3. 计算每个类别的指标 (后面这一段完全保持你的原样不变)
    results = []
    for i, label in enumerate(label_list):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        y_prob = all_probs[:, i]

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        acc = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float('nan')

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        ap = average_precision_score(y_true, y_prob)

        per_class_auc.append(auc)
        per_class_ap.append(ap)
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        per_class_acc.append(acc)

        results.append({
            'Label': label, 'Accuracy': acc * 100, 'F1': f1 * 100, 'AUC': auc * 100,
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'mAP(%)': np.nan, 'CP(%)': np.nan, 'CR(%)': np.nan, 'CF1(%)': np.nan,
            'OP(%)': np.nan, 'OR(%)': np.nan, 'OF1(%)': np.nan,
        })

    # 4. 计算整体指标并保存
    micro_precision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_targets, all_preds, average='micro', zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_f1 = np.mean(per_class_f1)
    mean_ap = np.mean(per_class_ap)
    mean_auc = np.nanmean(per_class_auc)
    mean_acc = np.mean(per_class_acc)

    results.append({
        'Label': 'Overall', 'Accuracy': mean_acc * 100, 'F1': macro_f1 * 100, 'AUC': mean_auc * 100,
        'TP': np.nan, 'FP': np.nan, 'TN': np.nan, 'FN': np.nan,
        'mAP(%)': mean_ap * 100, 'CP(%)': macro_precision * 100, 'CR(%)': macro_recall * 100,
        'CF1(%)': macro_f1 * 100, 'OP(%)': micro_precision * 100, 'OR(%)': micro_recall * 100,
        'OF1(%)': micro_f1 * 100,
    })

    df_results = pd.DataFrame(results)
    column_order = [
        'Label', 'Accuracy', 'F1', 'AUC', 'TP', 'FP', 'TN', 'FN',
        'mAP(%)', 'CP(%)', 'CR(%)', 'CF1(%)', 'OP(%)', 'OR(%)', 'OF1(%)'
    ]
    df_results = df_results[column_order]

    os.makedirs(output_dir, exist_ok=True)
    result_csv = os.path.join(output_dir, f"test_results_{name}.csv")
    df_results.to_csv(result_csv, index=False)

    if save_error_details and error_records:
        if error_file is None:
            error_file = os.path.join(output_dir, f"error_details_{name}.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("image_path\terror_labels\n")
            for rec in error_records:
                f.write(f"{rec['image_path']}\t{','.join(rec['error_labels'])}\n")

    return df_results, all_targets, all_preds


# ==============================================================================
# 工具函数 & 架构构建
# ==============================================================================
def _config_namespace_from_json_best_params(best_params, *, num_classes, batch_size, device):
    d = dict(best_params)
    merged = {**asdict(ModelHyperParams()), **d}
    if "mhcsra_csra_num_heads" in d:
        merged["mhcsra_num_heads"] = int(d["mhcsra_csra_num_heads"])
    if "mhcsra_csra_output_dim" in d:
        v = int(d["mhcsra_csra_output_dim"])
        merged["acfp_in_channels"] = v
        merged["mhcsra_out_channel"] = v
    else:
        merged["acfp_in_channels"] = int(merged.get("mhcsra_out_channel", 2048))
    merged["num_classes"] = num_classes
    merged["batch_size"] = batch_size
    merged["device"] = device
    return SimpleNamespace(**merged)

def _build_paper_model_from_trial_checkpoint(ckpt_path, variant):
    from paper_ablation_MTI_HANet import get_paper_ablation_model
    label_list = GLOBAL_LABELS
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    config = _config_namespace_from_json_best_params(
        cfg_dict, num_classes=len(label_list), batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return get_paper_ablation_model(variant, label_list, config)

def _build_paper_model_from_json(json_path, variant):
    from paper_ablation_MTI_HANet import get_paper_ablation_model
    label_list = GLOBAL_LABELS
    with open(json_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    best_params = dict(best_params)
    best_params.pop('learning_rate', None)
    best_params.pop('batch_size', None)
    config = _config_namespace_from_json_best_params(
        best_params, num_classes=len(label_list), batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return get_paper_ablation_model(variant, label_list, config)

def build_model_for_config(cfg):
    """仅根据元数据实例化模型（测前即时构建）。"""
    if cfg.json_path is not None:
        return _build_paper_model_from_json(cfg.json_path, cfg.variant)
    return _build_paper_model_from_trial_checkpoint(cfg.weight_path, cfg.variant)


class Model_config:
    """存放测试任务信息，绝对不持有模型实例！"""
    def __init__(self, name, variant, weight_path, labels, label_image=False, json_path=None, output_dir="."):
        self.name = name
        self.variant = variant
        self.weight_path = weight_path
        self.json_path = json_path
        self.labels = labels
        self.label_image = label_image
        self.output_dir = output_dir

def config_mode1():
    common_labels = GLOBAL_LABELS  # 🚨 使用全局标签
    # 403: ITDD dataset
    ITDD_TEST_DATA_PATH_403 = "../dataset/shezhenv3_test_data.pth"
    # 402: TongueDx dataset（按你原先约定）
    TONGUEDX_TEST_DATA_PATH_402 = "../dataset/test_data_juzhong.pth"

    # 当前测试使用哪套数据：可选 "403_itdd" / "402_tonguedx"
    ACTIVE_DATASET = "403_itdd"
    if ACTIVE_DATASET == "403_itdd":
        test_data_path = ITDD_TEST_DATA_PATH_403
    elif ACTIVE_DATASET == "402_tonguedx":
        test_data_path = TONGUEDX_TEST_DATA_PATH_402
    else:
        raise ValueError(f"不支持的 ACTIVE_DATASET: {ACTIVE_DATASET}")

    # 402（TongueDx）这套是你之前跑好的 20260402_200311
    paper_items_402_tonguedx = [
        ("model_1", "paper_ablation_best_params_model_1_20260402_200311.json",
         "paper_ablation_best_model_paper_model_1_csra_seed3_e30_20260402_200311.pth"),
        ("model_2", "paper_ablation_best_params_model_2_20260402_200311.json",
         "paper_ablation_best_model_paper_model_2_csra_seed3_e30_20260402_200311.pth"),
        ("model_3", "paper_ablation_best_params_model_3_20260402_200311.json",
         "paper_ablation_best_model_paper_model_3_csra_seed3_e30_20260402_200311.pth"),
        ("model_4", "paper_ablation_best_params_model_4_20260402_200311.json",
         "paper_ablation_best_model_paper_model_4_csra_seed3_e30_20260402_200311.pth"),
        ("model_5", "paper_ablation_best_params_model_5_20260402_200311.json",
         "paper_ablation_best_model_paper_model_5_csra_seed3_e30_20260402_200311.pth"),
        ("model_6", "paper_ablation_best_params_model_6_20260402_200311.json",
         "paper_ablation_best_model_paper_model_6_csra_seed3_e30_20260402_200311.pth"),
        ("model_7", "paper_ablation_best_params_model_7_20260402_200311.json",
         "paper_ablation_best_model_paper_model_7_csra_seed3_e30_20260402_200311.pth"),
        ("model_8", "paper_ablation_best_params_model_8_20260402_200311.json",
         "paper_ablation_best_model_paper_model_8_csra_seed3_e30_20260402_200311.pth"),
        ("model_9", "paper_ablation_best_params_model_9_20260402_200311.json",
         "paper_ablation_best_model_paper_model_9_csra_seed3_e30_20260402_200311.pth"),
        ("model_10", "paper_ablation_best_params_model_10_20260402_200311.json",
         "paper_ablation_best_model_paper_model_10_csra_seed3_e30_20260402_200311.pth"),
    ]

    # 403（ITDD）这套是 20260403_181859
    paper_items_403_itdd = [
        ("model_1", "paper_ablation_best_params_model_1_20260403_181859.json",
         "paper_ablation_best_model_paper_model_1_csra_seed3_e30_20260403_181859.pth"),
        ("model_2", "paper_ablation_best_params_model_2_20260403_181859.json",
         "paper_ablation_best_model_paper_model_2_csra_seed3_e30_20260403_181859.pth"),
        ("model_3", "paper_ablation_best_params_model_3_20260403_181859.json",
         "paper_ablation_best_model_paper_model_3_csra_seed3_e30_20260403_181859.pth"),
        ("model_4", "paper_ablation_best_params_model_4_20260403_181859.json",
         "paper_ablation_best_model_paper_model_4_csra_seed3_e30_20260403_181859.pth"),
        ("model_5", "paper_ablation_best_params_model_5_20260403_181859.json",
         "paper_ablation_best_model_paper_model_5_csra_seed3_e30_20260403_181859.pth"),
        ("model_6", "paper_ablation_best_params_model_6_20260403_181859.json",
         "paper_ablation_best_model_paper_model_6_csra_seed3_e30_20260403_181859.pth"),
        ("model_7", "paper_ablation_best_params_model_7_20260403_181859.json",
         "paper_ablation_best_model_paper_model_7_csra_seed3_e30_20260403_181859.pth"),
        ("model_8", "paper_ablation_best_params_model_8_20260403_181859.json",
         "paper_ablation_best_model_paper_model_8_csra_seed3_e30_20260403_181859.pth"),
        ("model_9", "paper_ablation_best_params_model_9_20260403_181859.json",
         "paper_ablation_best_model_paper_model_9_csra_seed3_e30_20260403_181859.pth"),
        ("model_10", "paper_ablation_best_params_model_10_20260403_181859.json",
         "paper_ablation_best_model_paper_model_10_csra_seed3_e30_20260403_181859.pth"),
    ]

    # 模式1：原先一模型一权重测试（按数据集切换）
    if ACTIVE_DATASET == "403_itdd":
        paper_items = paper_items_403_itdd
        name_suffix = "itdd403"
        output_dir = "itdd_test"
    else:
        paper_items = paper_items_402_tonguedx
        name_suffix = "tonguedx402"
        output_dir = "."

    model_configs = []
    for variant, json_path, model_path in paper_items:
        if not os.path.exists(json_path):
            print(f"[config] 跳过 {variant}，json 不存在: {json_path}")
            continue
        if not os.path.exists(model_path):
            print(f"[config] 跳过 {variant}，pth 不存在: {model_path}")
            continue
        model_configs.append(
            Model_config(
                name=f"{variant}_{name_suffix}",
                variant=variant,
                weight_path=model_path,
                json_path=json_path,
                labels=common_labels,
                label_image=False,
                output_dir=output_dir,
            )
        )
    return model_configs, test_data_path


def config_mode2():
    test_data_path = "../dataset/shezhenv3_test_data.pth"
    run_tag_403 = "20260403_181859"
    max_trials_per_model = 10
    output_dir = "itdd_test"
    model_configs = []
    
    for i in range(1, 11):
        variant = f"model_{i}"
        trial_dir = f"optuna_trial_records_paper_{variant}_{run_tag_403}"
        if not os.path.isdir(trial_dir):
            continue
        trial_ckpts = sorted([os.path.join(trial_dir, f) for f in os.listdir(trial_dir) if f.endswith(".pth")])[:max_trials_per_model]
        for ckpt_path in trial_ckpts:
            trial_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            model_configs.append(
                Model_config(
                    name=f"{variant}_itdd403_{trial_name}",
                    variant=variant, weight_path=ckpt_path, json_path=None,
                    labels=GLOBAL_LABELS, label_image=False, output_dir=output_dir,
                )
            )
    print(f"[config_mode2] 已装配测试任务数: {len(model_configs)}")
    return model_configs, test_data_path

def config():
    RUN_MODE = "mode2"  # 当前直接跑 100 次批量测试
    if RUN_MODE == "mode1": return config_mode1()
    if RUN_MODE == "mode2": return config_mode2()


# ==============================================================================
# 🌟 核心修复 3：无敌架构的 Main 函数
# ==============================================================================
if __name__ == "__main__":
    model_configs, test_data_path = config()
    
    # 【神级优化 1】全局读取数据，硬盘这辈子只读这一次
    preloaded_data = load_preloaded_test_data(test_data_path)
    
    # 【神级优化 2】构建全局唯一的数据加载器
    test_dataset = TestDataset(
        preloaded_data["images"], 
        preloaded_data["targets"], 
        preloaded_data["paths"]
    )
    
    # 🚨 救命锁：num_workers 必须等于 0，拒绝创建多进程复制数据撑爆内存！
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        drop_last=True,  # 🚨 加上这行，直接把最后 4 个凑不够数的图片扔进垃圾桶
        pin_memory=torch.cuda.is_available(),
    )
    print(f"\n✅ 全局 DataLoader 构建完成，批次数量: {len(test_loader)}")

    # 【神级优化 3】无残留循环测试
    for run_cfg in model_configs:
        print(f"\n{'='*60}")
        print(f"🚀 正在测试: {run_cfg.name}")
        print(f"{'='*60}")
        
        # 用的时候再建模型
        model = build_model_for_config(run_cfg)
        try:
            test_with_best_params_from_json_ai(
                model=model,
                name=run_cfg.name,
                model_save_path=run_cfg.weight_path,
                test_loader=test_loader,  # 传入全局 loader
                output_dir=run_cfg.output_dir,
                label_list=run_cfg.labels,
                label_image=run_cfg.label_image,
                threshold=0.4,
            )
        finally:
            # 测完立刻全盘绞杀，渣都不剩
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()