import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体与负号支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================================
# ⚙️ 核心配置区
# ==========================================================
DATASETS_CONFIG = {
    "TongueDx": {
        "labels": [
            'TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
            'Crack', 'Toothmark', 'FurThick', 'FurYellow'
        ],
        "json_file": "aaa_star_samples_tonguedx.json",
        "train_data_path": "../dataset/train_data.pth",  # 👈 新增：TongueDx 训练集路径
        "baseline_weights": "../our_version7/paper_ablation_best_model_paper_model_1_TongueDx_csra_seed3_e30_20260410_200836.pth",
        "baseline_json": "../our_version7/paper_ablation_best_params_model_1_TongueDx_20260410_200836.json",
        "ours_weights": "../our_version7/paper_ablation_best_model_mode4_ours_TongueDx_seed3_e30_20260425_214653.pth",
        "ours_json": "../our_version7/paper_ablation_best_params_mode4_ours_TongueDx_20260425_214653.json",
        "sota_weights": "../other_model/model_pth/fftrans_tonguedx.pth",
    },
    "ITDD": {
        "labels": [
            'HealthyTongue', 'PeelingCoating', 'RedTongue', 'PurpleTongue', 'ChubbyTongue',
            'ThinTongue', 'RedDot', 'Crack', 'Toothmark', 'FurWhite', 'FurYellow',
            'FurBlack', 'SmoothCoating'
        ],
        "json_file": "aab_star_samples_itdd.json",
        "train_data_path": "../dataset/shezhenv3_train_data.pth",  # 👈 新增：请核对 ITDD 的训练集路径！
        "baseline_weights": "../our_version7/paper_ablation_best_model_paper_model_1_ITDD_csra_seed3_e30_20260410_200836.pth",
        "baseline_json": "../our_version7/paper_ablation_best_params_model_1_ITDD_20260410_200836.json",
        "ours_weights": "../our_version7/paper_ablation_best_model_mode4_ours_ITDD_seed3_e30_20260425_172259.pth",
        "ours_json": "../our_version7/paper_ablation_best_params_mode4_ours_ITDD_20260425_172259.json",
        "sota_weights": "../other_model/model_pth/fftrans_itdd.pth",
    }
}


# ==========================================================
# 🧠 提取模型学到的矩阵 M (Learned Correlation Matrix)
# ==========================================================
def extract_learned_matrix(weights_path):
    if not os.path.exists(weights_path):
        print(f"⚠️ 找不到权重文件: {weights_path}")
        return None

    print(f"正在从 {os.path.basename(weights_path)} 中提取矩阵 M...")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    matrix_key = None
    for k in state_dict.keys():
        if 'correlation_matrix' in k:
            matrix_key = k
            break

    if matrix_key is None:
        raise ValueError("❌ 在权重文件中找不到 'correlation_matrix'！")

    M_tensor = state_dict[matrix_key].numpy()

    # Min-Max 归一化到 [0, 1] 区间
    M_min, M_max = M_tensor.min(), M_tensor.max()
    M_normalized = (M_tensor - M_min) / (M_max - M_min + 1e-8)
    return M_normalized


# ==========================================================
# 📊 计算真实的条件概率矩阵 (Ground Truth Co-occurrence)
# ==========================================================
def calculate_ground_truth_matrix(data_path, num_classes):
    if not os.path.exists(data_path):
        print(f"⚠️ 找不到训练集文件: {data_path}")
        return None

    print(f"正在从 {os.path.basename(data_path)} 计算真实条件概率 P(j|i)...")
    data = torch.load(data_path, map_location='cpu')

    targets = data.get('labels', data.get('targets', None))
    if targets is None:
        raise ValueError("数据集中未找到 'labels' 或 'targets' 字段。")

    if torch.is_tensor(targets):
        targets = targets.numpy()

    adj_matrix = np.zeros((num_classes, num_classes))
    nums_matrix = np.zeros(num_classes)

    for sample_labels in targets:
        for j in range(num_classes):
            if sample_labels[j] == 1:
                nums_matrix[j] += 1
                for k in range(num_classes):
                    if k != j and sample_labels[k] == 1:
                        adj_matrix[j][k] += 1

    nums_expanded = nums_matrix[:, np.newaxis]
    cond_prob_matrix = adj_matrix / (nums_expanded + 1e-8)
    np.fill_diagonal(cond_prob_matrix, 1.0)

    return cond_prob_matrix


# ==========================================================
# 🎨 绘制 1x2 对比热力图
# ==========================================================
def plot_matrix_comparison(M_learned, M_gt, labels, dataset_name):
    # 根据类别数量自适应调整画布大小和字体
    num_classes = len(labels)
    fig_width = 20 if num_classes <= 8 else 24
    fig_height = 8 if num_classes <= 8 else 10
    text_size = 10 if num_classes <= 8 else 8

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    cmap = "Blues"

    # 左图：Ground Truth
    ax1 = axes[0]
    im1 = ax1.imshow(M_gt, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax1.set_title(f"{dataset_name} - Ground Truth $P(j|i)$", fontsize=20, pad=15, fontweight='bold')

    # 右图：Learned Matrix
    ax2 = axes[1]
    im2 = ax2.imshow(M_learned, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax2.set_title(f"{dataset_name} - Learned Matrix $M$", fontsize=20, pad=15, fontweight='bold', color='darkred')

    # 格式化
    for ax, im in zip([ax1, ax2], [im1, im2]):
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)

        matrix = M_gt if ax == ax1 else M_learned
        for i in range(num_classes):
            for j in range(num_classes):
                text_color = "white" if matrix[i, j] > 0.5 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", color=text_color, fontsize=text_size)

    plt.tight_layout()

    save_name = f"Matrix_Comparison_{dataset_name}"
    plt.savefig(f"{save_name}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ {dataset_name} 矩阵对比图已保存: {save_name}.pdf/.png\n")


# ==========================================================
# 🚀 主运行逻辑
# ==========================================================
def main():
    print("🚀 开始批量生成标签相关性矩阵热力图...")

    for dataset_name, config in DATASETS_CONFIG.items():
        print(f"\n{'=' * 50}")
        print(f"📌 处理数据集: {dataset_name}")
        print(f"{'=' * 50}")

        labels = config["labels"]
        weights_path = config["ours_weights"]
        train_data_path = config["train_data_path"]

        # 提取模型学习的 M
        M_learned = extract_learned_matrix(weights_path)
        M_learned = M_learned.T
        if M_learned is None:
            continue

        # 计算 Ground Truth M
        M_gt = calculate_ground_truth_matrix(train_data_path, len(labels))
        if M_gt is None:
            continue

        # 绘图出图
        plot_matrix_comparison(M_learned, M_gt, labels, dataset_name)

    print("🎉 所有数据集矩阵图生成完毕！")


if __name__ == "__main__":
    main()