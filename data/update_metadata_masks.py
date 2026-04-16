#!/usr/bin/env python3
"""更新 metadata CSV 中的 mask 路径，加入新推理的 no_mask_pred"""

import os
import pandas as pd

NNUNET_ROOT = "/mnt/cfs_algo_bj/models/experiments/lirunze/code/nnUNet/output"
METADATA_FILE = "/mnt/cfs_algo_bj/models/experiments/lirunze/code/project/outputs/metadata/metadata_master.csv"

# 疾病到 mask 文件夹的映射（no_mask_pred 版本）
DISEASE_TO_MASK_FOLDER = {
    "SST": "Dataset001_jianjia_no_mask_pred",
    "IST": "Dataset002_jianxiu_no_mask_pred",
    "SSC": "Dataset005_gangxia_no_mask_pred",
    "LHBT": "Dataset006_gangshang_no_mask_pred",
    "IGHL": "Dataset008_jianguanjie_no_mask_pred",
    "RIPI": "Dataset009_gongretou_no_mask_pred",
    "GHOA": "Dataset010_yenang_2_no_mask_pred",
}

DISEASES = ["SST", "IST", "SSC", "LHBT", "IGHL", "RIPI", "GHOA"]


def get_new_mask_path(exam_id, disease):
    """Get nnUNet mask path from no_mask_pred folder."""
    mask_folder = DISEASE_TO_MASK_FOLDER.get(disease, "")
    if not mask_folder:
        return ""
    mask_path = os.path.join(NNUNET_ROOT, mask_folder, "%s.nii.gz" % exam_id)
    return mask_path if os.path.exists(mask_path) else ""


def main():
    print("读取 metadata CSV...")
    df = pd.read_csv(METADATA_FILE)
    print(f"共 {len(df)} 行")

    # 检查列是否存在
    mask_cols_exist = [f"mask_path_{d}" in df.columns for d in DISEASES]
    print(f"mask_path_* 列存在: {any(mask_cols_exist)}, 数量: {sum(mask_cols_exist)}")

    # 如果 mask_path 列不存在，就跳过更新
    if not any(mask_cols_exist):
        print("没有 mask_path 列，跳过更新")
        return

    # 初始化计数器
    updated = 0

    for idx, row in df.iterrows():
        exam_id = row['exam_id']

        for disease in DISEASES:
            old_mask_col = f"mask_path_{disease}"

            # 检查列是否存在
            if old_mask_col not in df.columns:
                continue

            # 如果原来的 mask 路径为空，尝试从 no_mask_pred 获取
            old_val = row.get(old_mask_col, "")
            if pd.isna(old_val) or str(old_val).strip() == "":
                new_mask_path = get_new_mask_path(exam_id, disease)
                if new_mask_path:
                    df.at[idx, old_mask_col] = new_mask_path
                    updated += 1

        if (idx + 1) % 500 == 0:
            print(f"已处理 {idx + 1}/{len(df)}, 更新了 {updated} 个 mask")

    # 更新 has_any_mask 列
    mask_cols = [f"mask_path_{d}" for d in DISEASES if f"mask_path_{d}" in df.columns]
    if mask_cols:
        df['has_any_mask'] = df[mask_cols].notna().astype(int).replace(0, pd.NA).any(axis=1).astype(int)

    # 保存
    output_file = METADATA_FILE.replace('.csv', '_updated.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n完成! 共更新了 {updated} 个 mask 路径")
    print(f"保存到: {output_file}")


if __name__ == "__main__":
    main()