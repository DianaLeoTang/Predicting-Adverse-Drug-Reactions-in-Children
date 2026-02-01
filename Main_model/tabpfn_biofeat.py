import os  # 导入操作系统接口模块，用于文件路径操作和目录创建
import sys  # 导入系统相关参数和函数，用于程序退出和标准输出刷新
import json  # 导入JSON模块，用于解析和保存JSON格式数据
import joblib  # 导入joblib模块，用于保存和加载特征选择器等对象
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据处理和CSV文件读写
from datetime import datetime  # 从datetime模块导入datetime类，用于获取当前时间戳

from sklearn.metrics import (  # 从sklearn.metrics导入评估指标
    roc_auc_score,  # ROC曲线下面积（AUC）计算函数
    average_precision_score,  # 平均精确率（PR AUC）计算函数
    accuracy_score,  # 准确率计算函数
    f1_score,  # F1分数计算函数
    matthews_corrcoef,  # 马修斯相关系数（MCC）计算函数
    balanced_accuracy_score,  # 平衡准确率计算函数
    recall_score  # 召回率（敏感度/特异度）计算函数
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # 从sklearn.model_selection导入数据划分函数
# StratifiedKFold: 分层K折交叉验证，保持各类别比例
# train_test_split: 训练集和测试集划分函数
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # 从sklearn.feature_selection导入特征选择函数
# SelectKBest: 选择K个最佳特征的选择器
# mutual_info_classif: 互信息特征选择评分函数

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier  # 导入AutoTabPFN分类器
# TabPFN是一个基于Transformer的表格数据分类器，适合小样本场景

# =========================
# 可配置参数
# =========================

# 指定 TabPFN 模型权重路径（按本地环境调整）
os.environ['TABPFN_MODEL_PATH'] = "/public/home/tianyao/.conda/envs/tabpfn/lib/python3.11/site-packages/tabpfn/models/tabpfn-v2-classifier-od3j1g5m.ckpt"  # 设置TabPFN预训练模型文件路径环境变量

# - FEATURES_PATH：包含标签列（多个 ADR 端点）与 'Smiles'
# - DATA_PATH：包含 'Smiles' 与 'BioFeat'（需解析成数值向量）
FEATURES_PATH = '/public/home/tianyao/biosignature/features.csv'  # 特征文件路径：包含标签列（多个ADR端点）与'Smiles'列
DATA_PATH     = '/public/home/tianyao/xgboost/433_labeled_results_with_smiles.csv'  # 数据文件路径：包含'Smiles'与'BioFeat'（需解析成数值向量）

# 从第几个任务开始（1-based），用于断点续跑
START_TASK_ID = 1  # 任务起始索引，用于从指定任务开始继续运行（断点续跑功能）

# 分层 K 折交叉验证设置
N_SPLITS = 10  # 交叉验证的折数，设置为10折交叉验证
SHUFFLE = True  # 是否在划分前打乱数据，True表示打乱
RANDOM_STATE = 42  # 随机数种子，确保结果可复现

# 特征选择设置（互信息 SelectKBest）
USE_MI_SELECTION = True  # 是否启用互信息特征选择（维度很高时强烈建议开启），True表示启用
N_FEATURES_MI = 500  # 每折选择的特征上限（自动取 min(k, 当前维度)），从高维特征中选择500个最重要的特征

# AutoTabPFN 设置
TABPFN_DEVICE = 'auto'  # 计算设备选择：'cuda'（GPU）| 'cpu'（CPU）| 'auto'（自动选择）
TABPFN_MAX_TIME = 600  # 每次 fit 的时间上限（秒），600秒=10分钟，防止单次训练时间过长

# 输出目录
OUTPUT_ROOT = 'tabpfn_cv_biofeat'  # 输出根目录名称
os.makedirs(OUTPUT_ROOT, exist_ok=True)  # 创建输出目录，如果已存在则不报错

# 是否先划出外部留出集（CV 之前）
USE_EXTERNAL_HOLDOUT = True  # 是否在交叉验证前先划分外部留出测试集，True表示启用
EXTERNAL_TEST_SIZE = 0.1  # 外部留出集的比例，0.1表示10%的数据作为最终测试集

# ==============
# 工具函数
# ==============

def log_message(message: str):
    """带时间戳的日志输出"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 获取当前时间并格式化为字符串：年-月-日 时:分:秒
    print(f"[{ts}] {message}")  # 打印带时间戳的日志消息
    sys.stdout.flush()  # 立即刷新标准输出缓冲区，确保日志实时显示

def parse_biofeat(biofeat_str):
    """将 BioFeat 的 JSON 字符串解析为一维 numpy 数组"""
    try:  # 尝试解析BioFeat字符串
        arr = np.array(json.loads(biofeat_str))  # 将JSON字符串解析为Python对象，然后转换为numpy数组
        return arr.flatten()  # 将数组展平为一维数组并返回
    except Exception as e:  # 如果解析失败，捕获异常
        log_message(f"BioFeat 解析失败: {str(e)}; 样例片段: {str(biofeat_str)[:120]} ...")  # 记录错误日志，显示前120个字符
        return None  # 返回None表示解析失败

def evaluate_model(y_true, y_pred, y_pred_proba):
    """稳健计算各类评估指标（异常时返回 NaN）"""
    metrics = {}  # 初始化空字典，用于存储所有评估指标
    try:  # 尝试计算ROC AUC
        metrics['ROC AUC'] = roc_auc_score(y_true, y_pred_proba)  # 计算ROC曲线下面积，使用真实标签和预测概率
    except Exception:  # 如果计算失败（如只有一个类别）
        metrics['ROC AUC'] = np.nan  # 返回NaN值
    try:  # 尝试计算PR AUC
        metrics['PR AUC'] = average_precision_score(y_true, y_pred_proba)  # 计算精确率-召回率曲线下面积
    except Exception:  # 如果计算失败
        metrics['PR AUC'] = np.nan  # 返回NaN值
    try:  # 尝试计算准确率
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)  # 计算分类准确率：正确预测数/总样本数
    except Exception:  # 如果计算失败
        metrics['Accuracy'] = np.nan  # 返回NaN值
    try:  # 尝试计算F1分数
        metrics['F1'] = f1_score(y_true, y_pred)  # 计算F1分数：精确率和召回率的调和平均数
    except Exception:  # 如果计算失败
        metrics['F1'] = np.nan  # 返回NaN值
    try:  # 尝试计算MCC
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)  # 计算马修斯相关系数，适用于不平衡数据集
    except Exception:  # 如果计算失败
        metrics['MCC'] = np.nan  # 返回NaN值
    try:  # 尝试计算平衡准确率
        metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)  # 计算平衡准确率：各类别召回率的平均值
    except Exception:  # 如果计算失败
        metrics['Balanced Accuracy'] = np.nan  # 返回NaN值
    try:  # 尝试计算敏感度（召回率）
        metrics['Sensitivity'] = recall_score(y_true, y_pred, pos_label=1)  # 计算敏感度：真正例/(真正例+假负例)
    except Exception:  # 如果计算失败
        metrics['Sensitivity'] = np.nan  # 返回NaN值
    try:  # 尝试计算特异度
        metrics['Specificity'] = recall_score(y_true, y_pred, pos_label=0)  # 计算特异度：真负例/(真负例+假正例)
    except Exception:  # 如果计算失败
        metrics['Specificity'] = np.nan  # 返回NaN值
    return metrics  # 返回包含所有评估指标的字典

def select_features_mi(X, y, k):
    """互信息特征选择：在训练折上拟合选择器，避免信息泄漏"""
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))  # 创建互信息特征选择器，选择k个最佳特征（不超过特征总数）
    X_sel = selector.fit_transform(X, y)  # 在训练数据上拟合选择器并转换特征，返回选择后的特征矩阵
    return X_sel, selector  # 返回选择后的特征矩阵和选择器对象（用于后续转换测试集）

# =====
# 主流程
# =====

def main():
    try:  # 使用try-except捕获所有异常
        log_message("启动：TabPFN + 特征选择 + 分层10折交叉验证")  # 记录程序启动日志

        # 读取 CSV
        log_message("读取输入 CSV ...")  # 记录读取数据日志
        df_labels = pd.read_csv(FEATURES_PATH)  # 读取特征文件：含标签与Smiles列
        df_feats  = pd.read_csv(DATA_PATH)  # 读取数据文件：含Smiles与BioFeat列

        # 依据 'Smiles' 合并
        log_message("按 'Smiles' 进行合并 ...")  # 记录合并数据日志
        df = pd.merge(df_labels, df_feats, on='Smiles', how='inner')  # 基于Smiles列进行内连接合并
        log_message(f"合并后形状: {df.shape}")  # 记录合并后的数据形状（行数，列数）

        # 解析 BioFeat -> Fingerprints（数值特征向量）
        log_message("解析 BioFeat 到数值数组 ...")  # 记录解析特征日志
        if 'BioFeat' not in df.columns:  # 检查是否存在BioFeat列
            raise KeyError("合并结果中不包含 'BioFeat' 列，请确认 DATA_PATH 文件中含有该列。")  # 抛出键错误异常
        df['Fingerprints'] = df['BioFeat'].apply(parse_biofeat)  # 对每行的BioFeat应用解析函数，转换为数值数组
        df = df.dropna(subset=['Fingerprints'])  # 删除Fingerprints列为空的行

        # 组装特征矩阵 X
        X_all = np.vstack(df['Fingerprints'].values)  # 将所有Fingerprints数组垂直堆叠成特征矩阵

        # 识别任务列（标签列）：排除特征及辅助列
        exclude_cols = {'Smiles', 'BioFeat', 'Fingerprints'}  # 定义需要排除的列名集合
        task_cols = [c for c in df.columns if c not in exclude_cols]  # 列表推导式：获取所有非排除列作为任务列（ADR标签列）

        if len(task_cols) == 0:  # 如果没有找到任务列
            raise ValueError("未找到标签列（任务列）。请确认 FEATURES_PATH 文件中包含 ADR 标签列。")  # 抛出值错误异常

        # 为数据集创建输出目录
        dataset_name = os.path.splitext(os.path.basename(DATA_PATH))[0]  # 从数据路径提取文件名（不含扩展名）作为数据集名称
        dataset_dir = os.path.join(OUTPUT_ROOT, dataset_name)  # 拼接输出根目录和数据集名称
        os.makedirs(dataset_dir, exist_ok=True)  # 创建数据集输出目录，如果已存在则不报错

        # 全局汇总日志
        log_path = os.path.join(OUTPUT_ROOT, f'cv_summary_{dataset_name}.txt')  # 拼接全局汇总日志文件路径
        mode = 'a' if os.path.exists(log_path) and START_TASK_ID > 1 else 'w'  # 如果日志文件存在且不是从第一个任务开始，则追加模式，否则覆盖模式
        with open(log_path, mode) as logf:  # 打开日志文件
            if mode == 'a':  # 如果是追加模式（断点续跑）
                logf.write("\n" + "=" * 80 + "\n")  # 写入分隔线
                logf.write(f"恢复执行，从任务 {START_TASK_ID} 开始\n")  # 写入恢复执行信息
                logf.write("=" * 80 + "\n\n")  # 写入分隔线和空行

            # 遍历每个任务（ADR 端点）
            for task_idx, task_name in enumerate(task_cols, start=1):  # 遍历所有任务列，索引从1开始
                if task_idx < START_TASK_ID:  # 如果当前任务索引小于起始任务索引
                    continue  # 跳过该任务（用于断点续跑）

                log_message(f"处理任务 {task_idx}/{len(task_cols)}: {task_name}")  # 记录当前处理的任务信息
                task_dir = os.path.join(dataset_dir, f'task_{task_idx:04d}_{task_name}')  # 创建任务输出目录路径（4位数字编号+任务名）
                os.makedirs(task_dir, exist_ok=True)  # 创建任务输出目录

                # 取出该任务的标签，并过滤 NaN
                y_all = df[task_name].values  # 获取该任务的所有标签值（numpy数组）
                mask = ~np.isnan(y_all)  # 创建布尔掩码：非NaN的位置为True
                X = X_all[mask]  # 使用掩码过滤特征矩阵，只保留有标签的样本
                y = y_all[mask]  # 使用掩码过滤标签，只保留非NaN的标签

                # 有效性检查（至少包含两个类别）
                if len(y) == 0 or len(np.unique(y)) < 2:  # 如果标签为空或只有一个类别
                    log_message(f"任务 {task_idx}（{task_name}）无效（样本为空或仅单一类别），跳过。")  # 记录跳过信息
                    with open(os.path.join(dataset_dir, 'last_completed_task.txt'), 'w') as ck:  # 打开断点记录文件
                        ck.write(str(task_idx))  # 写入当前任务索引（即使跳过也记录）
                    continue  # 跳过该任务

                # 可选：外部留出集（先划分出一部分样本，剩余做 CV）
                if USE_EXTERNAL_HOLDOUT:  # 如果启用外部留出集
                    X_train_dev, X_holdout, y_train_dev, y_holdout = train_test_split(  # 划分训练开发集和留出测试集
                        X, y,  # 输入特征和标签
                        test_size=EXTERNAL_TEST_SIZE,  # 测试集比例
                        stratify=y,  # 分层划分，保持各类别比例
                        random_state=RANDOM_STATE  # 随机种子，确保可复现
                    )
                else:  # 如果不使用外部留出集
                    X_train_dev, y_train_dev = X, y  # 训练开发集就是全部数据
                    X_holdout, y_holdout = None, None  # 留出集设为None

                # 分层 10 折交叉验证
                kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)  # 创建分层K折交叉验证对象
                fold_metrics_list = []  # 初始化每折指标列表
                fold_pred_records = []  # 初始化每折预测记录列表

                for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_train_dev, y_train_dev), start=1):  # 遍历每一折，索引从1开始
                    X_tr = X_train_dev[tr_idx]  # 根据训练索引获取训练集特征
                    y_tr = y_train_dev[tr_idx]  # 根据训练索引获取训练集标签
                    X_te = X_train_dev[te_idx]  # 根据测试索引获取测试集特征
                    y_te = y_train_dev[te_idx]  # 根据测试索引获取测试集标签

                    # 特征选择：在训练折上拟合选择器，避免信息泄漏
                    if USE_MI_SELECTION:  # 如果启用互信息特征选择
                        X_tr_sel, selector = select_features_mi(X_tr, y_tr, k=N_FEATURES_MI)  # 在训练集上选择特征并拟合选择器
                        X_te_sel = selector.transform(X_te)  # 使用训练集上拟合的选择器转换测试集特征（避免信息泄漏）
                    else:  # 如果不使用特征选择
                        X_tr_sel, X_te_sel = X_tr, X_te  # 直接使用原始特征
                        selector = None  # 选择器设为None

                    # 训练 AutoTabPFN 模型
                    model = AutoTabPFNClassifier(device=TABPFN_DEVICE, max_time=TABPFN_MAX_TIME)  # 创建AutoTabPFN分类器，设置设备和最大训练时间
                    model.fit(X_tr_sel, y_tr)  # 在训练集上训练模型

                    # 验证折预测
                    y_te_proba = model.predict_proba(X_te_sel)[:, 1]  # 获取测试集的正类预测概率（第二列）
                    y_te_pred = (y_te_proba > 0.5).astype(int)  # 根据0.5阈值将概率转换为二分类预测（0或1）

                    # 计算评估指标
                    m = evaluate_model(y_te, y_te_pred, y_te_proba)  # 计算所有评估指标
                    fold_metrics_list.append({'fold': fold_id, **m})  # 将折编号和指标合并后添加到列表

                    # 记录每折预测
                    fold_pred_records.append(pd.DataFrame({  # 创建包含预测结果的DataFrame
                        'fold': fold_id,  # 折编号
                        'y_true': y_te,  # 真实标签
                        'y_pred': y_te_pred,  # 预测标签
                        'y_proba': y_te_proba  # 预测概率
                    }))

                    # 可选：保存每折特征选择器（体积小）；不建议保存模型（体积较大）
                    if selector is not None:  # 如果使用了特征选择器
                        joblib.dump(selector, os.path.join(task_dir, f'selector_fold{fold_id}.joblib'))  # 保存特征选择器到文件

                    log_message(f"任务 {task_idx} | 折 {fold_id} 指标: {m}")  # 记录当前折的评估指标

                # 保存每折指标
                cv_df = pd.DataFrame(fold_metrics_list)  # 将指标列表转换为DataFrame
                cv_df.to_csv(os.path.join(task_dir, 'cv_fold_metrics.csv'), index=False)  # 保存为CSV文件，不包含行索引

                # 保存每折预测
                preds_df = pd.concat(fold_pred_records, ignore_index=True)  # 合并所有折的预测结果，重置索引
                preds_df.to_csv(os.path.join(task_dir, 'cv_fold_predictions.csv'), index=False)  # 保存为CSV文件

                # 写入任务级汇总（均值 ± 标准差）
                with open(os.path.join(task_dir, 'cv_summary.txt'), 'w') as tf:  # 打开任务级汇总文件
                    tf.write(f"Task: {task_name}\n")  # 写入任务名称
                    tf.write("Stratified 10-Fold CV（均值 ± 标准差）:\n")  # 写入标题
                    for metric in ['ROC AUC', 'PR AUC', 'Accuracy', 'F1', 'MCC',  # 遍历所有评估指标
                                   'Balanced Accuracy', 'Sensitivity', 'Specificity']:
                        mean_val = cv_df[metric].mean()  # 计算该指标的均值
                        std_val = cv_df[metric].std()  # 计算该指标的标准差
                        tf.write(f"- {metric}: {mean_val:.6f} ± {std_val:.6f}\n")  # 写入均值±标准差格式的结果

                # 如启用外部留出集：在全部 train_dev 上重训并评估
                if USE_EXTERNAL_HOLDOUT:  # 如果启用了外部留出集
                    if USE_MI_SELECTION:  # 如果启用了特征选择
                        X_train_sel, selector_full = select_features_mi(X_train_dev, y_train_dev, k=N_FEATURES_MI)  # 在全部训练开发集上选择特征
                        X_holdout_sel = selector_full.transform(X_holdout)  # 使用训练集上拟合的选择器转换留出集特征
                    else:  # 如果不使用特征选择
                        X_train_sel, X_holdout_sel = X_train_dev, X_holdout  # 直接使用原始特征
                        selector_full = None  # 选择器设为None

                    final_model = AutoTabPFNClassifier(device=TABPFN_DEVICE, max_time=TABPFN_MAX_TIME)  # 创建最终模型
                    final_model.fit(X_train_sel, y_train_dev)  # 在全部训练开发集上训练最终模型

                    y_hold_proba = final_model.predict_proba(X_holdout_sel)[:, 1]  # 获取留出集的正类预测概率
                    y_hold_pred = (y_hold_proba > 0.5).astype(int)  # 根据0.5阈值转换为二分类预测
                    hold_metrics = evaluate_model(y_holdout, y_hold_pred, y_hold_proba)  # 计算留出集评估指标

                    # 保存外部留出集结果
                    pd.DataFrame([hold_metrics]).to_csv(os.path.join(task_dir, 'external_holdout_metrics.csv'), index=False)  # 保存留出集指标
                    pd.DataFrame({  # 创建包含预测结果的DataFrame
                        'y_true': y_holdout,  # 真实标签
                        'y_pred': y_hold_pred,  # 预测标签
                        'y_proba': y_hold_proba  # 预测概率
                    }).to_csv(os.path.join(task_dir, 'external_holdout_predictions.csv'), index=False)  # 保存留出集预测结果

                # 写入全局汇总日志
                with open(log_path, 'a' if mode == 'a' else 'w') as lg:  # 打开全局汇总日志文件（追加或覆盖模式）
                    if mode != 'a':  # 如果是覆盖模式（首次写入）
                        # 首次写入
                        lg.write("")  # 写入空字符串（初始化文件）
                        mode_local = 'a'  # 设置局部模式为追加（用于后续写入）
                    lg.write(f"\nTask {task_idx} ({task_name}) CV metrics (mean ± std):\n")  # 写入任务信息
                    for metric in ['ROC AUC', 'PR AUC', 'Accuracy', 'F1', 'MCC',  # 遍历所有评估指标
                                   'Balanced Accuracy', 'Sensitivity', 'Specificity']:
                        mean_val = cv_df[metric].mean()  # 计算均值
                        std_val = cv_df[metric].std()  # 计算标准差
                        lg.write(f"{metric}: {mean_val:.6f} ± {std_val:.6f}\n")  # 写入均值±标准差
                    lg.write("-" * 60 + "\n")  # 写入分隔线

                # 记录断点
                with open(os.path.join(dataset_dir, 'last_completed_task.txt'), 'w') as ck:  # 打开断点记录文件
                    ck.write(str(task_idx))  # 写入当前完成的任务索引（用于断点续跑）

                log_message(f"完成任务 {task_idx}: {task_name}")  # 记录任务完成日志

        log_message("全部任务处理完成。")  # 记录所有任务完成日志

    except Exception as e:  # 捕获所有异常
        log_message(f"程序异常终止: {str(e)}")  # 记录异常信息
        import traceback  # 导入traceback模块，用于获取详细异常堆栈
        log_message(traceback.format_exc())  # 记录完整的异常堆栈信息
        sys.exit(1)  # 以错误状态码退出程序

if __name__ == "__main__":  # 如果脚本被直接运行（而非被导入）
    main()  # 调用主函数
