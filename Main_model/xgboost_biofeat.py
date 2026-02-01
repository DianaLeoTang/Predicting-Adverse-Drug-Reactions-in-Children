import os  # 导入操作系统接口模块，用于文件路径操作和目录创建
import sys  # 导入系统相关参数和函数，用于程序退出和标准输出刷新
import json  # 导入JSON模块，用于解析和保存JSON格式数据
import joblib  # 导入joblib模块，用于保存和加载机器学习模型
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据处理和CSV文件读写
import optuna  # 导入Optuna库，用于超参数优化
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

import xgboost as xgb  # 导入XGBoost库，用于梯度提升树模型

# =========================
# 可配置参数
# =========================

FEATURES_PATH = '/public/home/tianyao/biosignature/features.csv'  # 特征文件路径：包含标签列（多个ADR端点）与'Smiles'列
DATA_PATH     = '/public/home/tianyao/xgboost/433_labeled_results_with_smiles.csv'  # 数据文件路径：包含'Smiles'与'BioFeat'（需解析成数值向量）

# 从第几个任务开始（1-based），用于断点续跑
START_TASK_ID = 1  # 任务起始索引，用于从指定任务开始继续运行（断点续跑功能）

# 分层 K 折交叉验证设置
N_SPLITS = 10  # 交叉验证的折数，设置为10折交叉验证
SHUFFLE = True  # 是否在划分前打乱数据，True表示打乱
RANDOM_STATE = 42  # 随机数种子，确保结果可复现

# Optuna 设置
N_TRIALS = 100  # Optuna超参数搜索的试验次数，共进行100次超参数组合尝试
USE_GPU = False  # 是否使用GPU加速XGBoost训练，False表示使用CPU
OPTUNA_TIMEOUT = 7200  # 单个任务的超参搜索时间限制（秒），7200秒=2小时，None表示无限制
# Optuna 优化目标，0.6*ROC_AUC + 0.4*PR_AUC
OPTUNA_OBJECTIVE_WEIGHTS = {  # Optuna优化目标的权重配置字典
    'roc_auc': 0.6,  # ROC AUC的权重为0.6
    'pr_auc': 0.4  # PR AUC的权重为0.4，两者加权组合作为优化目标
}

# 输出目录
OUTPUT_ROOT = 'xgb_optuna_cv_biofeat'  # 输出根目录名称
os.makedirs(OUTPUT_ROOT, exist_ok=True)  # 创建输出目录，如果已存在则不报错

# 是否先划出外部留出集（CV 之前）
USE_EXTERNAL_HOLDOUT = True  # 是否在交叉验证前先划分外部留出测试集
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

def objective(trial, X, y):
    """Optuna 优化目标函数，基于分层 10 折交叉验证"""
    # 定义超参数搜索空间
    params = {  # 创建XGBoost参数字典
        'objective': 'binary:logistic',  # 二分类任务，使用逻辑回归目标函数
        'eval_metric': 'auc',  # 评估指标使用AUC
        'tree_method': 'gpu_hist' if USE_GPU else 'hist',  # 树构建方法：GPU加速或CPU直方图方法
        'verbosity': 0,  # 设置日志详细程度为0（不输出训练过程）
        
        # 超参数搜索范围
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # 学习率：对数尺度搜索0.01-0.3
        'max_depth': trial.suggest_int('max_depth', 3, 9),  # 树的最大深度：整数搜索3-9
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # 叶子节点最小样本权重和：整数搜索1-10
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # 样本采样比例：浮点数搜索0.6-1.0
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # 特征采样比例：浮点数搜索0.6-1.0
        'gamma': trial.suggest_float('gamma', 0, 10),  # 最小损失减少量：浮点数搜索0-10
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  # L1正则化系数：对数尺度搜索
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),  # L2正则化系数：对数尺度搜索
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # 树的数量：整数搜索100-500
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),  # 正样本权重缩放：浮点数搜索1.0-10.0
    }
    
    # 分层 10 折交叉验证
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)  # 创建分层K折交叉验证对象
    roc_scores = []  # 初始化ROC AUC分数列表
    pr_scores = []  # 初始化PR AUC分数列表
    
    for train_idx, val_idx in kf.split(X, y):  # 遍历每一折的训练和验证索引
        X_tr, X_val = X[train_idx], X[val_idx]  # 根据索引划分训练集和验证集特征
        y_tr, y_val = y[train_idx], y[val_idx]  # 根据索引划分训练集和验证集标签
        
        # 检查验证集中是否至少有两个类别
        if len(np.unique(y_val)) < 2:  # 如果验证集只有一个类别
            continue  # 跳过这一折，无法计算AUC
            
        # 训练模型
        model = xgb.XGBClassifier(**params)  # 使用当前超参数创建XGBoost分类器
        model.fit(  # 训练模型
            X_tr, y_tr,  # 训练集特征和标签
            eval_set=[(X_val, y_val)],  # 验证集，用于早停
            early_stopping_rounds=25,  # 早停轮数：验证集性能25轮不提升则停止
            verbose=False  # 不输出训练过程
        )
        
        # 预测并评估
        y_val_proba = model.predict_proba(X_val)[:, 1]  # 获取验证集的正类预测概率（第二列）
        try:  # 尝试计算评估指标
            roc_auc = roc_auc_score(y_val, y_val_proba)  # 计算ROC AUC
            pr_auc = average_precision_score(y_val, y_val_proba)  # 计算PR AUC
            roc_scores.append(roc_auc)  # 将ROC AUC添加到列表
            pr_scores.append(pr_auc)  # 将PR AUC添加到列表
        except Exception:  # 如果计算失败
            pass  # 跳过这一折的评估
    
    # 如果所有折都无法评估，返回一个很低的分数
    if len(roc_scores) == 0:  # 如果没有任何有效的评估分数
        return -1.0  # 返回-1.0作为惩罚分数
    
    # 计算加权平均分数
    mean_roc_auc = np.mean(roc_scores)  # 计算所有折的ROC AUC平均值
    mean_pr_auc = np.mean(pr_scores)  # 计算所有折的PR AUC平均值
    
    # 加权组合分数作为优化目标
    weighted_score = (  # 计算加权组合分数
        OPTUNA_OBJECTIVE_WEIGHTS['roc_auc'] * mean_roc_auc +  # ROC AUC权重×平均值
        OPTUNA_OBJECTIVE_WEIGHTS['pr_auc'] * mean_pr_auc  # PR AUC权重×平均值
    )
    
    return weighted_score  # 返回加权分数，Optuna会最大化这个分数

# =====
# 主流程
# =====

def main():
    try:  # 使用try-except捕获所有异常
        log_message("启动：XGBoost + Optuna 超参优化 + 分层10折交叉验证（BioFeat）")  # 记录程序启动日志

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

            # 遍历每个任务
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

                # ============ Optuna 超参优化 ============
                log_message(f"启动 Optuna 超参优化 (任务 {task_idx}: {task_name}), 共 {N_TRIALS} 次尝试...")  # 记录超参优化开始日志
                
                study = optuna.create_study(  # 创建Optuna优化研究对象
                    direction='maximize',  # 优化方向：最大化目标函数值
                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)  # 使用TPE（Tree-structured Parzen Estimator）采样器，设置随机种子
                )
                
                study.optimize(  # 执行超参数优化
                    lambda trial: objective(trial, X_train_dev, y_train_dev),  # 优化目标函数：传入trial对象和训练数据
                    n_trials=N_TRIALS,  # 优化试验次数
                    timeout=OPTUNA_TIMEOUT  # 超时时间限制（秒）
                )
                
                best_params = study.best_params  # 获取最佳超参数组合
                best_value = study.best_value  # 获取最佳目标函数值
                
                log_message(f"Optuna 优化完成: 最佳分数 = {best_value:.6f}")  # 记录最佳分数
                log_message(f"最佳超参: {best_params}")  # 记录最佳超参数
                
                # 保存优化结果
                with open(os.path.join(task_dir, 'optuna_best_params.json'), 'w') as f:  # 打开JSON文件用于保存最佳参数
                    json.dump({  # 将字典写入JSON文件
                        'best_params': best_params,  # 最佳超参数
                        'best_score': best_value,  # 最佳分数
                        'n_trials': N_TRIALS,  # 优化试验次数
                        'weights': OPTUNA_OBJECTIVE_WEIGHTS  # 优化目标权重
                    }, f, indent=2)  # 缩进2个空格，使JSON格式更易读

                # 把最佳超参加入到 XGB 配置
                best_xgb_params = {  # 创建XGBoost参数字典
                    'objective': 'binary:logistic',  # 二分类逻辑回归目标
                    'eval_metric': 'auc',  # 评估指标
                    'tree_method': 'gpu_hist' if USE_GPU else 'hist',  # 树构建方法
                    'verbosity': 0,  # 日志详细程度
                    **best_params  # 展开最佳超参数字典，合并到参数字典中
                }

                # ============ 分层 10 折交叉验证 ============
                log_message(f"使用最佳超参进行 {N_SPLITS} 折交叉验证评估...")  # 记录交叉验证开始日志
                
                kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)  # 创建分层K折交叉验证对象
                fold_metrics_list = []  # 初始化每折指标列表
                fold_pred_records = []  # 初始化每折预测记录列表

                for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_train_dev, y_train_dev), start=1):  # 遍历每一折，索引从1开始
                    X_tr = X_train_dev[tr_idx]  # 根据训练索引获取训练集特征
                    y_tr = y_train_dev[tr_idx]  # 根据训练索引获取训练集标签
                    X_te = X_train_dev[te_idx]  # 根据测试索引获取测试集特征
                    y_te = y_train_dev[te_idx]  # 根据测试索引获取测试集标签
                    
                    # 使用最佳超参训练模型
                    model = xgb.XGBClassifier(**best_xgb_params)  # 使用最佳超参数创建XGBoost分类器
                    
                    # 使用早停避免过拟合
                    model.fit(  # 训练模型
                        X_tr, y_tr,  # 训练集特征和标签
                        eval_set=[(X_te, y_te)],  # 验证集，用于早停判断
                        verbose=False,  # 不输出训练过程
                        early_stopping_rounds=25  # 早停轮数：25轮无提升则停止
                    )

                    # 验证折预测
                    y_te_proba = model.predict_proba(X_te)[:, 1]  # 获取测试集的正类预测概率（第二列）
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
                    tf.write(f"超参优化: Optuna ({N_TRIALS} 次尝试), 最佳分数: {best_value:.6f}\n")  # 写入超参优化信息
                    tf.write(f"最佳超参: {json.dumps(best_params, indent=2)}\n\n")  # 写入最佳超参数（JSON格式）
                    tf.write(f"Stratified 10-Fold CV 评估结果（均值 ± 标准差）:\n")  # 写入标题
                    for metric in ['ROC AUC', 'PR AUC', 'Accuracy', 'F1', 'MCC',  # 遍历所有评估指标
                                   'Balanced Accuracy', 'Sensitivity', 'Specificity']:
                        mean_val = cv_df[metric].mean()  # 计算该指标的均值
                        std_val = cv_df[metric].std()  # 计算该指标的标准差
                        tf.write(f"- {metric}: {mean_val:.6f} ± {std_val:.6f}\n")  # 写入均值±标准差格式的结果

                if USE_EXTERNAL_HOLDOUT:  # 如果启用了外部留出集
                    # 在全部训练开发集上训练最终模型
                    final_model = xgb.XGBClassifier(**best_xgb_params)  # 使用最佳超参数创建最终模型
                    final_model.fit(X_train_dev, y_train_dev)  # 在全部训练开发集上训练模型

                    # 在外部留出测试集上评估
                    y_hold_proba = final_model.predict_proba(X_holdout)[:, 1]  # 获取留出集的正类预测概率
                    y_hold_pred = (y_hold_proba > 0.5).astype(int)  # 根据0.5阈值转换为二分类预测
                    hold_metrics = evaluate_model(y_holdout, y_hold_pred, y_hold_proba)  # 计算留出集评估指标

                    # 保存外部留出集结果
                    pd.DataFrame([hold_metrics]).to_csv(os.path.join(task_dir, 'external_holdout_metrics.csv'), index=False)  # 保存留出集指标
                    pd.DataFrame({  # 创建包含预测结果的DataFrame
                        'y_true': y_holdout,  # 真实标签
                        'y_pred': y_hold_pred,  # 预测标签
                        'y_proba': y_hold_proba  # 预测概率
                    }).to_csv(os.path.join(task_dir, 'external_holdout_predictions.csv'), index=False)  # 保存留出集预测结果
                    
                    # 保存最终模型
                    model_path = os.path.join(task_dir, 'final_model.joblib')  # 拼接模型保存路径
                    joblib.dump(final_model, model_path)  # 使用joblib保存模型到文件
                    log_message(f"保存最终模型到: {model_path}")  # 记录模型保存日志

                # 写入全局汇总日志
                with open(log_path, 'a' if mode == 'a' else 'w') as lg:  # 打开全局汇总日志文件（追加或覆盖模式）
                    if mode != 'a':  # 如果是覆盖模式（首次写入）
                        lg.write("")  # 写入空字符串（初始化文件）
                    lg.write(f"\nTask {task_idx} ({task_name}):\n")  # 写入任务信息
                    lg.write(f"Optuna 最佳分数: {best_value:.6f}\n")  # 写入最佳分数
                    lg.write(f"CV metrics (mean ± std):\n")  # 写入标题
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
