# -*- coding: utf-8 -*-
"""
Vol-Norm 外层随机包裹版（仅用一个历史文件：{tag}_simulated_alpha_expression.txt）
核心：base = x / ts_std_dev(x, d)
外层（可选，随机）：ts_backfill(base,k) -> rank/zscore（二选一） -> ts_mean(...,w)
注意：分子不使用 backfill，backfill 只在最外层。
"""

import os
import re
import random
import asyncio
from datetime import datetime

from machine_lib import *      # 你的登录/提交/并发/装饰器等
from config import *           # 需要 RECORDS_PATH

# ==================== 算法参数 ====================
STD_WINDOWS = (22, 66, 120, 252)   # 分母标准差窗口
BACKFILL_WINDOWS = (5, 10, 22, 66) # 外层 ts_backfill 的窗口
SMOOTH_WINDOWS = (5, 22)           # ts_mean 的窗口
RANDOM_SEED = None                 # 固定随机性请填 int

# 外层包裹概率
P_BACKFILL_OUTER = 0.6             # 概率包 ts_backfill(base,k)
P_RANK = 0.35                      # 与 zscore 互斥
P_ZSCORE = 0.25
P_TS_MEAN = 0.35                   # 概率再套 ts_mean(...,w)

# 每个 base 展开几个随机变体（上限，实际会去重）
N_VARIANTS_PER_BASE = 2


# ==================== 工具函数 ====================
def set_seed(seed):
    if seed is not None:
        random.seed(seed)

def extract_field_names(group):
    """
    稳健抽字段名。优先列：name/field/id/code；抓不到就正则兜底。
    """
    if group is None:
        return []
    candidate_cols = ['name', 'field', 'id', 'code']
    col = None
    for c in candidate_cols:
        if hasattr(group, 'columns') and c in group.columns:
            col = c
            break
    if col:
        names = group[col].astype(str).tolist()
    else:
        names = []
        try:
            for _, row in group.iterrows():
                s = " ".join([str(v) for v in row.values])
                names.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s))
        except Exception:
            pass
    # 去重清洗
    uniq, seen = [], set()
    for n in names:
        n2 = n.strip()
        if n2 and n2 not in seen:
            seen.add(n2)
            uniq.append(n2)
    return uniq

def build_base_exprs(fields, std_windows=STD_WINDOWS):
    """基础模板：x / ts_std_dev(x, d)"""
    out, seen = [], set()
    for x in fields:
        for d in std_windows:
            e = f"({x}) / ts_std_dev({x}, {d})"
            if e not in seen:
                seen.add(e)
                out.append(e)
    return out

def wrap_outer_random(expr):
    """
    外层随机包裹：
      1) 以 P_BACKFILL_OUTER 概率套 ts_backfill(expr,k)
      2) rank / zscore 二选一（互斥）
      3) 以 P_TS_MEAN 概率再套 ts_mean(expr,w)
    顺序固定：backfill -> rank/zscore -> ts_mean
    """
    e = expr

    # 1) ts_backfill 外层
    if random.random() < P_BACKFILL_OUTER:
        k = random.choice(BACKFILL_WINDOWS)
        e = f"ts_backfill({e}, {k})"

    # 2) rank / zscore 互斥
    r = random.random()
    if r < P_RANK:
        e = f"rank({e})"
    elif r < P_RANK + P_ZSCORE:
        e = f"zscore({e})"

    # 3) ts_mean 平滑
    if random.random() < P_TS_MEAN:
        w = random.choice(SMOOTH_WINDOWS)
        e = f"ts_mean({e}, {w})"

    return e

def build_expressions_with_outer(fields, std_windows=STD_WINDOWS, n_variants=N_VARIANTS_PER_BASE):
    """对每个 base 生成 n_variants 条外层随机包裹变体"""
    bases = build_base_exprs(fields, std_windows)
    out, seen = [], set()
    for b in bases:
        for _ in range(n_variants):
            e = wrap_outer_random(b)
            if e not in seen:
                seen.add(e)
                out.append(e)
    return out

def read_completed(filepath):
    """读取历史已完成表达式（忽略以 # 开头注释）"""
    done = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith('#'):
                    done.add(s)
    except FileNotFoundError:
        pass
    return done


# ==================== 主流程 ====================
@while_true_try_decorator
def run_task(dataset_id, region, delay, instrumentType, universe, n_jobs, tag=None):
    delay = int(delay); n_jobs = int(n_jobs)
    print(datetime.now(), f"================= 任务启动 {dataset_id} =================")

    s = login()
    group = get_datafields(s=s, dataset_id=dataset_id, region=region, delay=delay, universe=universe)
    s.close()

    fields = extract_field_names(group)
    print(datetime.now(), f"[INFO] 抽取字段数: {len(fields)}")
    if len(fields) == 0:
        print(datetime.now(), "[WARN] 无字段可用，跳过")
        return

    # 生成表达式（base + 外层随机包裹）
    exprs = build_expressions_with_outer(fields, std_windows=STD_WINDOWS, n_variants=N_VARIANTS_PER_BASE)
    print(datetime.now(), f"[INFO] 生成表达式: {len(exprs)}")

    if tag is None:
        tag = f"{region}_{dataset_id}_volnorm_outer"

    # 单一历史文件：simulated_alpha_expression.txt
    record_path = os.path.join(RECORDS_PATH, f"{tag}_simulated_alpha_expression.txt")
    done = read_completed(record_path)
    todo = [e for e in exprs if e not in done]
    print(datetime.now(), f"[INFO] 待回测: {len(todo)}（已记录完成 {len(done)}）")

    if not todo:
        print(datetime.now(), "[TIP] 无新表达式需要回测")
        return

    # 提交
    random.shuffle(todo)
    region_list = [(region, universe)] * len(todo)
    decay_list = [random.randint(0, 10) for _ in todo]
    delay_list = [delay] * len(todo)
    neut = "SUBINDUSTRY"

    asyncio.run(simulate_multiple_tasks(
        todo, region_list, decay_list, delay_list, tag, neut, [], n=n_jobs
    ))

    # 提交即入库：只写同一个历史文件，保持与 read_completed 对齐
    try:
        os.makedirs(RECORDS_PATH, exist_ok=True)
        with open(record_path, "a", encoding="utf-8") as f:
            for e in todo:
                f.write(e + "\n")
        print(datetime.now(), f"[INFO] 已写入历史记录：{record_path}（追加 {len(todo)} 条）")
    except Exception as e:
        print(datetime.now(), f"[WARN] 写历史记录失败：{e}")


def run_multi_datasets(dataset_ids, region, delay, instrumentType, universe, n_jobs, tag=None):
    for ds in dataset_ids:
        try:
            run_task(ds, region, delay, instrumentType, universe, n_jobs, tag)
        except Exception as e:
            print(datetime.now(), f"[ERROR] 数据集 {ds} 失败：{e}")


# ========== 启动入口 ==========
if __name__ == '__main__':
    # 建议：只放 fundamental/analyst 相关数据集，如：["fundamental31", "analyst69", "analyst4", ...]
    datasets_to_run = ["analyst14"]

    print(datetime.now(), "================= Vol-Norm 因子挖掘机启动 =================")
    print(datetime.now(), f"数据集列表：{datasets_to_run}")
    print(datetime.now(), "程序将按顺序处理每个数据集，完成后自动结束")
    print("================================================")

    run_multi_datasets(
        dataset_ids=datasets_to_run,
        region="EUR",             # EUR 区域
        delay=1,
        instrumentType="EQUITY",
        universe="TOP2500",
        n_jobs=6,
        tag=None
    )

    print(datetime.now(), "================= Vol-Norm 因子挖掘机完成 =================")
