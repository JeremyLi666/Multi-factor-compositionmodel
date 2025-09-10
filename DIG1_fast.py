import os
import random
import time
import asyncio
from datetime import datetime

from machine_lib import *
from config import *
from fields import *

from rich.console import Console

console = Console()

def small_first_order_factory(fields, ops_set, per_field_min=1, per_field_max=3, per_field_target=10):
    """
    每个字段尽量生成 per_field_target(默认10) 个表达式，优先组合：
      - ts_* 窗口: 5, 11, 22, 66, 120, 252
      - group_* 分组: sector/industry/cap分桶
      - 基础算子: rank, zscore, signed_power
    自动跳过当前环境缺失的算子；去重；避免与字段字符串冲突。
    """
    import random
    alpha_set = []

    # 可用集合（基于已过滤的全局算子）
    ts_avail = [op for op in [
        "ts_zscore", "ts_mean", "ts_std_dev", "ts_delta", "ts_rank", "ts_sum"
    ] if op in ts_ops]
    basic_avail = [op for op in [
        "rank", "zscore", "signed_power"
    ] if op in basic_ops or op == "signed_power"]
    group_avail = [op for op in [
        "group_neutralize", "group_rank", "group_zscore"
    ] if 'group_ops' in globals() and op in group_ops]

    ts_windows = [5, 11, 22, 66, 120, 252]
    group_choices = [
        "sector",
        "industry",
        "bucket(rank(cap), range='0.1, 1, 0.1')"
    ]

    for field in fields:
        # 针对每个字段收集候选，控制数量
        seen = set()
        per_field_exprs = []

        # 1) 基础算子优先
        for op in basic_avail:
            if len(per_field_exprs) >= per_field_target:
                break
            if op == "signed_power":
                expr = f"signed_power({field}, 2)"
            else:
                expr = f"{op}({field})"
            if expr not in seen and op not in field:
                seen.add(expr)
                per_field_exprs.append(expr)

        # 2) 时间序列算子（窗口多样化）
        for op in ts_avail:
            if len(per_field_exprs) >= per_field_target:
                break
            # 为每个 op 选择 1-2 个窗口
            sel_ws = random.sample(ts_windows, k=min(2, len(ts_windows)))
            for w in sel_ws:
                if len(per_field_exprs) >= per_field_target:
                    break
                expr = f"{op}({field}, {w})"
                if expr not in seen and op not in field:
                    seen.add(expr)
                    per_field_exprs.append(expr)

        # 3) 分组算子（不同分组）
        for op in group_avail:
            if len(per_field_exprs) >= per_field_target:
                break
            g = random.choice(group_choices)
            expr = f"{op}({field}, densify({g}))"
            if expr not in seen and op not in field:
                seen.add(expr)
                per_field_exprs.append(expr)

        # 如果仍不足，回退多取 ts 窗口
        i = 0
        while len(per_field_exprs) < per_field_target and ts_avail:
            op = ts_avail[i % len(ts_avail)]
            w = ts_windows[i % len(ts_windows)]
            expr = f"{op}({field}, {w})"
            if expr not in seen and op not in field:
                seen.add(expr)
                per_field_exprs.append(expr)
            i += 1

        alpha_set.extend(per_field_exprs)

    # 去重（全局）保持顺序
    g_seen = set()
    uniq = []
    for a in alpha_set:
        if a not in g_seen:
            g_seen.add(a)
            uniq.append(a)
    return uniq

@while_true_try_decorator
def run_task(dataset_id, region, delay, instrumentType, universe, n_jobs, tag=None):
    delay = int(delay)
    n_jobs = int(n_jobs)

    print(datetime.now(), "================= 回测任务启动 =================")
    print(datetime.now(), f"dataset_id:       {dataset_id}")
    print(datetime.now(), f"region:           {region}")
    print(datetime.now(), f"delay:            {delay}")
    print(datetime.now(), f"instrumentType:   {instrumentType}")
    print(datetime.now(), f"universe:         {universe}")
    print(datetime.now(), f"n_jobs:           {n_jobs}")
    print(datetime.now(), f"tag:              {tag}")
    print("================================================")

    print(datetime.now(), "开始登录...")
    s = login()
    print(datetime.now(), "登录成功，开始拉取字段...")

    group = get_datafields(s=s, dataset_id=dataset_id, region=region, delay=delay, universe=universe)
    s.close()
    
    
    if group is None or len(group) == 0:
        print(datetime.now(), "字段为空，跳过任务")
        return
    # 使用数据集特定的tag
    if tag is None:
        tag = f"{region}_{dataset_id}_fast_check"
    else:
        tag = f"{region}_{dataset_id}_fast_check"
    completed_file_path = os.path.join(RECORDS_PATH, f"{tag}_simulated_alpha_expression.txt")
    completed_alphas = read_completed_alphas_with_comments(completed_file_path)

    # 字段统计（官方原始 vs. 派生可用 vs. 本轮选用）
    total_official = len(group)
    try:
        matrix_cnt = int((group['type'] == "MATRIX").sum())
        vector_cnt = int((group['type'] == "VECTOR").sum())
    except Exception:
        matrix_cnt = vector_cnt = 0

    derived_fields = process_datafields(group, "matrix") + process_datafields(group, "vector")
    total_derived = len(derived_fields)

    # 使用全部可用字段参与生成（不抽样）
    pc_fields = derived_fields
    print(datetime.now(), "字段统计：")
    print(f"- 官方原始字段总数：{total_official}（MATRIX: {matrix_cnt}，VECTOR: {vector_cnt}）")
    print(f"- 处理后的可用字段数：{total_derived}")
    print(f"- 参与生成的字段数：{len(pc_fields)}（全部参与）")

    # 单层 + 全算子池，但每字段只采样 1-3 个表达式
    ops_pool = ts_ops + basic_ops
    print(datetime.now(), "开始构造表达式（单层，每个字段1-3个）...")
    raw_alpha_list = small_first_order_factory(pc_fields, ops_pool, per_field_min=3, per_field_max=5, per_field_target=10)
    print(datetime.now(), f"表达式生成完成：共 {len(raw_alpha_list)} 条")

    alpha_list = [alpha for alpha in raw_alpha_list if alpha not in completed_alphas]

    if len(alpha_list) == 0:
        print(datetime.now(), f"{tag} 所有表达式已完成，跳过")
        return

    print(datetime.now(), "表达式统计：")
    print(f"- 生成表达式总数：{len(raw_alpha_list)}")
    print(f"- 待回测表达式数：{len(alpha_list)}（已剔除历史重复）")

    random.shuffle(alpha_list)

    region_list = [(region, universe)] * len(alpha_list)
    decay_list = [random.randint(0, 10) for _ in alpha_list]
    delay_list = [delay] * len(alpha_list)

    neut = 'SUBINDUSTRY'

    print(datetime.now(), f"开始提交回测：共 {len(alpha_list)} 条表达式")
    asyncio.run(simulate_multiple_tasks(
        alpha_list, region_list, decay_list, delay_list,
        tag, neut, [], n=n_jobs
    ))
    # 回测完成后，保存本次提交的表达式清单（与成功结果文件区分开）
    submitted_file_path = os.path.join(RECORDS_PATH, f"{tag}_submitted_alpha_expression.txt")
    try:
        save_completed_alphas(submitted_file_path, alpha_list)
        print(datetime.now(), f"已保存提交表达式清单至：{submitted_file_path}")
    except Exception as e:
        print(datetime.now(), f"保存提交清单失败：{e}")
    print(datetime.now(), "回测提交完成。")

def plan_dataset(dataset_id, region, delay, instrumentType, universe, n_jobs, tag=None):
    """
    预统计：返回该数据集的字段与表达式规模（不提交）。
    输出：{
        'dataset_id', 'tag', 'official_total', 'matrix_cnt', 'vector_cnt',
        'derived_total', 'selected_fields', 'generated_total', 'pending_total'
    }
    """
    # 登录并取字段
    print(datetime.now(), f"准备统计数据集 {dataset_id} ...")
    s = login()
    group = get_datafields(s=s, dataset_id=dataset_id, region=region, delay=delay, universe=universe)
    s.close()
    if group is None or len(group) == 0:
        return {
            'dataset_id': dataset_id, 'tag': tag,
            'official_total': 0, 'matrix_cnt': 0, 'vector_cnt': 0,
            'derived_total': 0, 'selected_fields': 0,
            'generated_total': 0, 'pending_total': 0
        }

    # 使用数据集特定的tag
    if tag is None:
        tag_local = f"{region}_{dataset_id}_fast_check"
    else:
        tag_local = f"{region}_{dataset_id}_fast_check"
    completed_file_path = os.path.join(RECORDS_PATH, f"{tag_local}_simulated_alpha_expression.txt")
    completed_alphas = read_completed_alphas_with_comments(completed_file_path)

    official_total = len(group)
    try:
        matrix_cnt = int((group['type'] == "MATRIX").sum())
        vector_cnt = int((group['type'] == "VECTOR").sum())
    except Exception:
        matrix_cnt = vector_cnt = 0

    derived_fields = process_datafields(group, "matrix") + process_datafields(group, "vector")
    derived_total = len(derived_fields)
    pc_fields = derived_fields  # 全部参与

    ops_pool = ts_ops + basic_ops
    raw_alpha_list = small_first_order_factory(pc_fields, ops_pool, per_field_min=1, per_field_max=3)
    alpha_list = [alpha for alpha in raw_alpha_list if alpha not in completed_alphas]

    return {
        'dataset_id': dataset_id,
        'tag': tag_local,
        'official_total': official_total,
        'matrix_cnt': matrix_cnt,
        'vector_cnt': vector_cnt,
        'derived_total': derived_total,
        'selected_fields': len(pc_fields),
        'generated_total': len(raw_alpha_list),
        'pending_total': len(alpha_list),
    }

def read_completed_alphas_with_comments(filepath):
    """
    从指定文件中读取已经完成的alpha表达式，跳过注释行（以#开头）
    """
    completed_alphas = set()
    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释行
                if line and not line.startswith('#'):
                    completed_alphas.add(line)
    except FileNotFoundError:
        print(datetime.now(), f"File not found: {filepath}")
    return completed_alphas

def generate_dataset_records(dataset_ids, region, delay, instrumentType, universe, n_jobs, tag=None):
    """
    为每个数据集生成记录文件，在第一行显示待回测因子数量
    """
    print(datetime.now(), "================= 生成数据集记录文件 =================")
    
    for ds in dataset_ids:
        # 获取数据集统计信息
        st = plan_dataset(ds, region, delay, instrumentType, universe, n_jobs, tag)
        
        # 生成数据集特定的tag
        if tag is None:
            dataset_tag = f"{region}_{ds}_fast_check"
        else:
            dataset_tag = f"{region}_{ds}_fast_check"
        
        # 生成记录文件路径
        record_file_path = os.path.join(RECORDS_PATH, f"{dataset_tag}_simulated_alpha_expression.txt")
        
        # 如果文件不存在，创建文件并写入统计信息
        if not os.path.exists(record_file_path):
            try:
                with open(record_file_path, 'w', encoding='utf-8') as f:
                    # 第一行写入待回测因子数量
                    f.write(f"# 待回测因子数量: {st['pending_total']}\n")
                    f.write(f"# 数据集ID: {st['dataset_id']}\n")
                    f.write(f"# 官方原始字段: {st['official_total']}（MATRIX: {st['matrix_cnt']}, VECTOR: {st['vector_cnt']}）\n")
                    f.write(f"# 处理后字段: {st['derived_total']}\n")
                    f.write(f"# 生成表达式总数: {st['generated_total']}\n")
                    f.write(f"# 创建时间: {datetime.now()}\n")
                    f.write("# " + "="*50 + "\n")
                
                print(datetime.now(), f"已创建数据集 {ds} 的记录文件: {record_file_path}")
                print(f"   - 待回测因子数量: {st['pending_total']}")
            except Exception as e:
                print(datetime.now(), f"创建数据集 {ds} 记录文件失败: {e}")
        else:
            print(datetime.now(), f"数据集 {ds} 的记录文件已存在: {record_file_path}")
    
    print(datetime.now(), "================= 数据集记录文件生成完成 =================")

def run_multi_datasets(dataset_ids, region, delay, instrumentType, universe, n_jobs, tag=None):
    """
    依次遍历多个 dataset_id，复用相同的 region/delay/universe 等参数。
    智能启动：检测每个数据集的待回测数量，从第一个有待回测的数据集开始。
    如果 tag=None，将自动按内部规则使用带有 dataset_id 的 tag。
    """
    # 预统计与汇总
    print(datetime.now(), "开始多数据集预统计...")
    stats = []
    total_official = total_pending = 0
    
    # 检测每个数据集的待回测数量
    start_index = 0
    for i, ds in enumerate(dataset_ids):
        st = plan_dataset(ds, region, delay, instrumentType, universe, n_jobs, tag)
        stats.append(st)
        total_official += st['official_total']
        total_pending += st['pending_total']
        print(f"- 数据集 {st['dataset_id']}: 官方原始字段 {st['official_total']}（M:{st['matrix_cnt']}/V:{st['vector_cnt']}），待回测表达式 {st['pending_total']}")
        
        # 如果当前数据集有待回测表达式，且还没有找到起始点，则设置为起始点
        if st['pending_total'] > 0 and start_index == 0:
            start_index = i
            print(datetime.now(), f"找到起始数据集：{ds}（索引 {i}），待回测表达式 {st['pending_total']} 个")
    
    print(f"合计：官方原始字段 {total_official}，待回测表达式 {total_pending}")
    
    if total_pending == 0:
        print(datetime.now(), "所有数据集的表达式都已完成，程序结束")
        return
    
    # 生成所有数据集的记录文件
    generate_dataset_records(dataset_ids, region, delay, instrumentType, universe, n_jobs, tag)
    
    print(datetime.now(), f"从数据集 {dataset_ids[start_index]} 开始处理（索引 {start_index}）")
    
    # 从检测到的起始点开始串行跑各数据集
    for i in range(start_index, len(dataset_ids)):
        ds = dataset_ids[i]
        print(datetime.now(), f"================= 开始数据集 {ds}（第 {i+1}/{len(dataset_ids)} 个）=================")
        try:
            run_task(
                dataset_id=ds,
                region=region,
                delay=delay,
                instrumentType=instrumentType,
                universe=universe,
                n_jobs=n_jobs,
                tag=tag
            )
        except Exception as e:
            print(datetime.now(), f"数据集 {ds} 运行失败：{e}")
        print(datetime.now(), f"================= 结束数据集 {ds} =================")
    
    print(datetime.now(), "所有数据集处理完成")

# ========== 启动入口 ==========
if __name__ == '__main__':
    # 按顺序遍历多个数据集 ID（字符串或数字皆可）
    datasets_to_run = ["AAA", "BBB", "CCC"]
    #上官网自己选n个datasetid输入进去
    #技巧：
    #1.优先选coverage高的，其次看这个数据集已经有多少个alpha
    #2.选已有因子多的数据集(不绝对，有时候太多，比如analyst4,会导致prod correlation爆炸）
    #3.可以输入多个数据集id，考虑分散性以保证每天都能稳定出货 自己领悟如何权衡不同地区的dataset之前怎么组合
    
    print(datetime.now(), "================= 因子挖掘机器启动 =================")
    print(datetime.now(), f"数据集列表：{datasets_to_run}")
    print(datetime.now(), "程序将按顺序处理每个数据集，完成后自动结束")
    print("================================================")
    
    run_multi_datasets(
        dataset_ids=datasets_to_run,
        region="CHN",
        delay=1,
        instrumentType="EQUITY",
        universe="TOP2000U",
        n_jobs=6,
        tag=None  # 使用None让每个数据集生成自己的tag
    )
    
    print(datetime.now(), "================= 因子挖掘机器完成 =================")