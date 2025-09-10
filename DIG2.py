from 增强machine_lib import *
from config import *
import asyncio
import aiofiles
import time
from collections import defaultdict
from datetime import datetime

@while_true_try_decorator
def run_task(dataset_id, region, delay, instrumentType, universe, n_jobs, tag=None):
    delay = int(delay)
    n_jobs = int(n_jobs)

    print(datetime.now(), f"================= Digging Consultant STEP2 ==================")
    print(datetime.now(), f"dataset_id:       {dataset_id}")
    print(datetime.now(), f"region:           {region}")
    print(datetime.now(), f"delay:            {delay}")
    print(datetime.now(), f"instrumentType:   {instrumentType}")
    print(datetime.now(), f"universe:         {universe}")
    print(datetime.now(), f"n_jobs:           {n_jobs}")
    print(datetime.now(), f"===========================================================")
    time.sleep(2)

    tag = f"{region}_{delay}_{instrumentType}_{universe}_{dataset_id}_step1"
    step1_tag_list = [tag]

    for step1_tag in step1_tag_list:
        if 'ILLIQUID' in step1_tag:
            region, delay, instrumentType, ILLIQUID_tag, universe, dataset_id, step_num = step1_tag.split('_')
            universe = f'{ILLIQUID_tag}_{universe}'
        else:
            region, delay, instrumentType, universe, dataset_id, step_num = step1_tag.split('_')

        delay = int(delay)

        if delay == 1:
            sharpe_step2_th = 1.0
            fitness_step2_th = 0.5
        elif delay == 0:
            sharpe_step2_th = 2.0
            fitness_step2_th = 1.0
        else:
            print(datetime.now(),"delay must be 0 or 1.")
            continue

        step2_tag = step1_tag.replace('_step1', '_step2')

        fo_tracker = get_alphas("2024-10-07", "2029-12-31",
                                sharpe_step2_th, fitness_step2_th,
                                100, 100,
                                region, universe, delay, instrumentType,
                                500, "track", tag=step1_tag)

        fo_layer = prune(fo_tracker['next'] + fo_tracker['decay'],
                         dataset_id, 3)

        if len(fo_layer) == 0:
            print(datetime.now(),'暂时没有满足条件的一阶段因子，请你继续运行digging_consultant_1step.py.')
            continue

        print(datetime.now(),f'qualified expression: {len(fo_layer)}')

        so_alpha_list = []

        for expr, decay in fo_layer:
            so_alpha_list.append((expr, decay))
            for alpha in get_group_second_order_factory([expr], group_ops):
                so_alpha_list.append((alpha, decay))

        # 读取已完成的alpha表达式
        completed_alphas = read_completed_alphas(f'records/{step2_tag}_simulated_alpha_expression.txt')

        raw_alpha_list = so_alpha_list
        # 排除已完成的alpha表达式
        alpha_list = [alpha_decay for alpha_decay in raw_alpha_list if alpha_decay[0] not in completed_alphas]
        if len(alpha_list) == 0:
            print(datetime.now(),f"已经完成了{len(completed_alphas)}个alpha表达式，一共有{len(raw_alpha_list)}个alpha表达式")
            print(datetime.now(),f"{step2_tag} is done.")
            continue
        print(datetime.now(),"{}progress: {}/{}".format(step2_tag, len(raw_alpha_list) - len(alpha_list), len(raw_alpha_list)))

        grouped_dict = defaultdict(list)
        for alpha, decay in alpha_list:
            grouped_dict[decay].append(alpha)

        for decay in grouped_dict:
            alpha_list = grouped_dict[decay]
            decay_list = [decay] * len(alpha_list)
            delay_list = [delay] * len(alpha_list)
            region_list = [(region, universe)] * len(alpha_list)

            if region in ['USA', 'EUR', 'ASI', 'CHN']:
                neut = 'SUBINDUSTRY'
            elif region in ['GLB']:
                neut = "SUBINDUSTRY"
            elif region in ['AMR']:
                neut = "SUBINDUSTRY"
            else:
                neut = 'SUBINDUSTRY'

            asyncio.run(simulate_multiple_tasks(alpha_list, region_list, decay_list, delay_list,
                                                step2_tag, neut,
                                                [], n=n_jobs))

    print(datetime.now(),"All done. Sleep 600s..")
    time.sleep(600)
    print(datetime.now(),"Wake up.")

if __name__ == '__main__':
    run_task("analyst4", "EUR", 1, "EQUITY", "TOP2500", 4)
