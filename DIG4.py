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

    print(datetime.now(),f"=================您的四阶段配置信息==================")
    print(datetime.now(),f"dataset_id: {dataset_id}")
    print(datetime.now(),f"region: {region}")
    print(datetime.now(),f"delay: {delay}")
    print(datetime.now(),f"instrumentType: {instrumentType}")
    print(datetime.now(),f"universe: {universe}")
    print(datetime.now(),f"n_jobs: {n_jobs}")
    print(datetime.now(),f"=============================================")
    time.sleep(2)

    # 生成step3的tag
    tag = f"{region}_{delay}_{instrumentType}_{universe}_{dataset_id}_step3"
    step3_tag_list = [tag]

    for step3_tag in step3_tag_list:
        if 'ILLIQUID' in step3_tag:
            region, delay, instrumentType, ILLIQUID_tag, universe, dataset_id, step_num = step3_tag.split('_')
            universe = f'{ILLIQUID_tag}_{universe}'
        else:
            region, delay, instrumentType, universe, dataset_id, step_num = step3_tag.split('_')

        delay = int(delay)

        if delay == 1:
            sharpe_step4_th = 1.5
            fitness_step4_th = 0.85
        elif delay == 0:
            sharpe_step4_th = 2.75
            fitness_step4_th = 1.5
        else:
            print(datetime.now(),"delay must be 0 or 1.")
            continue

        step4_tag = step3_tag.replace('_step3', '_step4')

        to_tracker = get_alphas("2024-10-07", "2029-12-31",
                                sharpe_step4_th, fitness_step4_th,
                                100, 100,
                                region, universe, delay, instrumentType,
                                500, "track", tag=step3_tag)

        to_layer = prune(to_tracker['next'] + to_tracker['decay'],
                         dataset_id, 3)

        if len(to_layer) == 0:
            print(datetime.now(),f'tag: {step3_tag} 暂时没有满足条件的三阶段因子，请你继续运行digging_consultant_3step.py.')
            continue

        fh_alpha_list = []
        for expr, decay in to_layer:
            for alpha in template_factory(expr, region):
                fh_alpha_list.append((alpha, decay))

        print(datetime.now(),f"Total expression for simulation: {len(fh_alpha_list)}")

        completed_alphas = read_completed_alphas(f'records/{step4_tag}_simulated_alpha_expression.txt')
        raw_alpha_list = fh_alpha_list
        alpha_list = [item for item in raw_alpha_list if item[0] not in completed_alphas]

        if not alpha_list:
            print(datetime.now(), f"{step4_tag} 已完成全部 {len(raw_alpha_list)} 个表达式")
            continue

        print(datetime.now(), f"{step4_tag} 模拟进度: {len(raw_alpha_list) - len(alpha_list)} / {len(raw_alpha_list)}")

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
                                                step4_tag, neut,
                                                [], n=n_jobs))

    print(datetime.now(),"All done. Sleep 600s...")
    time.sleep(600)
    print(datetime.now(),"Wake up.")

if __name__ == '__main__':
    run_task("analyst4", "USA", 1, "EQUITY", "TOP3000", 4)
