import os
from machine_lib import login, async_login, simulate_single, get_datafields, process_datafields, first_order_factory, ts_ops, basic_ops
from fields import *
import time
import random
import asyncio
import nest_asyncio 
nest_asyncio.apply ()
from config import *

class SessionManager:
    def __init__(self, session, start_time, expiry_time):
        self.session = session
        self.start_time = start_time
        self.expiry_time = expiry_time

    async def refresh_session(self):
        print("Session expired, logging in again...")
        await self.session.close()
        self.session = await async_login()
        self.start_time = time.time()

async def simulate_multiple_alphas(alpha_list, region_list, decay_list, delay_list, name, neut, stone_bag=[], n_jobs=5):
    n = n_jobs
    semaphore = asyncio.Semaphore(n)
    tasks = []
    tags = [name]

    session_managers = []
    for _ in range(n):
        # 记录登录的开始时间，并为每个 session_manager 创建独立的 session
        session_start_time = time.time()
        session = await async_login()
        session_expiry_time = 3 * 60 * 60  # 3 小时
        session_manager = SessionManager(session, session_start_time, session_expiry_time)
        session_managers.append(session_manager)

    # 将任务划分成 n 份
    chunk_size = (len(alpha_list) + n - 1) // n  # 向上取整
    task_chunks = [alpha_list[i:i + chunk_size] for i in range(0, len(alpha_list), chunk_size)]
    region_chunks = [region_list[i:i + chunk_size] for i in range(0, len(region_list), chunk_size)]
    decay_chunks = [decay_list[i:i + chunk_size] for i in range(0, len(decay_list), chunk_size)]
    delay_chunks = [delay_list[i:i + chunk_size] for i in range(0, len(delay_list), chunk_size)]

    for i, (alpha_chunk, region_chunk, decay_chunk, delay_chunk) in (
            enumerate(zip(task_chunks, region_chunks, decay_chunks, delay_chunks))):
        # 获取当前 chunk 对应的 session_manager
        current_session_manager = session_managers[i]
        for alpha, region, decay, delay in zip(alpha_chunk, region_chunk, decay_chunk, delay_chunk):
            # 将任务与当前的 session_manager 关联
            task = simulate_single(current_session_manager, alpha, region, name, neut, decay, delay, stone_bag, tags, semaphore)
            tasks.append(task)

    await asyncio.gather(*tasks)

    # 关闭所有会话
    for session_manager in session_managers:
        await session_manager.session.close()


def read_completed_alphas(filepath):
    """
    从指定文件中读取已经完成的alpha表达式
    """
    completed_alphas = set()
    try:
        with open(filepath, mode='r') as f:
            for line in f:
                completed_alphas.add(line.strip())
    except FileNotFoundError:
        print(f"File {filepath} not found.")
    return completed_alphas

if __name__ == '__main__':
    # 配置区域
    dataset_id = 'model51'
    step1_tag = "MD51"

    s = login()
    df = get_datafields(s, dataset_id=dataset_id, region='USA', universe='TOP3000', delay=1)
    pc_fields = process_datafields(df, "matrix") + process_datafields(df, "vector")
    #pc_fields = recommended_fields_1  # 这个是推荐字段，可以取消注释直接使用

    first_order = first_order_factory(pc_fields, ts_ops + basic_ops)

    # 用region_dict去找到对应region和univsere作为simulation的setting
    region_dict = {"usa": ("USA", "TOP3000"), "asi": ("ASI", "MINVOL1M"), "eur": ("EUR", "TOP1200"),
                   "glb": ("GLB", "TOP3000"), "hkg": ("HKG", "TOP800"), "twn": ("TWN", "TOP500"), "jpn": ("JPN", "TOP1600"),
                   "kor": ("KOR", "TOP600"), "chn": ("CHN", "TOP2000U"), "amr": ("AMR", "TOP600")}

    # 读取已完成的alpha表达式
    completed_alphas = read_completed_alphas(f'records/{step1_tag}_simulated_alpha_expression.txt')
    # 原始alpha列表
    alpha_list = first_order
    # 排除已完成的alpha表达式
    alpha_list = [alpha for alpha in alpha_list if alpha not in completed_alphas]
    print(len(alpha_list), "Waiting for Simulate")
    # 打乱alpha列表顺序
    random.shuffle(alpha_list)

    region_list = [('USA', 'TOP3000')] * len(alpha_list)  # 扩展 region_list
    decay_list = [6] * len(alpha_list)  # 扩展 decay_list
    delay_list = [1] * len(alpha_list)  # 扩展 decay_list

    stone_bag = []

    # 执行异步模拟，并控制并发数量为3
    asyncio.run(simulate_multiple_alphas(alpha_list, region_list, decay_list, delay_list,
                                         step1_tag, 'SUBINDUSTRY',
                                         stone_bag, n_jobs=3))
