
import time
from config import *
from machine_lib import *

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
            task = simulate_single(current_session_manager, alpha, region, name, neut, decay,
                                   delay, stone_bag, tags, semaphore)
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

    region = "USA"
    universe = "TOP3000"
    delay = 1
    instrumentType = "EQUITY"
    step1_tag = "analyst4_usa_1step"
    step2_tag = 'analyst4_usa_2step'

    fo_tracker = get_alphas("2024-10-07", "2025-12-31",
                            0.75, 0.5,
                            100, 100,
                            region, universe, delay, instrumentType,
                            500, "track", tag=step1_tag)
    print(len(fo_tracker['next']))
    print(len(fo_tracker['decay']))
    fo_layer = transform(fo_tracker['next'] + fo_tracker['decay'])

    print(fo_layer)
    so_alpha_dict = defaultdict(list)
    for expr, decay in fo_layer:
        for alpha in get_group_second_order_factory([expr], group_ops, region):
            so_alpha_dict[region].append((alpha,decay))

    for key, value in so_alpha_dict.items():
        print("%s : %d"%(key, len(value)))

    # 读取已完成的alpha表达式
    completed_alphas = read_completed_alphas(f'records/{step2_tag}_simulated_alpha_expression.txt')

    second_list = so_alpha_dict[region]
    # 排除已完成的alpha表达式
    second_list = [alpha_decay for alpha_decay in second_list if alpha_decay[0] not in completed_alphas]

    if len(second_list) == 0:
        print('暂时没有满足条件的一阶段因子，请你继续运行digging_1step.')
        time.sleep(600)
        exit()

    print(len(second_list), 'Waiting for simulation...')

    alpha_list = [alpha_decay[0] for alpha_decay in second_list]
    decay_list = [alpha_decay[1] for alpha_decay in second_list]
    region_list = [('USA', 'TOP3000')] * len(alpha_list)  # 扩展 region_list
    delay_list = [1] * len(alpha_list)  # 扩展 region_list

    stone_bag = []

    # 执行异步模拟，并控制并发数量为3
    asyncio.run(simulate_multiple_alphas(alpha_list, region_list, decay_list, delay_list,
                                         step2_tag, 'SUBINDUSTRY',
                                         stone_bag, n_jobs=3))