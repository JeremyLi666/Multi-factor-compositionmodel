import datetime
import os
import time
import pandas as pd

from config import RECORDS_PATH
from machine_lib import login

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 100)

def submit_alpha(s, alpha_id):
    submit_url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/submit"

    attempts = 0
    while attempts < 5:
        attempts += 1
        print(f"Attempt {attempts} to submit {alpha_id}.")
        # 第一轮提交
        while True:
            res = s.post(submit_url)
            if res.status_code == 201:
                print(f"Alpha {alpha_id} POST Status 201. Start submitting...")
                break
            elif res.status_code == 400:
                print(f"Alpha {alpha_id} POST Status {res.status_code}.")
                print(f"Alpha {alpha_id} Already POST.")
                print(res.content)
                break
            elif res.status_code == 403:
                print(f"Alpha {alpha_id} POST Status {res.status_code}.")
                print(pd.DataFrame(res.json()["is"]["checks"])[['name', 'value', 'result']])
                return res.status_code
            else:
                print(f"Alpha {alpha_id} POST Status {res.status_code}.")
                print(res.content)
                time.sleep(3)

        # 第二轮提交
        count = 0
        s_t = datetime.datetime.now()
        while True:
            res = s.get(submit_url)
            if res.status_code == 200:
                retry = res.headers.get('Retry-After', 0)
                if retry:
                    count += 1
                    time.sleep(float(retry))
                    if count % 75 == 0:
                        print(f"Alpha {alpha_id} GET Status 200. Waiting... {datetime.datetime.now()-s_t}.")
                else:
                    print(f"Alpha {alpha_id} was submitted successfully.")
                    return res.status_code
            elif res.status_code == 403:
                print(f"Alpha {alpha_id} GET Status {res.status_code}.")
                print(f"Alpha {alpha_id} submit failed. Need Improvement.")
                print(pd.DataFrame(res.json()["is"]["checks"])[['name', 'value', 'result']])
                return res.status_code
            elif res.status_code == 404:
                print(f"Alpha {alpha_id} GET Status {res.status_code}.")
                print(f"Alpha {alpha_id} submit failed. Time Out.")
                break
            else:
                print(f"Alpha {alpha_id} GET Status {res.status_code}.")
                print(f"Alpha {alpha_id} submit failed. Time Out.")
                print(res.headers)
                print(res.content)
                break

    return 404


if __name__ == '__main__':
    s = login()

    submitable_alpha_file = os.path.join(RECORDS_PATH, 'submitable_alpha.csv')
    df = pd.read_csv(submitable_alpha_file)
    df['pyramids'] = df['checks'].apply(lambda x: next(([y['name'] for y in item['pyramids']] for item in eval(x) if item['name'] == 'MATCHES_PYRAMID'), None))
    df = df.sort_values(by=['fitness', 'sharpe'], ascending=[True, True]).reset_index(drop=True)
    id_list = df['id'].tolist()

    # 这里面替换你的alpha_id
    submittable_alphas = id_list
    for alpha_id in submittable_alphas:
        status_code = submit_alpha(s, alpha_id)
        if status_code == 200 or status_code == 403:
            df = pd.read_csv(submitable_alpha_file)
            df = df[df['id'] != alpha_id]
            df.to_csv(submitable_alpha_file, index=False)

