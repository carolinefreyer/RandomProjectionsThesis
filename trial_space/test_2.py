from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from tqdm import tqdm
import numpy as np
from functools import partial

def task(name, sign):
    print(f"{name} {sign} Executing our Task on Process: {os.getpid()}")
    return os.getpid()

def main():
    with ProcessPoolExecutor(max_workers=3) as executor:
        task1 = executor.submit(task, sign = "+", name="C")
        task2 = executor.submit(task, name="V", sign = "-")
        print(task1)
        print(task1.result(), task2.result())


def add_prefix(prefix, data, _):
    mean = np.mean(data)
    print("%s: %s%s" % (os.getpid(), prefix, mean))
    return mean


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    m = 20
    # with ThreadPoolExecutor(10) as pool:
    #     means = []
    #     args = (("hi",data, i) for i in range(m))
    #     for r in pool.map(lambda p: add_prefix(*p), args):
    #         means.append(r)
    #
    # print(means)
    with ProcessPoolExecutor(max_workers=6) as executor:
        args = (("m", data, b) for b in range(m))
        func = partial(add_prefix, "m", data)
        for result in tqdm(executor.map(func, range(m))):
        # for idx, result in tqdm(enumerate(executor.map(add_prefix, repeat("m"), repeat(data), range(m)))):
            print("result", result)


