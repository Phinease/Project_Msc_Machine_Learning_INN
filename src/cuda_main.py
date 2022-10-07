import subprocess as s
import time
from tqdm import tqdm
import os
import torch


def clear():
    os.system('clear')


n_process, process, n_sample = 3, [], 100
for i in range(n_process):
    p = s.Popen(['python', 'src/cuda_run.py', str(i), str(n_process), str(n_sample)])
    print("Start process", i)
    process.append(p)
    time.sleep(1)

try:
    wait_time = 60
    print("Initializing .....")
    for i in range(wait_time):
        time.sleep(1)
        # clear()

    with open('runs/run0.txt', 'r') as fifo:
        n_data = int(fifo.readlines()[0])

    with tqdm(total=n_sample) as pbar:
        temp = [0] * n_process
        while True:
            for i, p in enumerate(process):
                if p.poll() == 0:
                    process.remove(p)

                with open('runs/run' + str(i) + '.txt', 'r') as fifo:
                    temp2 = eval(fifo.readlines()[-1])
                    pbar.update(temp2 - temp[i])
                    temp[i] = temp2

            time.sleep(0.1)
            if not process:
                break

    output = torch.tensor([])
    for i in range(n_process):
        t = torch.load('tensors/tensor' + str(i) + '.pt')
        output = torch.cat([output, t], dim=1)
    print(output.shape)
    torch.save(output, 'tensors/output.pt')
except KeyboardInterrupt:
    print("User stop")
except Exception as e:
    raise e
finally:
    for p in process:
        p.kill()
    print("End")
