import torch
import numpy as np


def error_evaluation(error_list):
    es = error_list.copy()
    es = torch.stack(es)
    es = es.view((-1))
    es = es.to('cpu').numpy()
    es.sort()
    ae = np.array(es).astype(np.float32)

    x, y, z = np.percentile(ae, [25, 50, 75])
    Mean = np.mean(ae)
    Med = np.median(ae)
    Tri = (x + 2 * y + z) / 4
    T25 = np.mean(ae[:int(0.25 * len(ae))])
    L25 = np.mean(ae[int(0.75 * len(ae)):])

    print("Mean\tMedian\tTri\tBest 25%\tWorst 25%")
    print("{:3f}\t{:3f}\t{:3f}\t{:3f}\t{:3f}".format(Mean, Med, Tri, T25, L25))
