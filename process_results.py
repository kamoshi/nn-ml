import pickle
from typing import final


with open("results_simple.pkl", "rb") as f:
    simple = pickle.load(f)

with open("results_conv.pkl", "rb") as f:
    conv = pickle.load(f)

with open("results_pool.pkl", "rb") as f:
    pool = pickle.load(f)


def to_average(results: list[tuple[dict[str, list[int]], list[float]]]):
    accu, loss = {}, {}
    final_accu, final_loss = [], []
    for _data, _score in results:
        for i, (a, l) in enumerate(zip(_data["accuracy"], _data["loss"])):
            accu.setdefault(i, []).append(a)
            loss.setdefault(i, []).append(l)
        final_loss.append(_score[0])
        final_accu.append(_score[1])
    
    for k, v in accu.items():
        accu[k] = sum(v) / len(v)
    for k, v in loss.items():
        loss[k] = sum(v) / len(v)
    
    return accu, loss, sum(final_accu) / len(final_accu), sum(final_loss) / len(final_loss)


simple = to_average(simple)
conv = to_average(conv)
pool = to_average(pool)


with open("processed_results.csv", "w") as f:
    for name, result in [("simple", simple), ("conv", conv), ("pool", pool)]:
        f.write(f"== {name} ==\n")
        f.write(f"Final accuracy: {result[2]:.4f}\n")
        f.write(f"Final loss: {result[3]:.4f}\n")
        f.write("Epoch, Accuracy, Loss\n")
        for i in range(20):
            f.write(f"{i+1}, {result[0][i]:.4f}, {result[1][i]:.4f}\n")
        f.write("\n")
