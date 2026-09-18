"""仅用冻结的 FedAvg 分类器训练服务端类别锚点。"""

from test_anchor_proto import run_experiment

if __name__ == "__main__":
    run_experiment("classifier")
