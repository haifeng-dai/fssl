"""先距离对比、再用冻结分类器训练服务端类别锚点。"""

from test_anchor_proto import run_experiment

if __name__ == "__main__":
    run_experiment("hybrid")
