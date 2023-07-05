from vocab import Vocabulary
import evaluation_zyf

root = "./runs/runX/checkpoint/model_best.pth.tar"
#root = "/data3/zhangyf/cross_modal_retrieval/SCAN_base_ft_f30k/runs/runX/checkpoint/model_best.pth.tar"
evaluation_zyf.evalrank(root, data_path="/data3/zhangyf/cross_modal_retrieval/SCAN/data", split="testall", fold5=True)
#evaluation_zyf.evalrank(root, data_path="./data", split="test", fold5=True)


evaluation_zyf.evalrank(root, data_path="/data3/zhangyf/cross_modal_retrieval/SCAN/data", split="testall")

