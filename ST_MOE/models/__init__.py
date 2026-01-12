from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .BNT import BrainNetworkTransformer, BrainNetworkTransformer_dfc # 导入BrainNetworkTransformer_dFC,dFC专家模型


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    # model_name是这个了 BrainNetworkTransformer
    return eval(config.model.name)(config).cuda()
