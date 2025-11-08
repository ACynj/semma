version_0.0.1: 对逆关系的嵌入进行了改进(动态阈值未改进)
version_0.0.2: 使用PPR改进了动态阈值（逆关系未改进）
version_0.0.3: 对逆关系的嵌入进行了改进 + 使用PPR改进了动态阈值

ARE-V1:阈值0.8 增强：0.1
ARE-V2：阈值 0.75 增强： 0.05
ARE-V3：阈值 0.8 增强： 0.05
ARE-V4：阈值 0.85 增强： 0.025
ARE-V5：阈值 0.8 增强： 0.025

ARE-V3_vip：阈值 0.8 增强： 0.05,step:200000
semma_vip:按照论文参数训练

ARE-V6：阈值 0.75 增强： 0.025
ARE-V7：阈值 0.7 增强： 0.025
ARE-V8：阈值 0.65 增强： 0.025

python script/pretrain.py -c config/transductive/pretrain_semma.yaml --gpus [0] --ckpt /T20030104/ynj/semma/ckpts/semma.pth

bash run_commands.sh