python script/run.py -c config/transductive/inference-fb.yaml --dataset CoDExSmall --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset CoDExLarge --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset NELL995 --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset DBpedia100k --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset ConceptNet100k --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset NELL23k --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset YAGO310 --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset Hetionet --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset WDsinger --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset AristoV4 --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset FB15k237_10 --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset FB15k237_20 --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

python script/run.py -c config/transductive/inference-fb.yaml --dataset FB15k237_50 --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

=================================================================================================

python script/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v1 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v2 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v3 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v4 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v1 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v2 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v3 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v4 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NELLInductive --version v1 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NELLInductive --version v2 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NELLInductive --version v3 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NELLInductive --version v4 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset ILPC2022 --version small --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset ILPC2022 --version large --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset HM --version 1k --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset HM --version 3k --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset HM --version 5k --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset HM --version indigo --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

=======================================================================================================

python script/run.py -c config/inductive/inference.yaml --dataset FBIngram --version 25 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset FBIngram --version 50 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset FBIngram --version 75 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset FBIngram --version 100 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WKIngram --version 25 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WKIngram --version 50 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WKIngram --version 75 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WKIngram --version 100 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NLIngram --version 0 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NLIngram --version 25 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NLIngram --version 50 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NLIngram --version 75 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset NLIngram --version 100 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT1 --version tax --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT1 --version health --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT2 --version org --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT2 --version sci --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT3 --version art --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT3 --version infra --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT4 --version sci --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT4 --version health --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset Metafam --version null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

python script/run.py -c config/inductive/inference.yaml --dataset FBNELL --version null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null