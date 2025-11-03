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

pythonscript/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v2 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

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


=========================================================================================================
# 命令顺序

0、0.508	0.727

python script/run.py -c config/transductive/inference-fb.yaml --dataset DBpedia100k --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

1、0.159 0.287

python script/run.py -c config/transductive/inference-fb.yaml --dataset ConceptNet100k --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

2、0.159 0.233
0.207 0.274
python script/run.py -c config/transductive/inference-fb.yaml --dataset Hetionet --epochs 0 --bpe null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0]

3、0.494	0.664 

python script/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v1 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

4、0.510	0.704

pythonscript/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v2 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

5、0.498	0.658

python script/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v3 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

6、0.494	0.680

python script/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v4 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

7、0.718	0.813

python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v1 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

8、0.692	0.798

python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v2 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

9、0.452	0.599

python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v3 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

10、0.660	0.743

python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v4 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

11、0.755	0.875

python script/run.py -c config/inductive/inference.yaml --dataset NELLInductive --version v1 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

12、0.532	0.722

python script/run.py -c config/inductive/inference.yaml --dataset NELLInductive --version v2 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

13、 0.516	0.700

python script/run.py -c config/inductive/inference.yaml --dataset NELLInductive --version v3 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null


14、0.472	0.713

python script/run.py -c config/inductive/inference.yaml --dataset NELLInductive --version v4 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

15、0.299	0.454

python script/run.py -c config/inductive/inference.yaml --dataset ILPC2022 --version small --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

16、0.306	0.427

python script/run.py -c config/inductive/inference.yaml --dataset ILPC2022 --version large --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

17、 0.058	0.102

python script/run.py -c config/inductive/inference.yaml --dataset HM --version 1k --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

18、0.050	0.084

python script/run.py -c config/inductive/inference.yaml --dataset HM --version 3k --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

19、0.050	0.082

python script/run.py -c config/inductive/inference.yaml --dataset HM --version 5k --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

20、0.436	0.644

python script/run.py -c config/inductive/inference.yaml --dataset HM --version indigo --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

21、0.399	0.645

python script/run.py -c config/inductive/inference.yaml --dataset FBIngram --version 25 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

22、0.342	0.549

python script/run.py -c config/inductive/inference.yaml --dataset FBIngram --version 50 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

23、0.406	0.605

python script/run.py -c config/inductive/inference.yaml --dataset FBIngram --version 75 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

24、0.453	0.641

python script/run.py -c config/inductive/inference.yaml --dataset FBIngram --version 100 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

25、0.292	0.499

python script/run.py -c config/inductive/inference.yaml --dataset WKIngram --version 25 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

26、0.165	0.316

python script/run.py -c config/inductive/inference.yaml --dataset WKIngram --version 50 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

27、0.378	0.520

python script/run.py -c config/inductive/inference.yaml --dataset WKIngram --version 75 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

28、0.184	0.300

python script/run.py -c config/inductive/inference.yaml --dataset WKIngram --version 100 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null


29、0.342	0.539

python script/run.py -c config/inductive/inference.yaml --dataset NLIngram --version 0 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

30、0.384	0.571

python script/run.py -c config/inductive/inference.yaml --dataset NLIngram --version 25 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

31、0.394	0.562

python script/run.py -c config/inductive/inference.yaml --dataset NLIngram --version 50 --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

32、0.090	0.156
0.094 0.155
python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT2 --version org --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

33、0.233	0.355

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT2 --version sci --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

34、0.277	0.428
0.261 0.419
python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT3 --version art --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

35、0.642	0.784

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT3 --version infra --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

36、0.297	0.461

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT4 --version sci --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

37、0.599	0.725

python script/run.py -c config/inductive/inference.yaml --dataset WikiTopicsMT4 --version health --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

38、0.154	0.435

python script/run.py -c config/inductive/inference.yaml --dataset Metafam --version null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null

39、0.478	0.652

python script/run.py -c config/inductive/inference.yaml --dataset FBNELL --version null --ckpt /T20030104/ynj/semma/ckpts/en.pth --gpus [0] --epochs 0 --bpe null



































