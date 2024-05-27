## Beware of Model Collapse! Fast and Stable Test-time Adaptation for Robust Question Answering

This is the official project repository for Demonstration Augmentation for Zero-shot In-context Learning (ACL 2024, Findings).

## Quick start

#### Environment

You should run the following script to install the dependencies first.

```
conda create -n DAIL python==3.10
conda activate DAIL
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers，sentence_transformers,openai
```

#### Get the results on MMLU

```
cd MMLU
python main.py --device {device} --model {model_path} --method {method}
# method should be in: (SelfICL, DAIL, zeroshot, fewshow).
```

#### Get the results on BBH

```
cd BBH
python main.py --device {device} --model {model_path} --method {method} --api_key {api_key}
# method should be in: (SelfICL, DAIL, zeroshot, fewshow).
```



