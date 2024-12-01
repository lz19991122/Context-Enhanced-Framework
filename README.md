# Context-Enhanced-Framework-for-Medical-Image-Report-Generation-Using-Multimodal-Contexts


**Datasets**

We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For IU X-Ray, you can download the dataset from [here](https://openi.nlm.nih.gov/) and then put the files in iu_xray.

For MIMIC-CXR, you can download the dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) and then put the files in mimic.


**Requirements**

System: Ubuntu 18.04
GPU: RTX 2080 TI (11G)
PyTorch==1.7.1, Python3.8
[Spacy](https://spacy.io/)
[nlg-eval](https://github.com/Maluuba/nlg-eval)

**Usage**

Run the code:
```bash
python train.py
```

**Test**

Pleause use the [nlg-eval](https://github.com/Maluuba/nlg-eval) and the [chexpert](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt/chexpert).
