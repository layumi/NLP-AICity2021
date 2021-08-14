# NLP-AICity2021

[[Poster]](https://github.com/layumi/NLP-AICity2021/blob/main/doc/08-poster.pdf)
[[Pdf]](https://github.com/layumi/NLP-AICity2021/blob/main/doc/CVPRW2021_NLP_AICity.pdf)
[[Video]](https://www.bilibili.com/video/BV1yK4y1G7zr)

We have two codebases. For the final submission, we conduct the feature ensemble, where features are from two codebases.

Our main code is at here: https://github.com/ShuaiBai623/AIC2021-T5-CLV 

### Download Data: 
1 – Merge all packages into one:
```bash
$ zip -FF AIC21_Track5_NL_Retrieval.zip --out AIC21_Track5_NL_Retrieval_full.zip
```
2 – Delete the smaller packages to save disk space (optional):
```bash
$ rm AIC21_Track5_NL_Retrieval.z*
```
3 – Unzip the merged package:
```bash
$ mkdir AIC21_Track5_NL_Retrieval
$ cd AIC21_Track5_NL_Retrieval
$ unzip -FF ../AIC21_Track5_NL_Retrieval_full.zip
```

### Prepare Data: 
```
python data/extract_vdo_frms.py
```

