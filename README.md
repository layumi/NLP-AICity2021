# NLP-AICity2021

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

