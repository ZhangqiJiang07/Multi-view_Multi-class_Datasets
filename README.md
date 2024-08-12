# Multi-view Multi-class Datasets
This repo contains some benchmarks for evaluating Multi-view Multi-class machine learning algorithms.

## 📄 Statistics of Datasets
📢 More information can be found in [[Google Sheets](https://docs.google.com/spreadsheets/d/15jSJqDot4-LPiX_GYSHJDkndXZGjYqOBm6LXl_j57U4/edit?usp=sharing) | [Tencent Docs](https://docs.qq.com/sheet/DY1lRV0VCcEJHcm5s?tab=BB08J2)].

|No. | Datasets |#Samples | #Classes | #Views| Tag | Reference |
|----|----|:----:|:----:|:----:|----|----|
|1|100Leaves|1,600|100|3|  | _Plant leaf classification using probabilistic integration of shape, texture and margin features_ |
|2|Caltech101-7|1,474|7|6|  | _Large-scale multi-view spectral clustering via bipartite graph_ |
|3|Caltech101-20|2,386|20|6|  | _Deep Incomplete Multi-View Learning Network with Insufficient Label Information_ |
|4|Caltech101|9,144|102|6|  | _Binary Multi-View Clustering_ |
|5|Deep Caltech101|8,677|101|2|  | _Trusted Multi-View Classification_ |
|6|Caltech256|30,607|257|3|  |_Auto-weighted Multi-view Clustering for Large-scale Data_|
|7|Deep AWA_2views|10,158|50|2|  |Deep Partial Multi-View Learning|
|8|Reuters_2views|18,758|6|2|  |Multi-view Spectral Clustering Network|
|9|


✨ The data format is as follows:
```
xxx.mat
|---gnd: matrix, double, start from 1, (sample_number, 1).
|---X: cell, (1, view_num)
|---|---X{i}: matrix, double, (sample_number, feature_dimension).
```




## 🔥 Update



## 🌋 Modality Evaluation
We simply adopt the SVM as a baseline to evaluate the contribution of each modality for the classification task.






## 📊 Label Distribution



## Acknowledgements
Most of the data were downloaded from these sites, for which we are very grateful:

[1] [https://github.com/liujiyuan13/mvdata](https://github.com/liujiyuan13/mvdata)

[2] [https://github.com/wangsiwei2010/large_scale_multi-view_clustering_datasets](https://github.com/wangsiwei2010/large_scale_multi-view_clustering_datasets)




