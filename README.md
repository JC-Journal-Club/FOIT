# FOIT: Fast Online Instance Transfer for Improved EEG Emotion Recognition

The Electroencephalogram (EEG)-based emotion recognition is promising yet limited by the requirement of large number of training samples. Collecting substantial labeled samples in the training trails is the key to good classification performance on the test trails. This process is time-consuming and laborious. In recent years, several studies have proposed various semi-supervised learning (e.g., active learning) and transfer learning (e.g., domain adaptation, style transfer mapping) methods to reduce the requirement on training data. However, most of them are iterative methods, which need considerable training time and are unfeasible in practice. To tackle this problem, we present the Fast Online Instance Transfer (FOIT) for improved EEG emotion recognition. FOIT selects auxiliary data from historical sessions or other subjects with high confidence on prediction of the baseline model, which are then combined with the training data for classifier training. The predictions on the test trails are made by the ensemble classifier. FOIT is a one-shot method, which avoids the time-consuming iterations. Experimental results demonstrate that FOIT brings significant accuracy improvement for the three-category classification (1%-8%) on the SEED dataset and four-category classification (1%-14%) on the SEED-IV dataset in three transfer situations. The time cost for our machine over the baselines is moderate (~25s in average on two datasets in three transfer situations). To achieve the comparative accuracies, the iterative methods require much more time (~30s - ~1400s). FOIT provides a practically feasible solution to improve the generalization of emotion recognition models and allows various choices of classifiers without any constrains.

## Datasets
The dataset files (SEED and SEED-IV) can be downloaded from the [BCMI official website](https://bcmi.sjtu.edu.cn/~seed/index.html)

To facilitate data retrieval, we divided both datasets into three folders according to the sessions, the file structure of the datasets should be like:
```
eeg_feature_smooth/
    1/
    2/
    3/
ExtractedFeatures/
    1/
    2/
    3/
```


## Usage
Run `python FOIT_ultra.py`, and the results will be printed in the terminal.

## Contributing
Issues are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Citation
If you find our work useful for your research, please consider citing our paper as:

```bibtex
@inproceedings{li2020foit,
  title={FOIT: Fast Online Instance Transfer for Improved EEG Emotion Recognition},
  author={Li, Jinpeng and Chen, Hao and Cai, Ting},
  booktitle={2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={2618--2625},
  year={2020},
  organization={IEEE}
}
```

## License
