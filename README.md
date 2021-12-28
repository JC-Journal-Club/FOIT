# FOIT: Fast Online Instance Transfer for Improved EEG Emotion Recognition

Code of paper FOIT: Fast Online Instance Transfer for Improved EEG Emotion Recognition

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
