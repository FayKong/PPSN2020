# Improving Imbalanced Classification by Anomaly Detection
This repository contains the code for introducing two additional attributes to imbalanced datasets and thus improving the imbalanced classification. 

## Introduction
The provided .py file is for the convenience of reproducing the main results of our paper [Improving Imbalanced Classification by Anomaly Detection](https://link.springer.com/chapter/10.1007/978-3-030-58112-1_35) (Kong, Kowalczyk, Menzel & Bäck, 2020). The code is implemented in Python 3.0 and the packages required for implementation are given in the next section.

The data used in this paper contains two parts, 2D chess data and imbalanced benchmark datasets. The codes to experiment with the 2D chess dataset are given in folder `2D chess`. `chess_data_generate.py` in this folder is for generating the 2D chess data used in the paper. The codes to experiment with benchmark datasets are given in folder `Imbalanced Benchmark` and the benchmark datasets are given in folder `data`. `*_four_types.py` files in folder `2D chess` and `Imbalanced Benchmark` provide the code to identify the four types of samples, and `*_LOF.py` files provide the code to calculate the LOF score for the given datasets. `*_added_*.py` files provide the code to introduce the two additional attributes and `*_added_resampling_*.py` files provide the code to resample the datasets with two additional attributes. The experimental results are given in folder `results`. In the following, we will decribe the detailed technical requiremnets and how to run our code step by step.

## Requirements

Scikit-Learn and Numpy software packages are used to obtain the two proposed additional attributes (LOF score and four types of samples). Scikit-Learn software package is also used for performing stratified cross-validation and classification. Imbalanced Learn software package provides the resampling techniques to resample the imbalanced datasets. Matplotlib is used to visualize the 2d chess dataset in different scenarios. The four required libraries can be installed by running pip install -r requirements.txt from the main directory via the command line.

| Packages | Description |
| :----------- | :------------ | 
| Imbalanced Learn | For implementing resampling techniques, e.g. SMOTE, ADASYN etc. | 
| Scikit-Learn | For calculating LOF score, performing stratified CV and classification. |
| Numpy | For efficiently dealing with data. |
| Matplotlib | For plotting.|

The introductions on how to run our code step by step will be given in the following sections.

## 1. Calculating the Two Proposed Additional Attributes

The first step in our experimental setup is to calculate the two additional attributes for every dataset. Running the `*_four_types.py` file in both '2D chess' and 'Imbalanced Benchmark' folders can achieve the first introduced additional attribute 'four types of samples', while running `*_LOF.py` in both folders can achieve the second additional attribute 'LOF score'.

## 2. Considering Different Combinations

After introducing the two additional attributes, we consider different scenarios of resampling techniques and whether to add additional attributes. This can be achieved by running `*_DT/SVM.py`, `*_added_*.py`, `*_added_resampling_*.py` files in both '2D chess' and 'Imbalanced Benchmark' folders. The experimental results are shown in folder `results`.

## 3. Visualization on 2D Chess Dataset

The `chess_LOF_plot.py` file is used to achieve the figure in our paper.

# How to Cite
## paper Reference
Kong, J., Kowalczyk, W., Menzel, S. and Bäck, T., 2020, September. Improving imbalanced classification by anomaly detection. In International Conference on Parallel Problem Solving from Nature (pp. 512-523). Springer, Cham.

## BibTex Reference
```
@inproceedings{kong2020improving,
  title={Improving imbalanced classification by anomaly detection},
  author={Kong, Jiawen and Kowalczyk, Wojtek and Menzel, Stefan and B{\"a}ck, Thomas},
  booktitle={International Conference on Parallel Problem Solving from Nature},
  pages={512--523},
  year={2020},
  organization={Springer}
}
```

# Acknowledgements
This research has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement number 766186 (ECOLE).

