# G-LBM: Generative Low-dimensional Background Modeling
This repository is the implementation of the G-LBM model presented in the following paper preprint:
```
@article{rezaei2020g,
  title={G-LBM: Generative Low-dimensional Background Model Estimation from Video Sequences},
  author={Rezaei, Behnaz and Farnoosh, Amirreza and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2003.07335},
  year={2020}
}
```
## Requirements

This model is implemented with folowing frameworks:
* Python3.7, 
* Pytorch 1.2 
* CUDA 9.0 and 10.0

## Datasets

The following datasets are used for experiments in the paper:

BMC2012 dataset:

```
@inproceedings{vacavant2012benchmark,
  title={A benchmark dataset for outdoor foreground/background extraction},
  author={Vacavant, Antoine and Chateau, Thierry and Wilhelm, Alexis and Lequi{\`e}vre, Laurent},
  booktitle={Asian Conference on Computer Vision},
  pages={291--300},
  year={2012},
  organization={Springer}
}
...
@article{jodoin2017extensive,
  title={Extensive benchmark and survey of modeling methods for scene background initialization},
  author={Jodoin, Pierre-Marc and Maddalena, Lucia and Petrosino, Alfredo and Wang, Yi},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={11},
  pages={5244--5256},
  year={2017},
  publisher={IEEE}
}
```


## Training
1) Before running the train and test scripts, you have to save the motion masks of the videos by running the /utils/saveflows.py on the video datasets. We use [PyFlow](https://github.com/pathak22/pyflow) pipeline to compute optical flows fot generating motion masks.
2) Execute the code/train.py with proper input arguments to train the model on the dataset.


## Test
For running the trained model on eacch video execute the code/test.py by proper input arguments.

## Reference
Rezaei, Behnaz, Amirreza Farnoosh, and Sarah Ostadabbas. "G-LBM: Generative Low-dimensional Background Model Estimation from Video Sequences." arXiv preprint arXiv:2003.07335 (2020).

