# TALoS: Enhancing Semantic Scene Completion via Test-time Adaptation on the Line of Sight

This repository contains the official PyTorch implementation of the paper "TALoS: Enhancing Semantic Scene Completion via Test-time Adaptation on the Line of Sight" paper (NeurIPS 2024) by [Hyun-Kurl Jang*](https://blue-531.github.io/
) , [Jihun Kim*](https://jihun1998.github.io/
) and [Hyeokjun Kweon*](https://sangrockeg.github.io/
).
## News
(* denotes equal contribution.)
<ul>
  <li> TALoS is accepted at NeurIPS 2024 🎉.</li>
  <li> Official code and Paper will be released soon! </li>
</ul>

## Introduction
<img src='/assets/qual_kitti.png' width="1000" height="429"/>

Our main idea is simple yet effective: 
**an observation made at one moment could serve as supervision for the SSC prediction at another moment.** 
While traveling through an environment, an autonomous vehicle can continuously observe the overall scene structures, including objects that were previously occluded (or will be occluded later), which are concrete guidances for the adaptation of scene completion. Given the characteristics of the LiDAR sensor, an observation of a point at a specific spatial location at a specific moment confirms not only the occupation at that location itself but also the absence of obstacles along the line of sight from the sensor to that location.
The proposed method, named 
**Test-time Adaptation via Line of Sight (TALoS)**
, is designed to explicitly leverage these characteristics, obtaining self-supervision for geometric completion.
Additionally, we extend the TALoS framework for semantic recognition, another key goal of SSC, by collecting the reliable regions only among the semantic segmentation results predicted at each moment.
Further, to leverage valuable future information that is not accessible at the time of the current update, we devise a novel dual optimization scheme involving the model gradually updating across the temporal dimension.
## Installation

- PyTorch >= 1.10 
- pyyaml
- Cython
- tqdm
- numba
- Numpy-indexed
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [spconv](https://github.com/tyjiang1997/spconv1.0) (tested with spconv==1.0 and cuda==11.3)



## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
├── model_load_dir
    ├──pretrained.pth
└── dataset/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        │   └── voxels/ 
        |       ├── 000000.bin
        |       ├── 000000.label
        |       ├── 000000.invalid
        |       ├── 000000.occluded
        |       ├── 000001.bin
        |       ├── 000001.label
        |       ├── 000001.invalid
        |       ├── 000001.occluded
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

## Test-Time Adaptation
1. Download the pre-trained models and put them in ```./model_load_dir```. [[link]](https://drive.google.com/file/d/12jYauPbVodnSA-faBjFucUNgxeGU0pmP/view?usp=drive_link)
2. (Optional) Download pre-trained model results and put them in ```./experiments/baseline``` for comparison. [[link]](https://drive.google.com/file/d/1gt65t7hkdnnax2v7BALgUsunTaGHRVkh/view?usp=drive_link)
3. Generate predictions on the Dataset.

### Validation set
```
python run_tta_val.py --do_adapt --do_cont --use_los --use_pgt 
```
### Test set
```
python run_tta_test.py --do_adapt --do_cont --use_los --use_pgt --sq_num={sequence number} 
```
## Evaluation
To evaluate test sequences in SemanticKITTI, you should submit the generated predictions to [link](https://codalab.lisn.upsaclay.fr/competitions/7170).
After generate predictions, prepare your submission in the designated format, as described in the competition page.
Use the validation script from the [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) to ensure that the folder structure and number of label files in the zip file is correct.

<img src='/assets/benchmark.png' width="500" height="198"/>


## Acknowledgements
We thanks for the open source project [SCPNet](https://github.com/SCPNet/Codes-for-SCPNet).
