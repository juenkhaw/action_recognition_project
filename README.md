# Backward Propagation with R(2+1)D Action Recognition RGB Stream Network
### Running

Please run on `visualize_init.py` with one of the arguments set provided below:

**Visualize with GPU**

`[dataset_path] [pretrained_model_path] [test_video_folder_name] [test_video_label]`

**Visualize with CPU**

Please add `-dv cpu` into the arguments.

**Visualize on different endpoint**

Please specify an endpoint by adding `-endp [endpoint]` into the arguments.

### Dataset Path

Please input path to the directory containing the rgb frames and optical flows as `dataset_path` 

Example: `../dataset/UCF-101`

**RGB Frame**

The directory outline of RGB frames:

`[dataset_path]/[dataset]_jpegs_256/jpegs_256/[video_name]`

**Dataset Source**
https://github.com/feichtenhofer/twostreamfusion

### Output

The output file should contain series of gradient maps, positive saliency and negative saliency for each respective RGB frames following its temporal timeline:

![Walking with dog sample](https://github.com/juenkhaw/action_recognition_project/blob/vis-module/v_WalkingWithDog_g01_c01.png)

### Arguments

**Mandatory**

- `dataset_path` path to directories of rgb frames
- `model_path` path to pretrained model
- `test_video` video folder name to be visualized
- `test_label` label index for the testing video

**Optional**

- `-dv` device chosen to perform training
- `-endp` module block where forprop to and backprop from
- `-filter` filter chosen to be visualised
- `-nframe` frame number for each testing clip
- `-v1` activate to allow reporting of activation shape after each forward propagation

Please refer to the source code for more details on arguments.

### Implementation References

- A Closer Look at Spatiotemporal Convolutions for Action Recognition https://arxiv.org/abs/1711.11248
- https://github.com/facebookresearch/VMZ
- https://github.com/irhumshafkat/R2Plus1D-PyTorch
- https://github.com/utkuozbulak/pytorch-cnn-visualizations
