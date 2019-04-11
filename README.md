# Guided Backward Propagation with R(2+1)D Action Recognition RGB and Optical Flow Stream Network

### Pre-trained Model

Please refer to here for RGB and optical flow stream networks pre-trained on UCF-101:

https://drive.google.com/open?id=1_esJwF6I6NFtCWohN2S20b0RJGTehWG0

### Running

Please run on `visualize_init.py` with the arguments set provided below:

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

**Optical Flow**

`[dataset_path]/[dataset]_tvl1_flow/tvl1_flow/[u/v]/[video_name]`

**Dataset Source**
https://github.com/feichtenhofer/twostreamfusion

### Output

The output should contain series of gradient maps, positive saliency and negative saliency for each respective RGB frames and optical flow displacement fields following its temporal timeline:

![Walking with dog sample_rgb](https://github.com/juenkhaw/action_recognition_project/blob/vis-module/v_WalkingWithDog_g01_c01.png)

![Walking with dog sample_flow_u](https://github.com/juenkhaw/action_recognition_project/blob/vis-module/v_WalkingWithDog_g01_c01_flow1.png)

![Walking with dog sample_flow_v](https://github.com/juenkhaw/action_recognition_project/blob/vis-module/v_WalkingWithDog_g01_c01_flow2.png)

### Arguments

**Mandatory**

- `test_video` video folder name to be visualized
- `test_label` label index for the testing video
- `modality` modality to be tested on

**Optional**

- `-visflow` determine whether to show visualization result on [u or v] optical flows
- `-dv` device chosen to perform training
- `-endp` module block where forprop to and backprop from
- `-filter` filter chosen to be visualised
- `-nframe` frame number for each testing clip
- `-v1` activate to allow reporting of activation shape after each forward propagation

**Network Endpoints**

- `Conv3d_1`
- `Conv3d_2_x`
- `Conv3d_3_x`
- `Conv3d_4_x`
- `Conv3d_5_x`

Please refer to the source code for more details on arguments.

### Implementation References

- A Closer Look at Spatiotemporal Convolutions for Action Recognition https://arxiv.org/abs/1711.11248
- https://github.com/facebookresearch/VMZ
- https://github.com/irhumshafkat/R2Plus1D-PyTorch
- https://github.com/utkuozbulak/pytorch-cnn-visualizations
