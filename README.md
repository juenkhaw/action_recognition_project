# R(2+1)D Action Recognition Network
### Running

Please run on `init.py` with one of the arguments provided below:

**Training from scratch (two-stream)**

`ucf 2-stream [dataset_path] -fusion average -cl 8 -ld 18 -parallel -train -test -runalltest -nclip 10 -save -v2`

**Training from scratch (single-modality)**

`ucf [2-stream/rgb/flow] [dataset_path] -cl 8 -ld 18 -parallel -train -test -runalltest -nclip 10 -save -v2`

**With pre-trained model (Testing only)**

`ucf 2-stream [dataset_path] -fusion average -cl 8 -ld 18 -parallel -test -runalltest -nclip 10 -save -v2 -loadmodel [model_path]`

**Test run on gpu**

`ucf 2-stream [dataset_path] -cl 8 -ld 18 -ep 5 -tm -tc 2 -dv gpu -parallel -train -test -runalltest -nclip 2 -v2`

**Test run on cpu**

`ucf 2-stream [dataset_path] -cl 8 -ld 18 -ep 5 -tm -tc 2 -dv cpu -train -test -runalltest -nclip 2 -v2`

### Dataset Path

Please input path to the directory containing the rgb frames and optical flows as `dataset_path` 

Example: `../dataset/UCF-101`

**RGB Frame**

The directory outline of RGB frames:

`[dataset_path]/[dataset]_jpegs_256/jpegs_256/[video_name]`

**Optical Flow**

The directory of optical flows should contain `u` and `v` optical flows:

`[dataset_path]/[dataset]_tvl1_flow/tvl1_flow/[u/v]/[video_name]`

**References**
https://github.com/feichtenhofer/twostreamfusion

### Parallelism

Turns out as suspect in causing machine to crash, pending to try on not setting host GPU `cuda:0` as the gatherer

If parallelism implementation is causing problems, please run it without `-parallel` tag

**References**
- https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
- https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

### Arguments

**Training**

- `dataset` video dataset to be trained and validated | choices = ['ucf', 'hmdb']
- `modality` modality to be trained and validated | choices = ['rgb', 'flow', '2-stream']
- `dataset_path` path link to the directory where rgb_frames and optical flows located
- `-cl` initial temporal length of each video training input | default = 8
- `-sp` dataset split selected in training and evaluating model (0 to train/test on all split) | default = 0
- `-ld`depth of the resnet | default = 18 | choices = [18, 34]
- `-ep` number of epochs for training process | default = 45
- `-bs` number of labelled sample for each batch | default = 32
- `-lr` initial learning rate (alpha) for updating parameters | default = 0.01
- `-ss` decaying lr for each [ss] epoches | default = 10
- `-gm` lr decaying rate | default = 0.1
- `-mo` momemntum for batch normalization | default = 0.1
- `-es` epson for batch normalization | default = 1e-3

**Fusion method**

- `-fusion` fusion method to be used | default = 'none' | choices = ['none', 'average']

**Debugging mode**

- `-tm` activate test mode to minimize dataset for debugging purpose | default = False
- `-tc` number of labelled samples to be left when test mode is activated | default = 2

**Pre-training**

- `-train` activate to train the model | default = False
- `-loadmodel` path to the pretrained model state_dict | default = None

**Device and parallelism**

- `-dv` device chosen to perform training' | default = 'gpu' | choices = ['gpu', 'cpu']
- `-parallel` activate to run on multiple gpus | default = False
- `-wn` number of workers for some processes (safer to set at 0; -1 set as number of device) | default = 0

**Testing**

- `-test` activate to evaluate the model | default = False
- `-tbs` number of sample in each testing batch | default = 1
- `-runalltest` activate to run all prediction methods to obtain multiple accuracy | default = False
- `-lm` load testing samples as series of clips (video) or a single clip | default = 'clip' | choices = ['video', 'clip']
- `-topk`comapre true labels with top-k predicted labels | default = 1
- `-nclip` number of clips for testing video in video-level prediction | default = 10

**Output**
- `-save` save model and accuracy' | default = False
- `-v1` activate to allow reporting of activation shape after each forward propagation | default = False
- `-v2` activate to allow printing of loss and accuracy after each epoch | default = False

### Implementation References

- A Closer Look at Spatiotemporal Convolutions for Action Recognition https://arxiv.org/abs/1711.11248
- https://github.com/facebookresearch/VMZ
- https://github.com/irhumshafkat/R2Plus1D-PyTorch
