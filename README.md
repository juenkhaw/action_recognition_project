# R(2+1)D Action Recognition Network
### Running

Please run on `init.py` with one of the arguments set provided below:

**Continue training on previous model (two-stream)**

`ucf 2-stream [dataset_path] -fusion [fusion_method] -train -loadmodel [model_path] -cl 8 -ld 18 -test -runalltest -dv gpu -save -savename [output_name] -v2`

**Training from scratch and testing with subbatches applied (two-stream)**

`ucf 2-stream [dataset_path] -fusion [fusion_method] -train -loadmodel [model_path] -cl 8 -ld 18 -sbs [training_subbatch_size] -test -runalltest -stbs [testing_subbatch_size] -dv gpu -save -savename [output_name] -v2`

**Evaluate on pre-trained model (testing only)**

`ucf [2-stream/rgb/flow] [dataset_path] -fusion [fusion_method] -loadmodel [model_path] -cl 8 -ld 18 -test -runalltest -save -savename [output_name] -v2`

**Training from scratch (two-stream)**

`ucf 2-stream [dataset_path] -fusion [fusion_method] -train -cl 8 -ld 18 -test -runalltest -save -savename [output_name] -v2`

**Training from scratch (single-modality)**

`ucf [2-stream/rgb/flow] [dataset_path] -train -cl 8 -ld 18 -test -runalltest -save -savename [output_name] -v2`

**Test run on gpu**

`ucf [2-stream/rgb/flow] [dataset_path] -train -cl 8 -ld 18 -ep 5 -sbs 2 -test -runalltest -stbs 5 -tm -tc 32 -v2`

**Test run on cpu**

`ucf [2-stream/rgb/flow] [dataset_path] -dv cpu -train -cl 8 -ld 18 -ep 5 -sbs 2 -test -runalltest -stbs 5 -tm -tc 32 -v2`

### Dataset Path

Please input path to the directory containing the rgb frames and optical flows as `dataset_path` 

Example: `../dataset/UCF-101`

**RGB Frame**

The directory outline of RGB frames:

`[dataset_path]/[dataset]_jpegs_256/jpegs_256/[video_name]`

**Optical Flow**

The directory of optical flows should contain `u` and `v` optical flows:

`[dataset_path]/[dataset]_tvl1_flow/tvl1_flow/[u/v]/[video_name]`

**Dataset Source**
https://github.com/feichtenhofer/twostreamfusion

### Parallelism

To be evaluated soon, just disable it (by not including `-parallel`) at current stage

Turns out as suspect in causing machine to crash, pending to try on not setting host GPU `cuda:0` as the gatherer

If parallelism implementation is causing problems, please run it without `-parallel` tag

**References**
- https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
- https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

### Output

The output file should contain contents with structure as following if training and testing are done in one shot:
<pre>
output.pth.tar
|-args  
|-content  
  |-modality (s)  
    |-split (s)  
      |-train_acc
      |-train_loss
      |-train_elapsed
      |-state_dict
      |-opt_dict
      |-sch_dict
      |-epoch
      |-video@N (s)
        |-predicted
        |-test_acc
        |-test_elpased
</pre>

### Arguments

**Training**

- `dataset` video dataset to be trained and validated
- `modality` modality to be trained and validated
- `dataset_path` path link to the directory where rgb_frames and optical flows located
- `-cl` initial temporal length of each video training input
- `-sp` dataset split selected in training and evaluating model (0 to perform train/test on all 3 splits)
- `-ld` depth (number of layers) of the resnet
- `-ep` number of epochs for training process
- `-bs` number of labelled sample for each batch
- `-sbs` number of labelled sample for each sub-batch
- `-lr` initial learning rate (alpha) for updating parameters
- `-ss` decaying lr for each [ss] epoches
- `-gm` lr decaying rate
- `-mo` momemntum for batch normalization
- `-es` epson for batch normalization

**Fusion method**

- `-fusion` fusion method to be used in merging scores from multiple modalities

**Debugging mode**

- `-tm` activate test mode to minimize dataset for debugging purpose
- `-tc` number of labelled samples to be left when test mode is activated

**Pre-training**

- `-train` activate to train the model
- `-loadmodel` path to the pretrained model `state_dict`

**Device and parallelism**

- `-dv` device chosen to perform training
- `-parallel` activate to run on multiple gpus
- `-wn` number of workers for some processes (safer to set at 0; -1 set as number of device)

**Testing**

- `-test` activate to evaluate the model
- `-tbs` number of sample in each testing batch
- `-stbs` number of clips in each testing subbatches
- `-runalltest` activate to run all prediction methods to obtain multiple accuracy
- `-lm` load testing samples as series of clips (video) or a single clip
- `-topk`comapre true labels with top-k predicted labels
- `-nclip` number of clips for testing video in video-level prediction

**Output**
- `-save` save model and accuracy
- `-saveintv` save models every [saveintv] epochs
- `-savename` name of the output file
- `-v1` activate to allow reporting of activation shape after each forward propagation
- `-v2` activate to allow printing of loss and accuracy after each epoch

Please refer to the source code for more details on arguments.

### Implementation References

- A Closer Look at Spatiotemporal Convolutions for Action Recognition https://arxiv.org/abs/1711.11248
- https://github.com/facebookresearch/VMZ
- https://github.com/irhumshafkat/R2Plus1D-PyTorch
- `pytorchsummary` by https://github.com/jacobkimmel/pytorch_modelsummary
