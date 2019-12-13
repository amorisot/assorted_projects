# Lab Notebook

To add your info, check out the [How to](#how-to) section.

Also see the [Bests](#bests) section.

## 2018-03-29 @ 20:00, Jessica

Adding the autoencoders.

| config_id | config link | results | max | status |
|--|--|--|--|--|
| capsnet_smallnorb_reconstruction_only | [config](https://github.com/jessstringham/G63-MLP/blob/efdab78/scripts/runners/config/capsnet_smallnorb_reconstruction_only.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/1d96b7e56020df8c39fc84538560a0482c30b3d7/experiment_analysis/CW4_results/capsnet_smallnorb_reconstruction_only__r1521403400_stats.json) | min loss: 0.003   | **IN PROGRESS** |
| cnn_smallnorb_reconstruction_only | [config](https://github.com/jessstringham/G63-MLP/blob/7534eea/scripts/runners/config/cnn_smallnorb_reconstruction_only.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/1d96b7e56020df8c39fc84538560a0482c30b3d7/experiment_analysis/CW4_results/cnn_smallnorb_reconstruction_only__r1521372424_stats.json) | max acc:  0.000%   | early stopped |

## 2018-03-28 @ 19:00, Jessica

Copying over Michael's tests!


### Skips

(additional experiments in progress)

Learning rate 0.0001

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_tinysmallnorb_skip | [config](https://github.com/jessstringham/G63-MLP/blob/23f3029/scripts/runners/config/capsnet_tinysmallnorb_skip.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_skip__r1521659406_stats.json) | max acc:  89.433%   | early stopped |
| capsnet_tinysmallnorb_skip | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_skip.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/019af411191ad2ae4782b6edd291189163be23a6/experiment_analysis/CW4_results/capsnet_tinysmallnorb_skip__r1522258420_stats.json) | max acc:  89.155%   | early stopped |
| capsnet_tinysmallnorb_skip | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_skip.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/019af411191ad2ae4782b6edd291189163be23a6/experiment_analysis/CW4_results/capsnet_tinysmallnorb_skip__r1522258421_stats.json) | max acc:  89.701%   | early stopped |

Learning rate 0.001 (note, this one doesn't learn very well! I think the learning rate is too high.)

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_tinysmallnorb_skip_learning_rate_0_001 | [config](https://github.com/jessstringham/G63-MLP/blob/147c632/scripts/runners/config/capsnet_tinysmallnorb_skip_learning_rate_0_001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_skip_learning_rate_0_001__r1521713426_stats.json) | max acc:  87.186%   | early stopped |
| capsnet_tinysmallnorb_skip_learning_rate_0_001 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_skip_learning_rate_0_001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/019af411191ad2ae4782b6edd291189163be23a6/experiment_analysis/CW4_results/capsnet_tinysmallnorb_skip_learning_rate_0_001__r1522258368_stats.json) | max acc:  88.113%   | early stopped |
| capsnet_tinysmallnorb_skip_learning_rate_0_001 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_skip_learning_rate_0_001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/019af411191ad2ae4782b6edd291189163be23a6/experiment_analysis/CW4_results/capsnet_tinysmallnorb_skip_learning_rate_0_001__r1522258388_stats.json) | max acc:  21.825%   | early stopped |

### CNN baseline

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| cnn_tinysmallnorb | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/cnn_tinysmallnorb.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/cnn_tinysmallnorb__r1522177950_stats.json) | max acc:  89.021%   | early stopped |
| cnn_tinysmallnorb | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/cnn_tinysmallnorb.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/cnn_tinysmallnorb__r1522178624_stats.json) | max acc:  87.577%   | early stopped |
| cnn_tinysmallnorb | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/cnn_tinysmallnorb.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/cnn_tinysmallnorb__r1522178625_stats.json) | max acc:  90.588%   | early stopped |


### tinysmallnorb capsnet baseline

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_tinysmallnorb_learning_rate_0_001 | [config](https://github.com/jessstringham/G63-MLP/blob/a9c72f1/scripts/runners/config/capsnet_tinysmallnorb_learning_rate_0_001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_learning_rate_0_001__r1522173947_stats.json) | max acc:  89.093%   | early stopped |
| capsnet_tinysmallnorb_learning_rate_0_001 | [config](https://github.com/jessstringham/G63-MLP/blob/a9c72f1/scripts/runners/config/capsnet_tinysmallnorb_learning_rate_0_001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_learning_rate_0_001__r1522174005_stats.json) | max acc:  89.763%   | early stopped |
| capsnet_tinysmallnorb_learning_rate_0_001 | [config](https://github.com/jessstringham/G63-MLP/blob/e44bc84/scripts/runners/config/capsnet_tinysmallnorb_learning_rate_0_001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_learning_rate_0_001__r1522177261_stats.json) | max acc:  89.454%   | early stopped |

### veclength 4 on tinysmallnorb

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_tinysmallnorb_veclength_4 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_4.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_4__r1522239896_stats.json) | max acc:  87.722%   | early stopped |
| capsnet_tinysmallnorb_veclength_4 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_4.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_4__r1522240520_stats.json) | max acc:  89.103%   | early stopped |
| capsnet_tinysmallnorb_veclength_4 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_4.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/019af411191ad2ae4782b6edd291189163be23a6/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_4__r1522260763_stats.json) | max acc:  87.299%   | early stopped |


#### don't use for error bars. these use the same seeds, so aren't as good comparisons

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_tinysmallnorb_veclength_4 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_4.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_4__r1522177860_stats.json) | max acc:  86.969%   | early stopped |
| capsnet_tinysmallnorb_veclength_4 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_4.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_4__r1522239674_stats.json) | max acc:  89.515%   | early stopped |


### veclength 16 on tinysmallnorb

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_tinysmallnorb_veclength_16 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_16.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_16__r1522239895_stats.json) | max acc:  89.412%   | early stopped |
| capsnet_tinysmallnorb_veclength_16 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_16.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_16__r1522240469_stats.json) | max acc:  88.464%   | early stopped |
| capsnet_tinysmallnorb_veclength_16 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_16.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_16__r1522240472_stats.json) | max acc:  88.948%   | early stopped |


#### don't use for error bars. these use the same seeds, so aren't as good comparisons

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_tinysmallnorb_veclength_16 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_16.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_16__r1522177863_stats.json) | max acc:  88.938%   | (wrong seed) early stopped |
| capsnet_tinysmallnorb_veclength_16 | [config](https://github.com/jessstringham/G63-MLP/blob/af3e528/scripts/runners/config/capsnet_tinysmallnorb_veclength_16.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/capsnet_tinysmallnorb_veclength_16__r1522239673_stats.json) | max acc:  89.443%   | (wrong seed) early stopped |

### A bunch of deeper nets

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| Deeper_1_added_conv | [config](https://github.com/jessstringham/G63-MLP/blob/7911936/scripts/runners/config/Deeper_1_added_conv.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_1_added_conv__r1522152937_stats.json) | max acc:  89.103%   | early stopped |
| Deeper_1_added_conv | [config](https://github.com/jessstringham/G63-MLP/blob/7044e75/scripts/runners/config/Deeper_1_added_conv.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_1_added_conv__r1522156928_stats.json) | max acc:  87.804%   | early stopped |
| Deeper_1_added_conv | [config](https://github.com/jessstringham/G63-MLP/blob/7044e75/scripts/runners/config/Deeper_1_added_conv.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_1_added_conv__r1522157302_stats.json) | max acc:  87.268%   | early stopped |
| Deeper_1_added_conv | [config](https://github.com/jessstringham/G63-MLP/blob/7044e75/scripts/runners/config/Deeper_1_added_conv.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_1_added_conv__r1522167163_stats.json) | max acc:  88.062%   | early stopped |
| Deeper_2_added_convs | [config](https://github.com/jessstringham/G63-MLP/blob/7911936/scripts/runners/config/Deeper_2_added_convs.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_2_added_convs__r1522153270_stats.json) | max acc:  88.629%   | early stopped |
| Deeper_2_added_convs | [config](https://github.com/jessstringham/G63-MLP/blob/7044e75/scripts/runners/config/Deeper_2_added_convs.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_2_added_convs__r1522156927_stats.json) | max acc:  88.711%   | early stopped |
| Deeper_2_added_convs | [config](https://github.com/jessstringham/G63-MLP/blob/7044e75/scripts/runners/config/Deeper_2_added_convs.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_2_added_convs__r1522157302_stats.json) | max acc:  89.443%   | early stopped |
| Deeper_2_added_convs | [config](https://github.com/jessstringham/G63-MLP/blob/7044e75/scripts/runners/config/Deeper_2_added_convs.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_2_added_convs__r1522167163_stats.json) | max acc:  88.124%   | early stopped |
| Deeper_4_layers | [config](https://github.com/jessstringham/G63-MLP/blob/fdf8729/scripts/runners/config/Deeper_4_layers.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_4_layers__r1521732002_stats.json) | max acc:  20.866%   | early stopped |
| Deeper_4_layers | [config](https://github.com/jessstringham/G63-MLP/blob/7911936/scripts/runners/config/Deeper_4_layers.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_4_layers__r1521812446_stats.json) | max acc:  20.866%   | early stopped |
| Deeper_5_layers | [config](https://github.com/jessstringham/G63-MLP/blob/7911936/scripts/runners/config/Deeper_5_layers.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_5_layers__r1521812446_stats.json) | max acc:  20.866%   | early stopped |
| Deeper_6_layers | [config](https://github.com/jessstringham/G63-MLP/blob/7911936/scripts/runners/config/Deeper_6_layers.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/Deeper_6_layers__r1521812470_stats.json) | max acc:  20.866%   | early stopped |
| MICHAEL_DEEPER_TEST_5_Layers | [config](https://github.com/jessstringham/G63-MLP/blob/a9c72f1/scripts/runners/config/MICHAEL_DEEPER_TEST_5_Layers.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/b5a01fab88684ee78cc761db4dafde62e8f0af8d/experiment_analysis/CW4_results/MICHAEL_DEEPER_TEST_5_Layers__r1522174406_stats.json) | max acc:  20.866%   | early stopped |




## 2018-03-15 @ 17:30, Jessica

Been looking at decoding stuff for fun images. Here are the experiments I'm going to run now:

### capsnet reconstruction (for the images)

 - cnn with dense reconstruction
   - cnn_smallnorb_decoder_learning_rate_0_0001.yaml (cnn_smallnorb_decoder didn't learn very well)
   - params: 13584635
   - blurry blobs

 - capsnet with dense reconstruction and orig alpha 
   - capsnet_smallnorb_leaky_relu_learning_rate_0_0001.yaml
   - params: 13500928
   - rerunning
   
 - capsnet with dense reconstruction (high alpha)
   - capsnet_smallnorb_weight_decoder_loss_high.yaml
   - params: 13500928
   - crosses for the planes! (this is a little surprising. So it probs wasn't mtl, it was the reconstruction error. But because it rotates around, that also suggests that capsnet is learning pose without us even encouraging it!)
   
 - capsnet with dense reconstruction (super high alpha)
   - capsnet_smallnorb_weight_decoder_loss_higher.yaml
   - params: 13500928
   - running
   
   
 - capsnet with deconv reconstruction
   - (Adrien!)
 
 - capsnet with dense (unless deconv turns out super awesome) + azimith
   - capsnet_mtlsmallnorb_dense_decoder.yaml
   - params: 13542913
   - running
   
### mtl (for the performance)
   
 - mtl with regular alpha
   - capsnet_mtlsmallnorb_regular_alpha.yaml
   - running
   
### mnist reconstruction

wonder what this looks like on mnist

 - cnn with dense reconstruction
   - cnn_mnist_decoder.yaml
   - params: 6435376

 - capsnet with dense reconstruction
   - capsnet_mnist_decoder.yaml
   - params: 6289978

emnist would be cool, but slow.

## 2018-03-09 @ 0:30, Jessica

There were a couple of underlying changes over the course of these runs. I had changed the dtype of the output from int64 to float32 for a few commits (it was for multigpu work. view the data_provider_wrappers.py's history for details). 

My (unverified) assumption is that while this would tweak everything, in the grand scheme of things it won't make a bigger difference than different weight initializations (and since the stable seed is already borked, we aren't losing anything.) But if we are concerned, we can rerun them.

### finished rerun with better learning rate

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_smallnorb_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/857a2ad/scripts/runners/config/capsnet_smallnorb_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/8c2202c7c5a214b68d78e25026d951e489c2c2f3/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_learning_rate_0_0001__r1520370875_stats.json) | 93.289%   | early stopped |

### finished smaller convs

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_smallnorb_leaky_relu_conv_128_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/857a2ad/scripts/runners/config/capsnet_smallnorb_leaky_relu_conv_128_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/8c2202c7c5a214b68d78e25026d951e489c2c2f3/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_conv_128_learning_rate_0_0001__r1520328543_stats.json) | 92.289%   | early stopped |
| capsnet_smallnorb_leaky_relu_conv_64_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/41463ae/scripts/runners/config/capsnet_smallnorb_leaky_relu_conv_64_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/8c2202c7c5a214b68d78e25026d951e489c2c2f3/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_conv_64_learning_rate_0_0001__r1520444699_stats.json) | 92.557%   | early stopped |


### finished adjustment of vector length

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_smallnorb_leaky_relu_vec_len_16_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/95cc048/scripts/runners/config/capsnet_smallnorb_leaky_relu_vec_len_16_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/8c2202c7c5a214b68d78e25026d951e489c2c2f3/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_vec_len_16_learning_rate_0_0001__r1520275151_stats.json) | 93.144%   | early stopped |
| capsnet_smallnorb_leaky_relu_vec_len_4_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/7aba7b0/scripts/runners/config/capsnet_smallnorb_leaky_relu_vec_len_4_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/8c2202c7c5a214b68d78e25026d951e489c2c2f3/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_vec_len_4_learning_rate_0_0001__r1520374761_stats.json) | 92.443%   | stopped after 10 epochs (others are marked as stopped after 11) |



## 2018-03-05 @ 18:30, Jessica

### Rerunning capsnet_smallnorb baseline

Unfortunately, this one's been a little sensitive. And our rng isn't being used everywhere. So this rerun failed to learn.
I'm going to try lowering the learning rate and rerun this one.

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_smallnorb_leaky_relu | [config](https://github.com/jessstringham/G63-MLP/blob/95cc048/scripts/runners/config/capsnet_smallnorb_leaky_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu__r1520254809_stats.json) | 20.784%   | **IN PROGRESS** |

### Effect of reducing capsnet_smallnorb size in different ways

I started to see OOM errors for some runs. Just in case, I tried reducing some params to see what would happen.


#### First conv layer feature map counts

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_smallnorb_leaky_relu_conv_128 | [config](https://github.com/jessstringham/G63-MLP/blob/95cc048/scripts/runners/config/capsnet_smallnorb_leaky_relu_conv_128.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_conv_128__r1520254774_stats.json) | 20.629%   | early stopped |
| capsnet_smallnorb_leaky_relu_conv_64 | [config](https://github.com/jessstringham/G63-MLP/blob/95cc048/scripts/runners/config/capsnet_smallnorb_leaky_relu_conv_64.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_conv_64__r1520254781_stats.json) | 91.742%   | **IN PROGRESS** |

#### Caps count

I'll look up the numbers, but these make a bigger difference in the parameter count

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_smallnorb_leaky_relu_caps_count_16 | [config](https://github.com/jessstringham/G63-MLP/blob/95cc048/scripts/runners/config/capsnet_smallnorb_leaky_relu_caps_count_16.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_caps_count_16__r1520254740_stats.json) | 90.093%   | early stopped |
| capsnet_smallnorb_leaky_relu_caps_count_8 | [config](https://github.com/jessstringham/G63-MLP/blob/95cc048/scripts/runners/config/capsnet_smallnorb_leaky_relu_caps_count_8.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_caps_count_8__r1520254763_stats.json) | 91.082%   | early stopped |

### Results from experiments last week

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_smallnorb_leaky_relu_vec_len_16_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/e11af88/scripts/runners/config/capsnet_smallnorb_leaky_relu_vec_len_16_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_vec_len_16_learning_rate_0_0001__e11af88__r1519748463.csv) | 92.794%   | early stopped |
| capsnet_smallnorb_leaky_relu_vec_len_4_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/91a7a0d/scripts/runners/config/capsnet_smallnorb_leaky_relu_vec_len_4_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_vec_len_4_learning_rate_0_0001__91a7a0d__r1519747168.csv) | 92.206%   | **IN PROGRESS** |


And from Adrien's

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_cifar10_16_vector | [config](https://github.com/jessstringham/G63-MLP/blob/aaf31e1/scripts/runners/config/capsnet_cifar10_16_vector.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/7b808ad56de0c246ee89504293b1efb29cc86b5b/experiment_analysis/CW4_results/capsnet_cifar10_16_vector__aaf31e1__r1519745467.csv) | 62.400%   | early stopped |
| capsnet_cifar10_4_vector | [config](https://github.com/jessstringham/G63-MLP/blob/aaf31e1/scripts/runners/config/capsnet_cifar10_4_vector.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/7b808ad56de0c246ee89504293b1efb29cc86b5b/experiment_analysis/CW4_results/capsnet_cifar10_4_vector__aaf31e1__r1519745453.csv) | 59.133%   | **IN PROGRESS** |

#### Archived results

One of these didn't learn; one of these I updated by accident but the learning rate improved results. Note that I'm rerunning the regular capsnet with learning rate 0.0001.

| config_id | config link | results | max val acc | status |
|--|--|--|--|--|
| capsnet_smallnorb_leaky_relu_vec_len_16 | [config](https://github.com/jessstringham/G63-MLP/blob/68cd57b/scripts/runners/config/capsnet_smallnorb_leaky_relu_vec_len_16.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_vec_len_16__68cd57b__r1519744595.csv) | 20.505%   | early stopped |
| capsnet_smallnorb_leaky_relu_vec_len_4 | [config](https://github.com/jessstringham/G63-MLP/blob/68cd57b/scripts/runners/config/capsnet_smallnorb_leaky_relu_vec_len_4.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/e0619f7d1302a09d22e92971cb7990c96b9860a2/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_vec_len_4__68cd57b__r1519744598.csv) | 89.206%   | **IN PROGRESS** |


## 2018-02-27 @ 16:30, Jessica

It's time for CW4! Here were the best runs from CW3. Moving all the old YAMLs over.

Note: augmentation does actually improve these! But I think we can start with these since we might experiment with augmentation.

### cnn

(updated 2018-03-15 shoot, cnn_smallnorb_batchnorm_leaky_relu_learning_rate_0_0001 was on cifar10)

| config_id | config link | results | max val acc |
|--|--|--|--|
| cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/3b79015/scripts/runners/config/cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001__3b79015__r1518959492.csv) | 87.652%   |
| cnn_smallnorb_leaky_relu_learning_rate_0_0005 | [config](https://github.com/jessstringham/G63-MLP/blob/12f19c8/scripts/runners/config/cnn_smallnorb_leaky_relu_learning_rate_0_0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/743c9906d18a8ef3670215add6a397c122d3a5ff/experiment_analysis/CW3_results/cnn_smallnorb_leaky_relu_learning_rate_0_0005__12f19c8__r1518832467.csv) | 90.649%   |
| cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/743c990/scripts/runners/config/cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001__743c990__r1518888036.csv) | 76.520%   |


### capsnet

| config_id | config link | results | max val acc |
|--|--|--|--|
| PARTIAL RUN: capsnet_emnist | [config](https://github.com/jessstringham/G63-MLP/blob/5a804ae/scripts/runners/config/capsnet_emnist.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/3b7901578c294e1bce24523abca0f555ee4b697b/experiment_analysis/CW3_results/capsnet_emnist__5a804ae__r1518865811.csv) | 87.449%   |
| capsnet_cifar10_leaky_relu | [config](https://github.com/jessstringham/G63-MLP/blob/56ccb9b/scripts/runners/config/capsnet_cifar10_leaky_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/2b79c1ad7404afe137f3db66674912feaad6b461/experiment_analysis/CW3_results/capsnet_cifar10_leaky_relu__56ccb9b__r1518802244.csv) | 63.773%   |
| capsnet_smallnorb_leaky_relu | [config](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/scripts/runners/config/capsnet_smallnorb_leaky_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/experiment_analysis/CW3_results/capsnet_smallnorb_leaky_relu__56ccb9b__r1518802244.csv) | 91.629%   |


## 2018-02-18 @ 9pm, Jessica

| config_id | config link | results | max val acc |
|--|--|--|--|
| cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/3b79015/scripts/runners/config/cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001__3b79015__r1518959492.csv) | 87.652%   |
| cnn_smallnorb_batchnorm_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/3b79015/scripts/runners/config/cnn_smallnorb_batchnorm_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_smallnorb_batchnorm_leaky_relu_learning_rate_0_0001__3b79015__r1518959500.csv) | 77.013%   |

## 2018-02-18 @ 9pm, Adrien

| config_id | config link | results | max val acc |
|--|--|--|--|
| cnn_cifar10_copy_antreas | [config](https://github.com/jessstringham/G63-MLP/blob/a438916/scripts/runners/config/cnn_cifar10_copy_antreas.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_cifar10_copy_antreas__a438916__r1518894260.csv) | 74.627%   |
| cnn_cifar10_copy_antreas_double_fc_layers | [config](https://github.com/jessstringham/G63-MLP/blob/438f62e/scripts/runners/config/cnn_cifar10_copy_antreas_double_fc_layers.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_cifar10_copy_antreas_double_fc_layers__438f62e__r1518907545.csv) | 74.827%   |
| cnn_cifar10_copy_antreas_double_filters | [config](https://github.com/jessstringham/G63-MLP/blob/438f62e/scripts/runners/config/cnn_cifar10_copy_antreas_double_filters.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_cifar10_copy_antreas_double_filters__438f62e__r1518907559.csv) | 76.373%   |
| cnn_cifar10_copy_antreas_extra_conv | [config](https://github.com/jessstringham/G63-MLP/blob/438f62e/scripts/runners/config/cnn_cifar10_copy_antreas_extra_conv.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_cifar10_copy_antreas_extra_conv__438f62e__r1518907568.csv) | 74.480%   |


## 2018-02-18 @ 1pm, Jessica

Gave up on the EMNIST run (it's still improving!)

Also ran the best CNN with data augmentation. We won't have time to rerun everything with it, but I think it's a good thing to check in on next time.

| config_id | config link | results | max val acc |
|--|--|--|--|
| PARTIAL RUN: capsnet_emnist | [config](https://github.com/jessstringham/G63-MLP/blob/5a804ae/scripts/runners/config/capsnet_emnist.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/3b7901578c294e1bce24523abca0f555ee4b697b/experiment_analysis/CW3_results/capsnet_emnist__5a804ae__r1518865811.csv) | 87.449%   |
| cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001_augment | [config](https://github.com/jessstringham/G63-MLP/blob/118024b/scripts/runners/config/cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001_augment.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/3b7901578c294e1bce24523abca0f555ee4b697b/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001_augment__118024b__r1518911112.csv) | 77.360%   |


## 2018-02-17 @ 8pm, Adrien

| config_id | config link | results | max val acc |
|--|--|--|--|
| cnn_cifar10 | [config](https://github.com/jessstringham/G63-MLP/blob/cdf930b/scripts/runners/config/cnn_cifar10.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10__cdf930b__r1518802183.csv) | 35.253%   |
| cnn_cifar10 | [config](https://github.com/jessstringham/G63-MLP/blob/e01fef6/scripts/runners/config/cnn_cifar10.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10__e01fef6__r1518827271.csv) | 39.907%   |
| cnn_cifar10_batchnorm_dropout_leaky_relu | [config](https://github.com/jessstringham/G63-MLP/blob/ce59fab/scripts/runners/config/cnn_cifar10_batchnorm_dropout_leaky_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_dropout_leaky_relu__ce59fab__r1518806988.csv) | 10.173%   |
| cnn_cifar10_batchnorm_dropout_leaky_relu | [config](https://github.com/jessstringham/G63-MLP/blob/e01fef6/scripts/runners/config/cnn_cifar10_batchnorm_dropout_leaky_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_dropout_leaky_relu__e01fef6__r1518826403.csv) | 29.987%   |
| cnn_cifar10_batchnorm_dropout_leaky_relu_learning_rate_0_0005 | [config](https://github.com/jessstringham/G63-MLP/blob/a882a17/scripts/runners/config/cnn_cifar10_batchnorm_dropout_leaky_relu_learning_rate_0_0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_dropout_leaky_relu_learning_rate_0_0005__a882a17__r1518829147.csv) | 58.413%   |
| cnn_cifar10_batchnorm_dropout_relu | [config](https://github.com/jessstringham/G63-MLP/blob/ce59fab/scripts/runners/config/cnn_cifar10_batchnorm_dropout_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_dropout_relu__ce59fab__r1518806976.csv) | 10.173%   |
| cnn_cifar10_batchnorm_dropout_relu | [config](https://github.com/jessstringham/G63-MLP/blob/e01fef6/scripts/runners/config/cnn_cifar10_batchnorm_dropout_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_dropout_relu__e01fef6__r1518826412.csv) | 40.013%   |
| cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/743c990/scripts/runners/config/cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001__743c990__r1518888036.csv) | 76.520%   |
| cnn_cifar10_batchnorm_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/cf09b2e/scripts/runners/config/cnn_cifar10_batchnorm_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_learning_rate_0_0001__cf09b2e__r1518886420.csv) | 75.693%   |
| cnn_cifar10_batchnorm_learning_rate_0.0005 | [config](https://github.com/jessstringham/G63-MLP/blob/a882a17/scripts/runners/config/cnn_cifar10_batchnorm_learning_rate_0.0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_learning_rate_0__a882a17__r1518829175.csv) | 44.853%   |
| cnn_cifar10_smaller_convs | [config](https://github.com/jessstringham/G63-MLP/blob/e01fef6/scripts/runners/config/cnn_cifar10_smaller_convs.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_smaller_convs__e01fef6__r1518826333.csv) | 24.920%   |
| cnn_cifar10_smaller_convs_learning_rate_0.0005 | [config](https://github.com/jessstringham/G63-MLP/blob/a882a17/scripts/runners/config/cnn_cifar10_smaller_convs_learning_rate_0.0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_smaller_convs_learning_rate_0__a882a17__r1518829227.csv) | 61.907%   |
| cnn_cifar10_tensorflow_tutorial | [config](https://github.com/jessstringham/G63-MLP/blob/4909fa9/scripts/runners/config/cnn_cifar10_tensorflow_tutorial.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_tensorflow_tutorial__4909fa9__r1518879455.csv) | 65.413%   |
| cnn_cifar10_tensorflow_tutorial | [config](https://github.com/jessstringham/G63-MLP/blob/66b9ce6/scripts/runners/config/cnn_cifar10_tensorflow_tutorial.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_tensorflow_tutorial__66b9ce6__r1518873300.csv) | 67.373%   |
| cnn_cifar10_tensorflow_tutorial_less_connections | [config](https://github.com/jessstringham/G63-MLP/blob/4909fa9/scripts/runners/config/cnn_cifar10_tensorflow_tutorial_less_connections.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_tensorflow_tutorial_less_connections__4909fa9__r1518879458.csv) | 64.013%   |
| cnn_smallnorb | [config](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/scripts/runners/config/cnn_smallnorb.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_smallnorb_1518736018.csv) | 81.454%   |





## 2018-02-17 @ 7pm, Jessica

Updated the table below to include later forms of in-progress things.

I tried a few lower learning rates on the small NORB runs.

| config_id | config link | results | max val acc |
|--|--|--|--|
| cnn_smallnorb_learning_rate_0_0005 | [config](https://github.com/jessstringham/G63-MLP/blob/12f19c8/scripts/runners/config/cnn_smallnorb_learning_rate_0_0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/743c9906d18a8ef3670215add6a397c122d3a5ff/experiment_analysis/CW3_results/cnn_smallnorb_learning_rate_0_0005__12f19c8__r1518832502.csv) | 87.515%   |
| cnn_smallnorb_leaky_relu_learning_rate_0_0005 | [config](https://github.com/jessstringham/G63-MLP/blob/12f19c8/scripts/runners/config/cnn_smallnorb_leaky_relu_learning_rate_0_0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/743c9906d18a8ef3670215add6a397c122d3a5ff/experiment_analysis/CW3_results/cnn_smallnorb_leaky_relu_learning_rate_0_0005__12f19c8__r1518832467.csv) | 90.649%   |

| config_id | config link | results | max val acc |
|--|--|--|--|
| capsnet_smallnorb_no_relu_learning_rate_0_0005 | [config](https://github.com/jessstringham/G63-MLP/blob/12f19c8/scripts/runners/config/capsnet_smallnorb_no_relu_learning_rate_0_0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/743c9906d18a8ef3670215add6a397c122d3a5ff/experiment_analysis/CW3_results/capsnet_smallnorb_no_relu_learning_rate_0_0005__12f19c8__r1518832463.csv) | 90.010%   |
| capsnet_smallnorb_leaky_relu_learning_rate_0.0005 | [config](https://github.com/jessstringham/G63-MLP/blob/12f19c8/scripts/runners/config/capsnet_smallnorb_leaky_relu_learning_rate_0.0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/743c9906d18a8ef3670215add6a397c122d3a5ff/experiment_analysis/CW3_results/capsnet_smallnorb_leaky_relu_learning_rate_0__12f19c8__r1518832411.csv) | 91.175%   |

## 2018-02-16 @ 10pm, Jessica

Found a few things!

### activation function on capsnets matters

Getting a few % increase by switching to Leaky ReLU.

smallnorb runs have continued kinda sucking. ReLU was the only one that managed to train normally. No-activation on convolutions couldn't learn at all. Leakly ReLU quickly got better than ReLU, then crashed down to 20% (which is chance on smallNORB). Below are the results:


| config_id | config link | results | max val acc |
|--|--|--|--|
| capsnet_cifar10_no_relu | [config](https://github.com/jessstringham/G63-MLP/blob/6fded59/scripts/runners/config/capsnet_cifar10_no_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/743c9906d18a8ef3670215add6a397c122d3a5ff/experiment_analysis/CW3_results/capsnet_cifar10_no_relu__6fded59__r1518830777.csv) | 55.227%   |
| capsnet_cifar10_1518728160_continued_from_1518696728 | [config](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/scripts/runners/config/capsnet_cifar10.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/experiment_analysis/CW3_results/capsnet_cifar10_1518728160_continued_from_1518696728.csv) | 58.947%   |
| capsnet_cifar10_leaky_relu | [config](https://github.com/jessstringham/G63-MLP/blob/56ccb9b/scripts/runners/config/capsnet_cifar10_leaky_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/2b79c1ad7404afe137f3db66674912feaad6b461/experiment_analysis/CW3_results/capsnet_cifar10_leaky_relu__56ccb9b__r1518802244.csv) | 63.773%   |

| config_id | config link | results | max val acc |
|--|--|--|--|
| capsnet_smallnorb_no_relu | [config](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/scripts/runners/config/capsnet_smallnorb_no_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/experiment_analysis/CW3_results/capsnet_smallnorb_no_relu__febdd55__r1518788972.csv) | 20.866%   |
| capsnet_smallnorb | [config](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/scripts/runners/config/capsnet_smallnorb.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/experiment_analysis/CW3_results/capsnet_smallnorb_1518696728.csv) | 90.021%   |
| capsnet_smallnorb_leaky_relu | [config](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/scripts/runners/config/capsnet_smallnorb_leaky_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/experiment_analysis/CW3_results/capsnet_smallnorb_leaky_relu__56ccb9b__r1518802244.csv) | 91.629%   |


### CNN cifar10 learning rate

-Meanwhile we've been having trouble getting CNN on CIFAR10 to train. Adrien is trying a few architectures. One thing I tried was using different learning rates.-

No wait, I think I'm getting this confused with smallNORB. Well, either way, here's the learning rate results:

| config_id | config link | results | max val acc |
|--|--|--|--|
| cnn_cifar10 | [config](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/scripts/runners/config/cnn_cifar10.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/experiment_analysis/CW3_results/cnn_cifar10__1518701641.csv) | 62.787%   |
| cnn_cifar10_learning_rate_0_0005 | [config](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/scripts/runners/config/cnn_cifar10_learning_rate_0_0005.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/experiment_analysis/CW3_results/cnn_cifar10_learning_rate_0_0005__ffb8155__r1518789110.csv) | 68.320%   |
| cnn_cifar10_learning_rate_10e-4 | [config](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/scripts/runners/config/cnn_cifar10_learning_rate_10e-4.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/14dd95a4685322cae2496d024a5bc362e5060cac/experiment_analysis/CW3_results/cnn_cifar10_learning_rate_10e-4__51c6cf1__r1518787072.csv) | 76.253%   |


## 2018-02-16 @ 10am, Jessica

First post! I should have started this sooner! Here's a rough recap:

I made a few code changes, and moved old results into a graveyard folder.


### "pre_relu"

I had a bug where PrimaryCapsules and Convolutions weren't actually using the 'activation' field.

I also used to 'flatten' before every dense layer, and now I require you to add a reshape layer. If I understand right, this won't really make a difference.

I also copy-pasted aguron's notebook routing code again, just in case. I don't think I functionally changed anything though.

The results from before this are in [this folder](https://github.com/jessstringham/G63-MLP/tree/a52b4fdc57d29902d2f66d99d54465f4f7665e64/experiment_analysis/CW3_results/CW3_results/pre_relu_capsnet)

This made a 3% difference in capsnet's cifar10 [55.2%](https://github.com/jessstringham/G63-MLP/blob/a52b4fdc57d29902d2f66d99d54465f4f7665e64/experiment_analysis/CW3_results/CW3_results/pre_relu_capsnet/shuffled_capsnet_cifar/epoch_stats.csv) vs [58.9%](https://github.com/jessstringham/G63-MLP/blob/a52b4fdc57d29902d2f66d99d54465f4f7665e64/experiment_analysis/CW3_results/CW3_results/capsnet_cifar10_1518696728.csv)! I'm planning to run a real experiment on this.

### Shuffling

I also copied over a change from `mlpractical` in this [commit](https://github.com/jessstringham/G63-MLP/commit/8cae14e8a8c0c448223e53f85a1581b13d332e48). At this point. Results from before this are in [pre_relu_capsnet/nonshuffled_capsnet_cifar](https://github.com/jessstringham/G63-MLP/tree/a52b4fdc57d29902d2f66d99d54465f4f7665e64/experiment_analysis/CW3_results/CW3_results/pre_relu_capsnet/nonshuffled_capsnet_cifar).


### Current state

We're running Capsnets and CNNs (with similar number of parameters) on small NORB, cifar10, and mnist (`sample_capsnet`) to sanity check.

We're having a hard time getting smallNORB to train. They're getting stuck at 20%, which is chance for smallNORB. Before the relu change above, the capsnet wasn't learning either. Now we're having trouble getting the cnn to learn. That's what Adrien is working on.


# How To

 * Add a line with `## date @ time, name`.

One way to print results is to copy and commit the jsons to the `experiment_analysis/CW4_results` folder, then run

    python explore_experiment_results.py links | grep -f <(git diff-tree --no-commit-id --name-only -r HEAD)

Where `HEAD` can also the git sha from the commit where you added the files. This will print out links for the files added in the commit. Otherwise, you can run

    python explore_experiment_results.py links

To print them all!

 * Use [permalinks](https://help.github.com/articles/getting-permanent-links-to-files/).

 * \% are "accuracy on validation set" unless otherwise stated.


# Bests

## cnn

| config_id | config link | results | max val acc |
|--|--|--|--|
| cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/3b79015/scripts/runners/config/cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_emnist_batchnorm_leaky_relu_learning_rate_0_0001__3b79015__r1518959492.csv) | 87.652%   |
| cnn_smallnorb_batchnorm_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/3b79015/scripts/runners/config/cnn_smallnorb_batchnorm_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/09764d7eb59861a91a9936997782a9414abb3788/experiment_analysis/CW3_results/cnn_smallnorb_batchnorm_leaky_relu_learning_rate_0_0001__3b79015__r1518959500.csv) | 77.013%   |
| cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/743c990/scripts/runners/config/cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/5b7f6440e30ddf07a1db04f1981dd0c71ec7f609/experiment_analysis/CW3_results/cnn_cifar10_batchnorm_leaky_relu_learning_rate_0_0001__743c990__r1518888036.csv) | 76.520%   |


## capsnet

| config_id | config link | results | max val acc |
|--|--|--|--|
| PARTIAL RUN: capsnet_emnist | [config](https://github.com/jessstringham/G63-MLP/blob/5a804ae/scripts/runners/config/capsnet_emnist.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/3b7901578c294e1bce24523abca0f555ee4b697b/experiment_analysis/CW3_results/capsnet_emnist__5a804ae__r1518865811.csv) | 87.449%   |
| capsnet_cifar10_leaky_relu | [config](https://github.com/jessstringham/G63-MLP/blob/56ccb9b/scripts/runners/config/capsnet_cifar10_leaky_relu.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/2b79c1ad7404afe137f3db66674912feaad6b461/experiment_analysis/CW3_results/capsnet_cifar10_leaky_relu__56ccb9b__r1518802244.csv) | 63.773%   |
| capsnet_smallnorb_leaky_relu_learning_rate_0_0001 | [config](https://github.com/jessstringham/G63-MLP/blob/857a2ad/scripts/runners/config/capsnet_smallnorb_leaky_relu_learning_rate_0_0001.yaml) | [results](https://github.com/jessstringham/G63-MLP/blob/8c2202c7c5a214b68d78e25026d951e489c2c2f3/experiment_analysis/CW4_results/capsnet_smallnorb_leaky_relu_learning_rate_0_0001__r1520370875_stats.json) | 93.289%   | early stopped |
