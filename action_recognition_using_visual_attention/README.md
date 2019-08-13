# Action Recognition Using Visual Attention


Soft attention based model for the task of action recognition in videos. Based on research papers [1](http://shikharsharma.com/projects/action-recognition-attention/) and [2](https://kgavrilyuk.github.io/videolstm.pdf).

<img src="https://github.com/dtransposed/Paper-Implementation/blob/master/action_recognition_using_visual_attention/images/3001.gif" width="300"> 

## Setting up the root path

Independently of the cloned repo, please create a root path for the project.

```
Root Path
├── HMDB_dataset (empty)
├── HMDB_predictions (empty)
└── HDMB_models (empty)
├── HMDB_videos (get dataset from http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
|	├── brush_hair
|	├── cartwheel
|	├── catch
|	├── ...
└── splits (get splits from http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
	├── brush_hair_test_split1.txt
	├── brush_hair_test_split2.txt
	└── brush_hair_test_split3.txt
	├── cartwheel_test_split1.txt
	├── ...

```

## Running the project

To run the project:

1. Edit parameter ```directory``` in  ```/data_creators/create_dataset_hmdb.py``` with the root path of your project and run the script to extract the dataset.

2. Edit ```config.yaml```. Firstly, fill out the missing paths with your root path. Secondly, alter parameters to your liking.

3. Run ```attention_hmdb_train.py``` to train your model. During training some artifacts will be saved in ```HDMB_models```: tensorboard logs, intermidiate models, plots of loss, accuracy and f1 score (empty plot, f1 score cannot be computed for multinomial classification problem).

4. Run ```attention_hmdb_predict.py``` to test your model. This will generate images (frame overlaid with heatmaps) in ```HDMB_predictions``` and return final test accuracy. 

## Additional Information

You can read about the project on my personal blog – [dtransposed](https://dtransposed.github.io/blog/Action-Recognition-Attention.html)



