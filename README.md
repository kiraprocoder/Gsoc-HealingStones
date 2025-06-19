
## Flow for now

- creating ground truth
- dataset generation
- model training 


## Ground truth generation
- GT_preprocess.py
Imports the files into blender. aligns them relative to each other by rotating axes.
after importing manually connect them together 

- GT_refrence_point.py
Exports the completed fragment connection as a json file which has information about rotation and quaternation.


- GT_refrence_reconstruction.py
run this script to apply ground truth to the fragments and create a pair which has applied ground truth relative to each other so that they are appear connnected 

## Dataset generation
- Dataset_generation.py
this script  takes the  reconstructed fragments with the applied ground truth and makes dataset for the model. dataset include positive pairs, negative pairs and hard negative pairs
## model training
- feature_extractor.py and graph_constructor.py

after executing these in order this generates a .pt graph file which is used to feed the model.



## needs improvements
- model.py
- trainer.py

there are room for improvements in these. I'll be working on the architecture of this for now 