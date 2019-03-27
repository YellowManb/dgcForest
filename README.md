# dgcForest
This is the main part of a paper I voted for. The title of the paper is: Deep Multigrained Cascade Forest for Hyperspectral Image Classification, delivered in the journal IEEE Transactions on Geoscience and Remote Sensing.

# main folders and their function
cascade:
             cascade_classifier.py  is the open source code from Zhou's paper[1],and it is cascade forest of gcForest and dgcForest
             fg_conv_cascade.py is our code,and it is classifier of deep multi-grained scanning
data_preprocessing:
	pca_block.py is our code and it is for PCA and neighbor region block
	rolling_window.py is a tool for pca_block.
estimators:
	Those are the open source code from  Zhou's paper[1].Those are estimators for cascade forest and deep mulit-grained scanning.
layers:
	fg_conv_layer.py is our deep multi-grained scanning layer;
	fg_pool_layer.py is our pool layer.
Note:fg_conv_layer.py,fg_pool_layer.py and fg_conv_cascade.py in cascade  constitute our deep multi-grained scanning

reference:
[1]Zhou Z H, Feng J. Deep Forest: Towards An Alternative to Deep Neural Networks[J]. 2017.
