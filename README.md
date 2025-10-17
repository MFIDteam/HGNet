# HGNet: A Hypergraph for Classification Head to Capture Multi-scale Spatial Dependencies

The HGNet_T (tiny) model we have released is a classification network with a graph structure (each edge connects two nodes), utilizing a graph alternating update strategy and Dependency-Aware Fusion (DAF) for the classification network design. Compared to HGNet_B (base), HGNet_T (tiny) has fewer parameters but lacks multi-scale hypergraph, sparse graph updates, and the orthogonal fusion module. After the publication of this paper, we will release the complete HGNet classification network along with more detailed usage instructions.

## Overall Structure of HGNet
![Collection](images/HGNet.png)

## Intorduction
Image classification models first extract high-dimensional features from images through the feature extraction module, and then directly map these high-dimensional features to class probabilities via pooling and fully connected layers. However, in the process of mapping features from high-dimensional to low-dimensional spaces, existing models inevitably lose a significant amount of image information, and if salient features are not selected, this loss would limit the model’s ability to predict class probabilities based on discriminative information. To address this issue, we propose a hypergraph-based HGNet for image classification. HGNet innovatively constructs multi-scale hypergraphs in the feature space and captures spatial dependencies among features through an alternating update strategy, enhancing the model's ability to perceive discriminative information. Additionally, to reduce information redundancy among features at different scales, we introduce a multidimensional orthogonalization module that explicitly decorrelates these features, allowing each feature to focus on its own discriminative information.

## Visualization
![Collection](images/Visualization.png)

We visualize the image features of the final layer of ResNet101 and HGNet using Class Activation Mapping (CAM), and visualize the regions of focus in these output features.
