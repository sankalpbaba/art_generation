# art_generation

Fast arbitrary image style transfer PRETRAINED MODEL HAS BEEN USED FROM TENSORFLOW HUB LIBRARY.
VGG19 IMAGE NET WEIGHTS HAVE BEEN USED FOR THE NEURAL TRANSFER.

# NERUAL STYLE TRANSFER
Neural Style Transfer is the technique of blending style from one image into another image keeping its content intact. The only change is the style configurations of the image to give an artistic touch to your image.

The content image describes the layout or the sketch and Style being the painting or the colors. It is an application of Computer Vision related to image processing techniques and Deep Convolutional Neural Networks.


**HOW DOES NEURAL STYLE WORK**
Training a style transfer model requires two networks: a pre-trained feature extractor and a transfer network.
NST uses a pre-trained model trained on ImageNet- VGG in TensorFlow.
Images themselves make no sense to the model. These have to be converted into raw pixels and given to the model to transform it into a set of features, which is what Convolutional Neural Networks are responsible for.
Thus, somewhere in between the layers, where the image is fed into the model, and the layer, which gives the output, the model serves as a complex feature extractor. All we need to leverage from the model is its intermediate layers, and then use them to describe the content and style of the input images.
The input image is transformed into representations that have more information about the content of the image, rather than the detailed pixel value.
The features that we get from the higher levels of the model can be considered more related to the content of the image.
**Content loss**
It helps to establish similarities between the content image and the generated image.
To obtain a representation of the style of a reference image, we use the correlation between different filter responses.
Content loss is calculated by Euclidean distance between the respective intermediate higher-level feature representation of input image (x) and content image (p) at layer l.
![CONTENT LOSS](https://user-images.githubusercontent.com/90260133/150290835-80436439-80ab-42eb-9b0a-bba63dc3c47d.png)

**Style loss**
Style loss is conceptually different from Content loss.

We cannot just compare the intermediate features of the two images and get the style loss.

That's why we introduce a new term called Gram matrices.

Gram matrix is a way to interpret style information in an image as it shows the overall distribution of features in a given layer. It is measured as the amount of correlation present between features maps in a given layer.

Style loss is calculated by the distance between the gram matrices (or, in other terms, style representation) of the generated image and the style reference image.

The contribution of each layer in the style information is calculated by the below formula:
![STYLE LOSS](https://user-images.githubusercontent.com/90260133/150290999-b4814dc3-f678-4873-8234-133f031f3d8d.png)
Thus, the total style loss across each layer is expressed as:
![STYLE FINAL LOSS](https://user-images.githubusercontent.com/90260133/150291021-d15040d9-6f42-4281-a684-141236826553.png)

**MODEL STRUCTURE**
<img width="620" alt="MODEL" src="https://user-images.githubusercontent.com/90260133/150291133-a4d79b68-3748-4a04-aa17-732448c7068a.png">

**SUMMARY**                               
Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
It uses representations (hidden layer activations) based on a pretrained ConvNet/IMAGENET.
The content cost function is computed using one hidden layer's activations.
The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
Optimizing the total cost function results in synthesizing new images.

**WE can also tune your hyperparameters:**

Which layers are responsible for representing the style? STYLE_LAYERS
How many iterations do you want to run the algorithm? num_iterations
What is the relative weighting between content and style? alpha/beta

**OUTPUT IMAGE**
![output](https://user-images.githubusercontent.com/90260133/150291217-c6a7bb89-fa86-405d-a558-fe437981b75a.png)


