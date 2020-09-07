# CNN-for-image-classification
Train a small CNN to classify images from CIFAR10 dataset.

## Analysis

### Preprocessing steps
* MLP:  
  * Randomly, select 20% of the whole training dataset as this problem’s training.  
  * 50% of testing dataset is testing, another 50% is validation.  
  * For the data_image dataset, I do the normalization on all of the training, validation and testing datasets. I reshape the data_image dataset, which is 32*32*3 into an array: X_train.reshape( len(X_train), 32*32*3 ).astype('float32'). Then it normalizes the data_image into [0,1]. For example, after the normalization, the shape of X_train_normalization is (10000, 3072).  
  * For the data_label dataset, I do one hot encoding on the label. For example, the shape of y_train_onehot is (10000,10).  
* CNN1 & CNN2:  
  * Randomly, select 20% of the whole training dataset as this problem’s training.  
  * 50% of testing dataset is testing, another 50% is validation.  
  * For the data_image dataset, because the CNN allows the image input, so I just do the normalization, which normalizes the data into [0,1]. For example, after the normalization, the shape of X_train_normalization is (10000, 32, 32, 3).  
  * For the data_label dataset, I do one hot encoding on the label. For example, the shape of y_train_onehot is (10000,10).  
  
### Description of the output layer used and the loss function, and why you made these choices.
* Unit:  
I use the dense layer for the output layer, the unit is 10, because there are 10 different classes in this dataset. 
* Activation function:  
The activation function is softmax. Softmax activation function is used widely in multi-class classification. And the output is the probability, we choose the max value of the result( probability) as our predict class. So in this problem, I choose use softmax as the activation function on the output layer. 
* Loss function:  
The loss function I choose to use the categorical crossentropy. We usually use softmax activation function and categorical crossentropy for the multi-classes classification. This loss function is used to evaluate the difference between the probability distribution obtained by the current training and the true distribution.   
* Optimizer:  
For the optimizer, I choose to use the Adam, which is the most widely used in machine learning area recent years. Adam optimizer combines the momentum, bias correction and AdaGrad/ RMSProp together. After doing several experiments, I find Adam optimizer has the best results.  
* Overall, I use the softmax +  categorical crossentropy + Adam.  

### Recommendations to improve the network. What changes would you make to the architecture and why?
* I would like to do more experiment on layers and neurons. Because, the model seems like a little bit easy.  
* I would like to run more epochs. Because during these 5 epochs, the accuracy is still higher, which is not steady or lower. This means the model is not perfected fitted.   
* After I can do more epochs, I will see the training and validation acc to avoid being overfitted, which shows a good result on training, but when we test on the testing set, it always bad.  
* We can tune the hyper parameters. For example, the dropout, filter size, etc.. Hyper parameters are the important things in the model. We should try different parameters and find the best fitted one in the model.  
