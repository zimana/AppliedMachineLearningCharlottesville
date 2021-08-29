#demo for Applied Machine Learning Festival 2019
#based on R-studio example 
#
library(keras)
#
#mnist - image dataset provided via keras
#
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
#
# reshape - 28 column , so 28x28 = 784
#array_reshape maps array to a "layer"
#
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
#
# rescale - turns training data into 0-1; Grayscale
# values of images is 255
#
x_train <- x_train / 255
x_test <- x_test / 255
#
#one hot encode - turns a class into a binary class
#
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
#
#Define the keras model. KMS is for a linear 
#stack of layers
#
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
summary(model)
#
#compile the model with loss function
#
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)
#
#Fit the batch size (history)
#Plot the model using history
#
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  view_metrics = TRUE,
  validation_split = 0.2
)
plot(history)
#
#Evaluate the model
#
model %>% evaluate(x_test, y_test) 
#
#Prediction
#
model %>% predict_classes(x_test[1:100,])
#
library(rmarkdown)
render("TomTom_Tensorflow_2019b.R")




