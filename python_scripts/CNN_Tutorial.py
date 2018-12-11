import tensorflow as tf
import CNN_Config as cnn
a = tf.truncated_normal([16,128,128,3])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.shape(a)))
b=tf.reshape(a,[16,49152])
print(sess.run(tf.shape(b)))

#Classes that can be detected
classes = ['Car', 'Not Car']
num_classes = len(classes)
 
train_path='train_set_cars'
 
# validation split
validation_size = 0.2
 
# batch size
batch_size = 16

#num of channels
num_channels = 3

#num of filters
num_filters_conv1 = 32
num_filters_conv2 = 32
num_filters_conv3 = 64

#convultional filter size (X by X)
filter_size_conv1 = 3
filter_size_conv2 = 3
filter_size_conv3 = 3

#fully connected layer size 
fc_layer_size = 128
 
data = tf.DataSet.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
 
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1 = cnn.create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
 
layer_conv2 = cnn.create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)
 
layer_conv3= cnn.create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
          
layer_flat = cnn.create_flatten_layer(layer_conv3)
 
layer_fc1 = cnn.create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)
 
layer_fc2 = cnn.create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)
 


