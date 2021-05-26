from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp
import os
import glob
import cv2

def normalize(tensor, mean, std):
    for channel in range(3):
        tensor[..., channel] = ((tensor[..., channel] - mean[channel]) / std[channel])
    return tensor

def load_image(img_path, max_size=512, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    image = image.resize((size, size))
    normalised_image = normalize(np.array(image) / 255.0, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    return tf.expand_dims(normalised_image, axis=0)

def im_convert(inputs):
    image = inputs * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))
    image = np.clip(image, 0, 1)
    return np.squeeze(image)
    
def softmax3d(inputs):
    a,b,c,d = inputs.shape
    inputs = tf.reshape(inputs, (1,-1))
    output = tf.nn.softmax(inputs)
    output = tf.reshape( output, (a,b,c,d))
    return output

def gram_matrix(inputs):
    shape_x = inputs.shape
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(inputs, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

def get_features(image, model, layers=None):
    if len(image.shape) < 3:
        image = tf.reshape(image, (1, tf.sqrt(tf.size(image)/3), tf.sqrt(tf.size(image)/3), 3))

    T = 1
    alpha = 0.001
    if layers is None:
        layers = [ 'conv1_relu',
                   'pool1_pool',
                   'conv2_block1_out',
                   'conv2_block2_out',
                   'conv2_block3_out',
                   'conv3_block1_out',
                   'conv3_block2_out',
                   'conv3_block3_out',
                   'conv3_block4_out',
                   'conv4_block1_out',
                   'conv4_block2_out',
                   'conv4_block3_out',
                   'conv4_block4_out',
                   'conv4_block5_out',
                   'conv4_block6_out',
                   'conv5_block1_out',
                   'conv5_block2_out',
                   'conv5_block3_out'
                    ]

    features_outputs_dict = {}
    features_outputs = []
    for layer in layers:
        features_outputs.append(model.get_layer(layer).output)
    
    features_model = models.Model(inputs=model.inputs, 
        outputs = features_outputs)

    features_outputs = features_model(image)

    if ARCHITECTURE == 'resnet50_swag':
        for layer, output in zip(layers, features_outputs):
            features_outputs_dict[layer] = softmax3d(output)

    else:
        for layer, output in zip(layers, features_outputs):
            features_outputs_dict[layer] = output

    return features_outputs_dict


def style_transfer(model, style, content, target, style_layer_weights, content_layer_weights,
                    style_weight, content_weight):
    content_features = get_features(content, model)
    style_features = get_features(style, model)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}


    def f(target):
        style_loss = 0
        with tf.GradientTape() as tape:
            tape.watch(target)
            target_features = get_features(target, model)
            content_loss = tf.math.reduce_mean((target_features[content_layer_weights] -
                                                    content_features[content_layer_weights]) ** 2)

            for layer in style_layer_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]

                layer_style_loss = style_layer_weights[layer] * tf.math.reduce_mean(
                    (target_gram - style_gram) ** 2)
                style_loss += style_weight * layer_style_loss

            total_loss = content_weight * content_loss + style_loss
            print(total_loss)
            
        gradients = tape.gradient(total_loss, target)
        gradients = tf.reshape(gradients, (1, -1))
        loss = tf.reshape(total_loss, (-1))
        return loss, gradients

    inputs = tf.reshape(target, (1, -1))
    results = tfp.optimizer.lbfgs_minimize(
        f, initial_position=inputs, max_iterations = 1000)

    final_img = im_convert(tf.reshape(results.position, target.shape))
    return final_img

def main(args):
    global ARCHITECTURE
    ARCHITECTURE = args.architecture
    resnet = tf.keras.applications.ResNet50(include_top = False, input_shape=(None, None, 3))

    style_image_list = []
    for files in glob.glob('./style_images/*'):
        style_image_list.append(files)
    print("Style images loaded")

    content_image_list = []
    for files in glob.glob('./content_images/*'):
        content_image_list.append(files)
    print("Content images loaded")

    for i_style in range(len(style_image_list)):
        style_img_name = style_image_list[i_style].split('/')
        style_img_name = style_img_name[-1].split('.')
        style_img_name = style_img_name[0].split('\\')[-1]
        for i_content in range(len(content_image_list)):

            print('processing content', i_content, ' style ', i_style)
            
            style = tf.cast(load_image(style_image_list[i_style]), tf.float32)
            content = tf.cast(load_image(content_image_list[i_content]), tf.float32)

            target = tf.identity(content)
            style_weights = {'conv1_relu': 1.0,
                            'conv2_block3_out': 1.0,
                            'conv3_block4_out': 1.0,
                            'conv4_block6_out': 1.0,
                            'conv5_block3_out': 1.0}
            content_weights = 'conv4_block6_out'

            content_weight = 1
            style_weight = 1e17

            final_styled = style_transfer(resnet, style, content, target, style_weights, content_weights,
                                                    style_weight, content_weight)
        
            content_img_name = content_image_list[i_content].split('/')
            content_img_name = content_img_name[-1].split('.')
            content_img_name = content_img_name[0].split('\\')[-1]

            if not os.path.exists('./results/' + ARCHITECTURE):
                os.makedirs('./results/' + ARCHITECTURE)

            save_path_cv2 = './results/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_cv.png'
            final_styled_cv2 = np.uint8(255 * final_styled)
            final_styled_cv2_bgr = final_styled_cv2[:, :, [2, 1, 0]]
            cv2.imwrite(save_path_cv2, final_styled_cv2_bgr)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TensorFlow SWAG')
    parser.add_argument('--architecture', type=str, default='resnet50_swag', choices = ['resnet50_swag', 'resnet50'], help='Architecture name')
    args = parser.parse_args()

    main(args)
