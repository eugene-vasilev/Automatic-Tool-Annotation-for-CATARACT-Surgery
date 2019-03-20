from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenetv2 import MobileNetV2

from keras.layers import Conv2D, Add, ZeroPadding2D, MaxPooling2D, Input, GlobalAveragePooling2D, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


def darknet_conv2d(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def darknet_conv2d_bn_leaky(input_node, *args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    x = darknet_conv2d(*args, **no_bias_kwargs)(input_node)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def darknet19(input_shape, output_len):
    def bottleneck_block(input_node, outer_filters, bottleneck_filters):
        """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
        x = darknet_conv2d_bn_leaky(input_node, outer_filters, (3, 3))
        x = darknet_conv2d_bn_leaky(x, bottleneck_filters, (1, 1))
        x = darknet_conv2d_bn_leaky(x, outer_filters, (3, 3))
        return x

    def bottleneck_x2_block(input_node, outer_filters, bottleneck_filters):
        """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
        x = bottleneck_block(input_node, outer_filters, bottleneck_filters)
        x = darknet_conv2d_bn_leaky(x, bottleneck_filters, (1, 1))
        x = darknet_conv2d_bn_leaky(x, outer_filters, (3, 3))
        return x

    inputs = Input(input_shape)
    x = darknet_conv2d_bn_leaky(inputs, 32, (3, 3))
    x = MaxPooling2D()(x)
    x = darknet_conv2d_bn_leaky(x, 64, (3, 3))
    x = MaxPooling2D()(x)
    x = bottleneck_block(x, 128, 64)
    x = MaxPooling2D()(x)
    x = bottleneck_block(x, 256, 128)
    x = MaxPooling2D()(x)
    x = bottleneck_x2_block(x, 512, 256)
    x = MaxPooling2D()(x)
    x = bottleneck_x2_block(x, 1024, 512)
    x = darknet_conv2d(output_len, (1, 1), activation='softmax')(x)
    out = GlobalAveragePooling2D()(x)

    return Model(inputs, out)


def darknet53(input_shape, output_len):
    def resblock_body(x, num_filters, num_blocks):
        """A series of resblocks starting with a downsampling Convolution2D."""
        # Darknet uses left and top padding instead of 'same' mode
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = darknet_conv2d_bn_leaky(x, num_filters, (3, 3), strides=(2, 2))
        for i in range(num_blocks):
            y = darknet_conv2d_bn_leaky(x, num_filters // 2, (1, 1))
            y = darknet_conv2d_bn_leaky(y, num_filters, (3, 3))
            x = Add()([x, y])

        return x

    inputs = Input(input_shape)
    x = darknet_conv2d_bn_leaky(inputs, 32, (3, 3))
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)

    x = GlobalAveragePooling2D()(x)
    out = Dense(output_len, activation='softmax')(x)

    return Model(inputs, out)


def resnet50(input_shape, output_len):
    return ResNet50(weights=None, input_shape=input_shape, classes=output_len)


def mobilenetv2(input_shape, output_len):
    return MobileNetV2(weights=None, input_shape=input_shape, classes=output_len)


def inceptionv3(input_shape, output_len):
    return InceptionV3(weights=None, input_shape=input_shape, classes=output_len)
