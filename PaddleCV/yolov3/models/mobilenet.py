import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay


def conv_bn(input,
            filter_size,
            num_filters,
            stride,
            padding,
            channels=None,
            num_groups=1,
            act='relu',
            use_cudnn=True,
            is_test=True,
            name=None):
    parameter_attr = ParamAttr(initializer=MSRA(),
                               name=name + '_weights')
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None,
        use_cudnn=use_cudnn,
        param_attr=parameter_attr,
        bias_attr=False)

    bn_name = name + "_bn"
    bn = fluid.layers.batch_norm(input=conv, 
                                 act=act,
                                 is_test=is_test,
            			 param_attr=ParamAttr(name=bn_name + "_scale",
                                                      regularizer=L2Decay(0.)),
            			 bias_attr=ParamAttr(name=bn_name + "_offset",
                                                     regularizer=L2Decay(0.)),
            			 moving_mean_name=bn_name + '_mean',
            			 moving_variance_name=bn_name + '_variance')
    
    return bn


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride,
                        scale, is_test=True, name=None):
    depthwise_conv = conv_bn(
        input=input,
        filter_size=3,
        num_filters=int(num_filters1 * scale),
        stride=stride,
        padding=1,
        num_groups=int(num_groups * scale),
        use_cudnn=False,
        is_test=is_test,
        name=name+"_dw")

    pointwise_conv = conv_bn(
        input=depthwise_conv,
        filter_size=1,
        num_filters=int(num_filters2 * scale),
        stride=1,
        padding=0,
        is_test=is_test,
        name=name+"_sep")
    return pointwise_conv



def mobile_net(img, scale=1.0, is_test=True):
    blocks = []
    # 608x608
    tmp = conv_bn(img, 3, int(32 * scale), 2, 1, 3, is_test=is_test, name='conv1')
    # 304x304
    tmp = depthwise_separable(tmp, 32, 64, 32, 1, scale, is_test=is_test, name='conv2_1')
    tmp = depthwise_separable(tmp, 64, 128, 64, 2, scale, is_test=is_test, name='conv2_2')
    # 76x76
    tmp = depthwise_separable(tmp, 128, 128, 128, 1, scale, is_test=is_test, name='conv3_1')
    tmp = depthwise_separable(tmp, 128, 256, 128, 2, scale, is_test=is_test, name='conv3_2')
    # 1/8
    # 38x38
    blocks.append(tmp)
    tmp = depthwise_separable(tmp, 256, 256, 256, 1, scale, is_test=is_test, name='conv4_1')
    tmp = depthwise_separable(tmp, 256, 512, 256, 2, scale, is_test=is_test, name='conv4_2')

    # 1/16
    blocks.append(tmp)
    for i in range(5):
        tmp = depthwise_separable(tmp, 512, 512, 512, 1, scale, is_test=is_test, name='conv5_'+str(i+1))

    # 19x19
    tmp = depthwise_separable(tmp, 512, 1024, 512, 2, scale, is_test=is_test, name='conv5_6')
    tmp = depthwise_separable(tmp, 1024, 1024, 1024, 1, scale, is_test=is_test, name='conv6')
    # 1/32
    blocks.append(tmp)  

    return blocks[::-1]
