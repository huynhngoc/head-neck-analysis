"""
Sample code to generate different architecture.
Comment out the unwanted piece and tune the paramters to generate different architectures.
Tuning parameters:
For Densenet:
  - n_filter
  - dense_block
  - batchnorm
  - dropout_rate
For ResNet:
  - n_upsampling (3 or 4. Deeper network may be hard to train)
  - n_filter
  - batchnorm
  - dropout_rate
"""


from deoxys.automation.architecture import generate_unet_architecture_json, \
    generate_densenet_2d_json, generate_resnet_json, \
    generate_vnet_architecture_json, generate_voxresnet_json

if __name__ == "__main__":
    # Unet
    # generate_unet_architecture_json(
    #     '../architecture.json',  # the filename
    #     n_upsampling=4,  # number of downsampling layers
    #     n_filter=64,
    #     # n_filter: number of filters in the 1st conv layer, this number is doubled in each downsampling step.
    #     # With n_upsampling=4 and n_filter=64, the number of filters in the Unet will go 64 - 128 - 256 - 512 - 1024 - 512 - 256 - 128 - 64
    #     # You can provide a list of number filters associate with each block in the Unet.
    #     # The list should contain {n_upsampling + 1} items, covering the downsampling path and the bottleneck.
    #     # For example, in the case of n_upsampling=4, n_filter could be [64, 80, 96, 112, 128]
    #     batchnorm=True,
    #     activation='relu',
    #     dropout_rate=0,
    #     # When dropout_rate = 0, there are no dropout layers.
    #     # Dropout layers are placed after each maxpooling layer and after concatenation in the skip-connection
    #     kernel=3,  # size of the kernel of every convolutional layer
    #     stride=1,  # stride use in the upconv layer
    #     n_class=2  # number of labels
    # )

    # # Densenet
    # generate_densenet_2d_json(
    #     '../architecture.json',  # the filename
    #     n_upsampling=4,  # number of downsampling layers (transition down step)
    #     n_filter=48,
    #     # number of filters in the 1st conv layer. This number increases by 16 in each downsampling step.
    #     # With n_upsampling=4 and n_filter=48, the number of filters in the Dense will go 48 - 64 - 80 - 96 - 112 - 96 - 80 - 64 - 48
    #     # You can provide a list of number of filters associate with each block in the DenseNet.
    #     # The list should contain {n_upsampling + 1} items.
    #     # For example, in the case of n_upsampling=4, n_filter could be [48, 64, 80, 96, 112]
    #     dense_block=3,
    #     # number of conv layers in the 1st dense block. This number increases by 1 in each downsampling step
    #     # Similar to n_filter, you can provide a list of number associated with each dense block in the DenseNet.
    #     # The list should contain {n_upsampling + 1} items.
    #     # For example, in the case of n_upsampling=4, dense_block could be [3, 4, 5, 6, 7] or [3, 3, 4, 4, 5]
    #     batchnorm=False,  # The original paper did not use batch normalization
    #     activation='relu',
    #     dropout_rate=0,
    #     kernel=3,
    #     stride=2,  # number of strides use in each downsampling steps and upsampling step
    #     n_class=2
    # )

    # # Resnet
    # generate_resnet_json(
    #     '../architecture.json',
    #     n_upsampling=3,
    #     n_filter=64,  # number of filters in all conv layers
    #     # number of residual block after each downsampling step (maxpool)
    #     res_block=2,
    #     batchnorm=True,
    #     activation='relu',
    #     dropout_rate=0,
    #     kernel=3,
    #     stride=2,
    #     n_class=2
    # )

    # for f in [32, 48, 64]:
    #     for n in [3, 4]:
    #         generate_unet_architecture_json(
    #             # the filename
    #             f'templates/architecture/2d_unet_{f}_D{n}.json',
    #             n_upsampling=n,  # number of downsampling layers
    #             n_filter=f,
    #             # n_filter: number of filters in the 1st conv layer, this number is doubled in each downsampling step.
    #             # With n_upsampling=4 and n_filter=64, the number of filters in the Unet will go 64 - 128 - 256 - 512 - 1024 - 512 - 256 - 128 - 64
    #             # You can provide a list of number filters associate with each block in the Unet.
    #             # The list should contain {n_upsampling + 1} items, covering the downsampling path and the bottleneck.
    #             # For example, in the case of n_upsampling=4, n_filter could be [64, 80, 96, 112, 128]
    #             batchnorm=True,
    #             activation='relu',
    #             dropout_rate=0,
    #             # When dropout_rate = 0, there are no dropout layers.
    #             # Dropout layers are placed after each maxpooling layer and after concatenation in the skip-connection
    #             kernel=3,  # size of the kernel of every convolutional layer
    #             stride=1,  # stride use in the upconv layer
    #             n_class=2  # number of labels
    #         )

    # for f in [32, 48, 64]:
    #     for n in [3, 4]:
    #         generate_vnet_architecture_json(
    #             # the filename
    #             f'templates/architecture/3d_unet_{f}_D{n}.json',
    #             n_upsampling=n,  # number of downsampling layers
    #             n_filter=f,
    #             # n_filter: number of filters in the 1st conv layer, this number is doubled in each downsampling step.
    #             # With n_upsampling=4 and n_filter=64, the number of filters in the Unet will go 64 - 128 - 256 - 512 - 1024 - 512 - 256 - 128 - 64
    #             # You can provide a list of number filters associate with each block in the Unet.
    #             # The list should contain {n_upsampling + 1} items, covering the downsampling path and the bottleneck.
    #             # For example, in the case of n_upsampling=4, n_filter could be [64, 80, 96, 112, 128]
    #             batchnorm=True,
    #             activation='relu',
    #             dropout_rate=0,
    #             # When dropout_rate = 0, there are no dropout layers.
    #             # Dropout layers are placed after each maxpooling layer and after concatenation in the skip-connection
    #             kernel=3,  # size of the kernel of every convolutional layer
    #             stride=2,  # stride use in the upconv layer
    #             n_class=2  # number of labels
    #         )

    # for f in [32, 48, 64]:
    #     for n in [3, 4]:
    #         generate_voxresnet_json(
    #             f'templates/architecture/3d_resnet_{f}_D{n}.json',
    #             n_upsampling=n,
    #             n_filter=f,  # number of filters in all conv layers
    #             # number of residual block after each downsampling step (maxpool)
    #             res_block=2,
    #             batchnorm=True,
    #             activation='relu',
    #             dropout_rate=0,
    #             kernel=3,
    #             stride=2,
    #             n_class=2
    #         )

    for f in [32, 48, 64]:
        for n in [3, 4]:
            generate_unet_architecture_json(
                # the filename
                f'templates/architecture/2d_unet_{f}_D{n}_dropout50.json',
                n_upsampling=n,  # number of downsampling layers
                n_filter=f,
                # n_filter: number of filters in the 1st conv layer, this number is doubled in each downsampling step.
                # With n_upsampling=4 and n_filter=64, the number of filters in the Unet will go 64 - 128 - 256 - 512 - 1024 - 512 - 256 - 128 - 64
                # You can provide a list of number filters associate with each block in the Unet.
                # The list should contain {n_upsampling + 1} items, covering the downsampling path and the bottleneck.
                # For example, in the case of n_upsampling=4, n_filter could be [64, 80, 96, 112, 128]
                batchnorm=True,
                activation='relu',
                dropout_rate=0.5,
                # When dropout_rate = 0, there are no dropout layers.
                # Dropout layers are placed after each maxpooling layer and after concatenation in the skip-connection
                kernel=3,  # size of the kernel of every convolutional layer
                stride=1,  # stride use in the upconv layer
                n_class=2  # number of labels
            )
