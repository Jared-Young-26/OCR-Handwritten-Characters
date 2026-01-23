import numpy as np

def conv2D_layer(X, kernel):
    """
    Performs convolution with X and kernel where both are two dimensional
    X is a single layer of a 3D feature and kernel is a single layer of a 4D 
    kernel
    Used as a helper function for conv2D
    """
    if len(X.shape) != 2 or len(kernel.shape) != 2:
        print(X.shape, kernel.shape)
        raise Exception("Input and kernel must be two dimensional")

    X_height = X.shape[0]
    X_width = X.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    map_shape = (X_height - kernel_height + 1, X_width - kernel_width + 1)
    map_height = map_shape[0]
    map_width = map_shape[1]

    # Maybe just ignore out of bounds results? For future thought
    if map_height < 0 or map_width < 0:
        raise Exception("Kernel size must be less or equal to imput size")

    feature_map = np.zeros(map_shape)
    for i in range(map_height):
        for j in range(map_width):
            val = sum(
                X[i + x, j + y] * kernel[x, y] 
                for x in range(kernel_height) 
                for y in range(kernel_width)
            )  
            feature_map[i, j] = val
    return feature_map

def conv2D(X, kernel):
    """
    This function performs the actual convolution on a 3D input X and a 4D 
    kernel kernel
    """
    if len(X.shape) != 3 or len(kernel.shape) != 4:
        print(X.shape, kernel.shape)
        raise Exception("Input must be 3D and kernel must be 4D")

    channel_in = X.shape[0]
    X_height = X.shape[1]
    X_width = X.shape[2]
    channel_out = kernel.shape[0]
    kernel_height = kernel.shape[2]
    kernel_width = kernel.shape[3]

    if channel_in != kernel.shape[1]:
        raise Exception("Number of input channels in both input and kernel must be equal")

    arrs = [
        [
            conv2D_layer(X[j], kernel[i][j]) 
            for j in range(channel_in)
        ] 
        for i in range(channel_out)
    ]

    layers = [np.sum(arr, axis = 0) for arr in arrs]
    return np.array(layers)
    
def main(debug = False):
    channel_in = 4
    channel_out = 3
    image_height = 3
    image_width = 3
    kernel_height = 2
    kernel_width = 2

    if not debug:
        X = np.random.rand(channel_in, image_height, image_width)
        kernel = np.random.rand(channel_out, channel_in, kernel_height, kernel_width)
    else:
        X = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]) 
        kernel = np.array([[[[0, 1], [2, 3]], [[1, 2], [3, 4]]]])    
 
    
    print(conv2D(X, kernel))

if __name__ == "__main__":
    main(debug = False) 
