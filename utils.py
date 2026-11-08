import numpy as np  

def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def convolve(img, kernel): #technically cross correlation 
    kernel_count, kernel_depth, kernel_rows, kernel_cols = kernel.shape
    img_depth, img_rows, img_cols = img.shape
    output = np.zeros((kernel_count,img_rows - kernel_rows + 1, img_cols - kernel_cols + 1))
    if(kernel_depth != img_depth):
        raise ValueError("kerenl depth must match image depth bruh")
    
    for k in range(kernel_count):
        for d in range(kernel_depth):
            for i in range(img_rows - kernel_rows + 1):
                for j in range(img_cols - kernel_cols + 1):
                    output[k, i, j] = np.sum(img[d, i:i + kernel_rows, j:j + kernel_cols] * kernel[k,d]) #hadamard product -> sum
    return output