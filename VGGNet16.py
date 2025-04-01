import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# image preprocessing - crop from 
img = Image.open('/home2/jeongsj/ComputerVision/BADUK.jpg')
preprocess = transforms.Compose([
    transforms.Resize(224), # 가로/세로 비율을 유지하면서, 작은 변의 길이로 크기를 줄임. 
    transforms.CenterCrop(224), # 이미지의 중앙 부분을 입력 크기로 잘라냄.
    transforms.ToTensor(), # (type) NumPy, PIL -> Tensor
    #transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406], # 평균으로 정규화 (괄호 안의 수의 개수 = 채널 수)
    #    std=[0.229, 0.224, 0.225] # 표준 편차로 정규화 (괄호 안의 수의 개수 = 채널 수)
    #)
])

print(img)
img_t = preprocess(img)
#print(img_t)
#print(type(img_t))
#print(img_t.dtype)
#print(img_t.shape)
'''
# 3. Tensor → PIL 이미지 변환 (HWC 형식 필요)
to_pil = transforms.ToPILImage()
image_pil = to_pil(img_t)

# 4. JPG로 저장
image_pil.save("/home2/jeongsj/ComputerVision/output.jpg", format="JPEG")

print("JPG 이미지 저장 완료!")
'''

# Feature Extraction
class Convolution_Layer:
    '''
    VGGNet16 구현 클래스 임.
    input: (RGB) Image[3, 224, 224] {3 dimension}
    output: (Prediction) Probability[1000, ] {1 dimension}
    Progress: input ->       -> Softmax -> ouput
    image: input
    filter: weight (parameter)
    '''
    def __init__(self, num_of_channel, filter_size, stride, padding_size): # 64(input channel), (3, 3)(filter size), 1(stride), 1(padding size)
        #self.input_t = input_t # 입력 (image / previous feature map)
        #self.input_size = (224, 224, 3) # 224 x 224 x RGB 이미지
        #self.convolutiopn_filter_3 = (3, 3) # non-linearity
        #self.convolutiopn_filter_1 = (1, 1) # linear transformation
        #self.convolutiopn_stride = 1
        #self.padding_of_conv = 1 # (3, 3) convolution layer

        #self.pooling_window = (2, 2) # max-pooling
        #self.pooling_stride = 2

        self.num_of_channel = num_of_channel
        self.filter_size = filter_size
        self.stride = stride
        self.padding_size = padding_size
        
    def convoltion_layer(self, input_t):
        # input: input_t[channel, height, width]
        # output: feature map[channel, height, width]
        # Progress: input -> 1.add padding -> 2.make filter(weights) -> 3.calculate output(feature map) size -> 4.convolution(by receptive field) -> ouput
        channel, height, width = input_t.shape
        f_channel = self.num_of_channel
        f_height, f_width = self.filter_size
        
        # 1. Add padding to input_t
        padded_input = torch.zeros((channel, height + 2 * self.padding_size, width + 2 * self.padding_size))
        padded_input[ : , self.padding_size:height+self.padding_size, self.padding_size:width+self.padding_size] = input_t # 0 심은 밭에 이미지를 콱 박아버림.
        
        # 2. Make filter(weights) 
        filters = torch.randn(f_channel, channel, f_height, f_width)
        
        # 3. Calculate output(feature map) size
        output_h = int((height + 2 * self.padding_size - f_height) / self.stride + 1)
        output_w = int((width + 2 * self.padding_size - f_width) / self.stride + 1)
        
        # 4. Convolution Calculator - sliding window 방식으로 작동
        # => 입력 이미지를 고정하고, 필터 슬라이딩 = 필터 고정하고, 이미지 슬라이딩 (locally)
        feature_maps = torch.zeros(f_channel, output_h, output_w)
        
        for f_c in range(f_channel): # 필터별 순회
            for f_h in range(0, output_h, self.stride): # 필터의 높이별 순회
                for f_w in range(0, output_w, self.stride): # 필터의 너비별 순회
                    receptive_field = padded_input[:, f_h:f_h+f_height, f_w:f_w+f_width]
                    feature_maps[f_c, f_h, f_w] = torch.sum(receptive_field * filters[f_c])
                               
        return feature_maps
        
    def normalization(self, input): # not employ normalization.
        # input:
        # output:
        # Progress: input -> Local Response Normalization(LRN) -> ouput
        pass
    
    def ReLU(self, input):
        # input: 
        # output:
        # Progress: input -> max(input, 0) -> ouput
        output = np.maximum(input, 0)
        return output
    
# Subsampling (Downsampling)
class Pooling_Layer:
    def __init__(self, filter_size, stride): # (2, 2)(pooling size), 2(pooling stride)
        #self.input_t = input_t # 입력 (image / previous feature map)
        #self.input_size = (224, 224, 3) # 224 x 224 x RGB 이미지
        #self.convolutiopn_filter_3 = (3, 3) # non-linearity
        #self.convolutiopn_filter_1 = (1, 1) # linear transformation
        #self.convolutiopn_stride = 1
        #self.padding_of_conv = 1 # (3, 3) convolution layer

        #self.pooling_window = (2, 2) # max-pooling
        #self.pooling_stride = 2

        self.filter_size = filter_size
        self.stride = stride
    
    def max_pooling_layer(self, input):
        # input:
        # output:
        # Progress: input -> max pooling -> ouput
        channel, height, width = input.shape
        f_height, f_width = self.filter_size
        
        # 1. Calculate output(pooling activation map) size
        output_h = height // self.stride
        output_w = width // self.stride
        
        # 2. Max pooling Calculator - sliding window 방식으로 작동
        pooling_activation_map = torch.zeros(channel, output_h, output_w)
        
        for c in range(0, channel): # 채널별 순회
            for f_h in range(0, output_h, self.stride): # 채널의 높이별 순회
                for f_w in range(0, output_w, self.stride): # 채널의 너비별 순회
                    receptive_field = input[c, f_h:f_h+f_height, f_w:f_w+f_width]
                    pooling_activation_map[c, f_h, f_w] = torch.max(receptive_field)
                    
        return pooling_activation_map
    
class Flatten_Layer:
    def __init__(self):
        pass
    
    def flatten_layer(self, input): # n차원 -> 1차원 # pooling size, pooling stride
        # input: feature map
        # output: 1차원
        # Progress: input ->  -> ouput
        output = input.view(-1)
        return output
        
class Fully_Connected_Layer:
    def __init__(self, shape): 
        self.shape = shape

    def fully_connected_layer(self, input):
        # input: 1차원
        # output:
        # Progress: input ->  -> ouput
        #output = input * w + b
        #loss = (output - y)**2
        #w = w + learning_rate + dloss/dw
        #b = b + learning_rate + dloss/db
        #return
        len_of_input, len_of_output = self.shape
        w = torch.randn(len_of_output, len_of_input)
        b = torch.randn(len_of_output)
        output = torch.matmul(w, input) + b
        return output
    
    def ReLU(self, input):
        # input: 
        # output:
        # Progress: input -> max(input, 0) -> ouput
        output = np.maximum(input, 0)
        return output
    
    def Softmax(self, input):
        # input: 1차원 tensor
        # output: 1차원 tensor
        # Progress: input -> exp_x[i] / sum(exp_x[0~n]) -> ouput 
        max_x = torch.max(input)
        exp_x = torch.exp(input - max_x)
        sum_exp_x = torch.sum(exp_x)
        output = exp_x / sum_exp_x
        return output 
    
class Axtivation_Function:
    def __init__(self):
        pass
    
    def ReLU(self, input):
        # input: 
        # output:
        # Progress: input -> max(input, 0) -> ouput
        output = np.maximum(input, 0)
        return output
    
    def Softmax(self, input):
        # input: 1차원 tensor
        # output: 1차원 tensor
        # Progress: input -> exp_x[i] / sum(exp_x[0~n]) -> ouput 
        exp_x = np.exp(input)
        sum_exp_x = np.sum(exp_x)
        output = exp_x / sum_exp_x
        return output

class VGGNet16:
    '''
    VGGNet16 구현 클래스 임.
    input: (RGB) Image[3, 224, 224] {3 dimension}
    output: (Prediction) Probability[1000, ] {1 dimension}
    Progress: input ->       -> Softmax -> ouput    
    '''
    def __init__(self, input_t):
        self.input_t = input_t
        self.input_size = (224, 224, 3) # 224 x 224 x RGB 이미지
        self.convolutiopn_filter_3 = (3, 3) # non-linearity
        self.convolutiopn_filter_1 = (1, 1) # linear transformation
        self.convolutiopn_stride = 1
        self.padding_of_conv = 1 # (3, 3) convolution layer

        self.pooling_window = (2, 2) # max-pooling
        self.pooling_stride = 2

    def MLRLoss(self, input): #multinomial logistic regression objective
        #return loss = nn.CrossEntropyLoss()
        pass
        
    
    def forward(self, input):
        # input:
        # output:
        # Progress: input ->  -> ouput
        #return
        pass
        

# input
print(img_t.shape)

# Feature Extractor #1
conv3_64_1_1 = Convolution_Layer(64, (3, 3), 1, 1)
conv3_64_1_1_feature_map = conv3_64_1_1.convoltion_layer(img_t) # number of filter channel = 64
conv3_64_1_1_activation_map = conv3_64_1_1.ReLU(conv3_64_1_1_feature_map)
print("#1 conv3-64의 첫번째 'feature map' shape: ", conv3_64_1_1_feature_map.shape)
print("#1 conv3-64의 첫번째 'activation map' shape: ", conv3_64_1_1_activation_map.shape)
conv3_64_1_2 = Convolution_Layer(64, (3, 3), 1, 1)
conv3_64_1_2_feature_map = conv3_64_1_2.convoltion_layer(conv3_64_1_1_activation_map) # number of filter channel = 64
conv3_64_1_2_activation_map = conv3_64_1_2.ReLU(conv3_64_1_2_feature_map)
print("#1 conv3-64의 두번째 'feature map' shape: ", conv3_64_1_2_feature_map.shape)
print("#1 conv3-64의 두번째 'activation map' shape: ", conv3_64_1_2_activation_map.shape)
max_pooling_1 = Pooling_Layer((2, 2), 2) # pooling size, pooling stride
max_pooling_1_activation_map = max_pooling_1.max_pooling_layer(conv3_64_1_2_activation_map)
print("#1 max pooling의 'activation map' shape: ", max_pooling_1_activation_map.shape)

# Feature Extractor #2
conv3_128_2_1 = Convolution_Layer(128, (3, 3), 1, 1)
conv3_128_2_1_feature_map = conv3_128_2_1.convoltion_layer(max_pooling_1_activation_map) # number of filter channel = 128
conv3_128_2_1_activation_map = conv3_128_2_1.ReLU(conv3_128_2_1_feature_map)
print("#2 conv3-128의 첫번째 'feature map' shape: ", conv3_128_2_1_feature_map.shape)
print("#2 conv3-128의 첫번째 'activation map' shape: ", conv3_128_2_1_activation_map.shape)
conv3_128_2_2 = Convolution_Layer(128, (3, 3), 1, 1)
conv3_128_2_2_feature_map = conv3_128_2_2.convoltion_layer(conv3_128_2_1_activation_map) # number of filter channel = 128
conv3_128_2_2_activation_map = conv3_128_2_2.ReLU(conv3_128_2_2_feature_map)
print("#2 conv3-128의 두번째 'feature map' shape: ", conv3_128_2_2_feature_map.shape)
print("#2 conv3-128의 두번째 'activation map' shape: ", conv3_128_2_2_activation_map.shape)
max_pooling_2 = Pooling_Layer((2, 2), 2)
max_pooling_2_activation_map = max_pooling_2.max_pooling_layer(conv3_128_2_2_activation_map)
print("#2 max pooling의 'activation map' shape: ", max_pooling_2_activation_map.shape)

# Feature Extractor #3
conv3_256_3_1 = Convolution_Layer(256, (3, 3), 1, 1)
conv3_256_3_1_feature_map = conv3_256_3_1.convoltion_layer(max_pooling_2_activation_map) # number of filter channel = 256
conv3_256_3_1_activation_map = conv3_256_3_1.ReLU(conv3_256_3_1_feature_map)
print("#3 conv3-256의 첫번째 'feature map' shape: ", conv3_256_3_1_feature_map.shape)
print("#3 conv3-256의 첫번째 'activation map' shape: ", conv3_256_3_1_activation_map.shape)
conv3_256_3_2 = Convolution_Layer(256, (3, 3), 1, 1)
conv3_256_3_2_feature_map = conv3_256_3_2.convoltion_layer(conv3_256_3_1_activation_map) # number of filter channel = 256
conv3_256_3_2_activation_map = conv3_256_3_2.ReLU(conv3_256_3_2_feature_map)
print("#3 conv3-256의 두번째 'feature map' shape: ", conv3_256_3_2_feature_map.shape)
print("#3 conv3-256의 두번째 'activation map' shape: ", conv3_256_3_2_activation_map.shape)
conv3_256_3_3 = Convolution_Layer(256, (3, 3), 1, 1)
conv3_256_3_3_feature_map = conv3_256_3_3.convoltion_layer(conv3_256_3_2_activation_map) # number of filter channel = 256
conv3_256_3_3_activation_map = conv3_256_3_3.ReLU(conv3_256_3_3_feature_map)
print("#3 conv3-256의 세번째 'feature map' shape: ", conv3_256_3_3_feature_map.shape)
print("#3 conv3-256의 세번째 'activation map' shape: ", conv3_256_3_3_activation_map.shape)
max_pooling_3 = Pooling_Layer((2, 2), 2)
max_pooling_3_activation_map = max_pooling_3.max_pooling_layer(conv3_256_3_3_activation_map)
print("#3 max pooling의 'activation map' shape: ", max_pooling_3_activation_map.shape)

# Feature Extractor #4
conv3_512_4_1 = Convolution_Layer(512, (3, 3), 1, 1)
conv3_512_4_1_feature_map = conv3_512_4_1.convoltion_layer(max_pooling_3_activation_map) # number of filter channel = 512
conv3_512_4_1_activation_map = conv3_512_4_1.ReLU(conv3_512_4_1_feature_map)
print("#4 conv3-512의 첫번째 'feature map' shape: ", conv3_512_4_1_feature_map.shape)
print("#4 conv3-512의 첫번째 'activation map' shape: ", conv3_512_4_1_activation_map.shape)
conv3_512_4_2 = Convolution_Layer(512, (3, 3), 1, 1)
conv3_512_4_2_feature_map = conv3_512_4_2.convoltion_layer(conv3_512_4_1_activation_map) # number of filter channel = 512
conv3_512_4_2_activation_map = conv3_512_4_2.ReLU(conv3_512_4_2_feature_map)
print("#4 conv3-512의 두번째 'feature map' shape: ", conv3_512_4_2_feature_map.shape)
print("#4 conv3-512의 두번째 'activation map' shape: ", conv3_512_4_2_activation_map.shape)
conv3_512_4_3 = Convolution_Layer(512, (3, 3), 1, 1)
conv3_512_4_3_feature_map = conv3_512_4_3.convoltion_layer(conv3_512_4_2_activation_map) # number of filter channel = 512
conv3_512_4_3_activation_map = conv3_512_4_3.ReLU(conv3_512_4_3_feature_map)
print("#4 conv3-512의 세번째 'feature map' shape: ", conv3_512_4_3_feature_map.shape)
print("#4 conv3-512의 세번째 'activation map' shape: ", conv3_512_4_3_activation_map.shape)
max_pooling_4 = Pooling_Layer((2, 2), 2)
max_pooling_4_activation_map = max_pooling_4.max_pooling_layer(conv3_512_4_3_activation_map)
print("#4 max pooling의 'activation map' shape: ", max_pooling_4_activation_map.shape)

# Feature Extractor #5
conv3_512_5_1 = Convolution_Layer(512, (3, 3), 1, 1)
conv3_512_5_1_feature_map = conv3_512_5_1.convoltion_layer(max_pooling_4_activation_map) # number of filter channel = 512
conv3_512_5_1_activation_map = conv3_512_5_1.ReLU(conv3_512_5_1_feature_map)
print("#5 conv3-512의 첫번째 'feature map' shape: ", conv3_512_5_1_feature_map.shape)
print("#5 conv3-512의 첫번째 'activation map' shape: ", conv3_512_5_1_activation_map.shape)
conv3_512_5_2 = Convolution_Layer(512, (3, 3), 1, 1)
conv3_512_5_2_feature_map = conv3_512_5_2.convoltion_layer(conv3_512_5_1_activation_map) # number of filter channel = 512
conv3_512_5_2_activation_map = conv3_512_5_2.ReLU(conv3_512_5_2_feature_map)
print("#5 conv3-512의 두번째 'feature map' shape: ", conv3_512_5_2_feature_map.shape)
print("#5 conv3-512의 두번째 'activation map' shape: ", conv3_512_5_2_activation_map.shape)
conv3_512_5_3 = Convolution_Layer(512, (3, 3), 1, 1)
conv3_512_5_3_feature_map = conv3_512_5_3.convoltion_layer(conv3_512_5_2_activation_map) # number of filter channel = 512
conv3_512_5_3_activation_map = conv3_512_5_3.ReLU(conv3_512_5_3_feature_map)
print("#5 conv3-512의 세번째 'feature map' shape: ", conv3_512_5_3_feature_map.shape)
print("#5 conv3-512의 세번째 'activation map' shape: ", conv3_512_5_3_activation_map.shape)
max_pooling_5 = Pooling_Layer((2, 2), 2)
max_pooling_5_activation_map = max_pooling_5.max_pooling_layer(conv3_512_5_3_activation_map)
print("#5 max pooling의 'activation map' shape: ", max_pooling_5_activation_map.shape)

# Flatten Layer
flatten = Flatten_Layer()
flatten_1_dim = flatten.flatten_layer(max_pooling_5_activation_map)
print("Flatten의 '1 dim' shape: ", flatten_1_dim.shape)

# Fully Connected Layer #1
fc_4096_1 = Fully_Connected_Layer((len(flatten_1_dim), 4096)) # number of channel = 4096
fc_4096_1_output = fc_4096_1.fully_connected_layer(flatten_1_dim)
fc_4096_1_output = fc_4096_1.ReLU(fc_4096_1_output)
print("#1 fc_4096의 result shape: ", fc_4096_1_output.shape)

# Fully Connected Layer #2
fc_4096_2 = Fully_Connected_Layer((4096, 4096)) # number of channel = 4096
fc_4096_2_output = fc_4096_2.fully_connected_layer(fc_4096_1_output)
fc_4096_2_output = fc_4096_2.ReLU(fc_4096_2_output)
print("#2 fc_4096의 result shape: ", fc_4096_2_output.shape)

# Fully Connected Layer #3
fc_1000_3 = Fully_Connected_Layer((4096, 1000)) # number of channel = 1000
fc_1000_3_output = fc_1000_3.fully_connected_layer(fc_4096_2_output)
fc_1000_3_output = fc_1000_3.Softmax(fc_1000_3_output)
print("#3 fc_1000의 result shape: ", fc_1000_3_output.shape)

# output
print("VGGNet16 최종 outout shape: ")
#print(fc_1000_3_output)
print(fc_1000_3_output.shape)

'''
def main():
    pass

if __name__ == '__main__':
    main()
'''
