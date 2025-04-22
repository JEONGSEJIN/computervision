import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''   
# Feature Extraction
class Convolution_Layer:
    #VGGNet16 구현 클래스 임.
    #input: (RGB) Image[3, 224, 224] {3 dimension}
    #output: (Prediction) Probability[1000, ] {1 dimension}
    #Progress: input ->       -> Softmax -> ouput
    #image: input
    #filter: weight (parameter)
    def __init__(self, input_t, num_of_channel, filter_size, stride, padding_size): # 64(input channel), (3, 3)(filter size), 1(stride), 1(padding size)
        self.input_t = input_t # 입력 (image / previous feature map)
        self.channel, self.height, self.width = input_t.shape # 224 x 224 x RGB 이미지
        self.stride = stride
        self.padding_size = padding_size
        self.f_channel = num_of_channel
        self.f_height, self.f_width = filter_size
        # 2. Make filter(weights) 
        self.filters = nn.Parameter(torch.randn(self.f_channel, self.channel, self.f_height, self.f_width))
        
    def convoltion_layer(self):
        # input: input_t[channel, height, width]
        # output: feature map[channel, height, width]
        # Progress: input -> 1.add padding -> (2.make filter(weights)) -> 3.calculate output(feature map) size -> 4.convolution(by receptive field) -> ouput
        
        # 1. Add padding to input_t
        padded_input = torch.zeros((self.channel, self.height + 2 * self.padding_size, self.width + 2 * self.padding_size))
        padded_input[ : , self.padding_size:self.height+self.padding_size, self.padding_size:self.width+self.padding_size] = self.input_t # 0 심은 밭에 이미지를 콱 박아버림.
        
        # 3. Calculate output(feature map) size
        output_h = int((self.height + 2 * self.padding_size - self.f_height) / self.stride + 1)
        output_w = int((self.width + 2 * self.padding_size - self.f_width) / self.stride + 1)
        
        # 4. Convolution Calculator - sliding window 방식으로 작동
        # => 입력 이미지를 고정하고, 필터 슬라이딩 = 필터 고정하고, 이미지 슬라이딩 (locally)
        feature_maps = torch.zeros(self.f_channel, output_h, output_w)
        
        for f_c in range(self.f_channel): # 필터별 순회
            for f_h in range(0, output_h, self.stride): # 필터의 높이별 순회
                for f_w in range(0, output_w, self.stride): # 필터의 너비별 순회
                    receptive_field = padded_input[:, f_h:f_h+self.f_height, f_w:f_w+self.f_width]
                    feature_maps[f_c, f_h, f_w] = torch.sum(receptive_field * self.filters[f_c])
                               
        return feature_maps
'''    
       
# Feature Extraction
class Convolution_Layer(nn.Module):
    '''
    VGGNet16 구현 클래스 임.
    input: (RGB) Image[3, 224, 224] {3 dimension}
    output: (Prediction) Probability[1000, ] {1 dimension}
    Progress: input ->       -> Softmax -> ouput
    image: input
    filter: weight (parameter)
    '''
    def __init__(self, input_channel, output_channel, filter_size, stride, padding_size=0): # 3(input channel), 64(output channel), (3, 3)(filter size), 1(stride), 1(padding size)
        super().__init__()
        #self.input_t = input_t # 입력 (image / previous feature map)
        #self.input_size = (224, 224, 3) # 224 x 224 x RGB 이미지
        #self.convolutiopn_filter_3 = (3, 3) # non-linearity
        #self.convolutiopn_filter_1 = (1, 1) # linear transformation
        #self.convolutiopn_stride = 1
        #self.padding_of_conv = 1 # (3, 3) convolution layer

        #self.pooling_window = (2, 2) # max-pooling
        #self.pooling_stride = 2
        self.input_channel = input_channel
        self.output_channel = output_channel # output_channel <- num_of_channel)
        self.filter_size = filter_size
        self.stride = stride
        self.padding_size = padding_size

        # 2. Make filter(weights) 
        self.f_height, self.f_width = self.filter_size     
        self.filters = nn.Parameter(torch.randn(self.output_channel, self.input_channel, self.f_height, self.f_width))
        self.bias = nn.Parameter(torch.randn(output_channel))
        
    def forward(self, input):
        # input: input_t[channel, height, width]
        # output: feature map[channel, height, width]
        # Progress: input -> 1.add padding -> 2.make filter(weights) -> 3.calculate output(feature map) size -> 4.convolution(by receptive field) -> ouput
        batch, channel, height, width = input.shape
        filters = self.filters.to(input.device)
        bias = self.bias.to(input.device)
        
        # 1. Add padding to input_t
        if self.padding_size > 0:
            padded_input = torch.zeros((batch, channel, height + 2 * self.padding_size, width + 2 * self.padding_size), device=device)
            padded_input[ : , : , self.padding_size:height+self.padding_size, self.padding_size:width+self.padding_size] = input # 0 심은 밭에 이미지를 콱 박아버림.
            input = padded_input
        
        # 3. Calculate output(feature map) size
        output_h = int((height + 2 * self.padding_size - self.f_height) / self.stride + 1)
        output_w = int((width + 2 * self.padding_size - self.f_width) / self.stride + 1)
        
        # 4. Convolution Calculator - sliding window 방식으로 작동
        # => 입력 이미지를 고정하고, 필터 슬라이딩 = 필터 고정하고, 이미지 슬라이딩 (locally)
        feature_maps = torch.zeros(batch, self.output_channel, output_h, output_w, device=device)
        
        for b in range(batch):
            for f_c in range(self.output_channel): # 필터별 순회
                for f_h in range(0, output_h, self.stride): # 필터의 높이별 순회
                    for f_w in range(0, output_w, self.stride): # 필터의 너비별 순회
                        receptive_field = input[b , :, f_h:f_h+self.f_height, f_w:f_w+self.f_width]
                        feature_maps[b, f_c, f_h, f_w] = torch.sum(receptive_field * self.filters[f_c])
                               
        return feature_maps
       
    def Local_response_normalization(self, input): # not employ normalization.
        # input:
        # output:
        # Progress: input -> Local Response Normalization(LRN) -> ouput
        pass
    
    def ReLU(self, input):
        # Progress: input -> max(input, 0) -> ouput
        #output = np.maximum(input, 0)
        # _, output = torch.max(input, 0)
        return F.relu(input)
    
# Subsampling (Downsampling)
class Pooling_Layer(nn.Module):
    def __init__(self, filter_size, stride, padding_size=0): # (2, 2)(pooling size), 2(pooling stride)
        super().__init__()
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
        self.padding_size = padding_size
    
    def max_pooling_layer(self, input):
        # input:
        # output:
        # Progress: input -> max pooling -> ouput
        batch, channel, height, width = input.shape
        f_height, f_width = self.filter_size
        device = input.device
        
        # 0. Add Padding to input_t
        if self.padding_size > 0:
            padded_input = torch.zeros((batch, channel, height + 2 * self.padding_size, width + 2 * self.padding_size), device=device)
            padded_input[ : , : , self.padding_size:height+self.padding_size, self.padding_size:width+self.padding_size] = input # 0 심은 밭에 이미지를 콱 박아버림.
            input = padded_input
            height += 2 * self.padding_size
            width += 2 * self.padding_size
        
        # 1. Calculate output(pooling activation map) size
        output_h = int((height - f_height) // self.stride + 1)
        output_w = int((width - f_width) // self.stride + 1)
        
        # 2. Max pooling Calculator - sliding window 방식으로 작동
        pooling_activation_map = torch.zeros(batch, channel, output_h, output_w, device=device)
        
        for b in range(batch):
            for c in range(0, channel): # 채널별 순회
                for f_h in range(0, output_h, self.stride): # 채널의 높이별 순회
                    for f_w in range(0, output_w, self.stride): # 채널의 너비별 순회
                        receptive_field = input[b, c, f_h:f_h+f_height, f_w:f_w+f_width]
                        pooling_activation_map[b, c, f_h, f_w] = torch.max(receptive_field)
                    
        return pooling_activation_map
    
    def average_pooling_layer(self, input):
        # input:
        # output:
        # Progress: input -> average pooling -> ouput
        batch, channel, height, width = input.shape
        f_height, f_width = self.filter_size
        device = input.device
        
        # 0. Add Padding to input_t
        if self.padding_size > 0:
            padded_input = torch.zeros((batch, channel, height + 2 * self.padding_size, width + 2 * self.padding_size), device=device)
            padded_input[ : , : , self.padding_size:height+self.padding_size, self.padding_size:width+self.padding_size] = input # 0 심은 밭에 이미지를 콱 박아버림.
            input = padded_input
            height += 2 * self.padding_size
            width += 2 * self.padding_size            
        
        # 1. Calculate output(pooling activation map) size
        output_h = int((height - f_height) // self.stride + 1)
        output_w = int((width - f_width) // self.stride + 1)
        
        # 2. Max pooling Calculator - sliding window 방식으로 작동
        pooling_activation_map = torch.zeros(batch, channel, output_h, output_w, device=device)
    
        for b in range(batch):
            for c in range(0, channel): # 채널별 순회
                for f_h in range(0, output_h, self.stride): # 채널의 높이별 순회
                    for f_w in range(0, output_w, self.stride): # 채널의 너비별 순회
                        receptive_field = input[b, c, f_h:f_h+f_height, f_w:f_w+f_width]
                        pooling_activation_map[b, c, f_h, f_w] = torch.mean(receptive_field)
                    
        return pooling_activation_map
        
class Flatten_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def flatten_layer(self, input): # n차원 -> 1차원 # pooling size, pooling stride
        # input: feature map
        # output: 1차원
        # Progress: input ->  -> ouput
        output = input.view(input.size(0), -1)
        return output
        
class Fully_Connected_Layer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.len_of_input, self.len_of_output = shape
        self.w = nn.Parameter(torch.randn(self.len_of_output, self.len_of_input))
        self.b = nn.Parameter(torch.randn(self.len_of_output))

    def fully_connected_layer(self, input):
        # input: 1차원
        # output:
        # Progress: input ->  -> ouput
        #output = input * w + b
        #loss = (output - y)**2
        #w = w + learning_rate + dloss/dw
        #b = b + learning_rate + dloss/db
        #return
        output = torch.matmul(input, self.w.t()) + self.b
        return output
    
    def ReLU(self, input):
        # input: 
        # output:
        # Progress: input -> max(input, 0) -> ouput
        #output = np.maximum(input, 0)
        return F.relu(input)
    
    def Softmax(self, input):
        # input: 1차원 tensor
        # output: 1차원 tensor
        # Progress: input -> exp_x[i] / sum(exp_x[0~n]) -> ouput 
        #max_x = torch.max(input).to(device)
        #exp_x = torch.exp(input - max_x).to(device)
        #sum_exp_x = torch.sum(exp_x).to(device)
        #output = exp_x / sum_exp_x
        return F.softmax(input, dim=-1) 
    
class Activation_Function(nn.Module):
    def __init__(self):
        super().__init__()
    
    def ReLU(self, input):
        # input: 
        # output:
        # Progress: input -> max(input, 0) -> ouput
        # output = torch.maximum(input, torch.tensor(0.0, device=input.device)) # input * (input > 0)
        #output = torch.relu(input).to(device)
        return F.relu(input)
    
    def Softmax(self, input):
        # input: 1차원 tensor
        # output: 1차원 tensor
        # Progress: input -> exp_x[i] / sum(exp_x[0~n]) -> ouput 
        #exp_x = np.exp(input)
        #sum_exp_x = np.sum(exp_x)
        #output = exp_x / sum_exp_x
        return F.softmax(input, dim=-1)

class Batch_normalization(nn.Module):
    def __init__(self, num_of_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_of_features = num_of_features
        self.eps = eps
        self.momentum = momentum

        # 학습 가능한 파라미터: scale (gamma), shift (beta)
        self.gamma = nn.Parameter(torch.ones(num_of_features))
        self.beta = nn.Parameter(torch.zeros(num_of_features))
        
        # running mean과 variance는 학습되지 않음
        self.register_buffer('running_mean', torch.zeros(num_of_features))
        self.register_buffer('running_var', torch.ones(num_of_features))
    
    # 나중에 코드 통합할 때, vggnet의 conv class 상속해서 batch norm 함수만 작성해주기?!
    # 근데, 입력 이미지 1장 / inference / test의 경우, 미니 배치의 평균, 분산 이용 못해서, 이거 사용 안해도됨.
    # inference에서는, 결과를 deterministic하게 하기 위하여 고정된 평균과 분산을 이용하여 정규화를 수행. (train mode - optimize the parameter / test mode - frozen parameter 따로)
    # 일단, 학습 이미지 1024장 있다고 가정하고 구현하자.
    # ** batch normalization에서, gamma, beta가 학습 가능한 변수들이라, 이 함수도 
    def forward(self, feature_map):
        # BN 의의: 각 레이어마다 정규화 하는 레이어를 두어, 변형된 분포가 나오지 않도록 조절하게 하는 것이 배치 정규화이다.
        # 방법: 미니 배치의 평균과 분산을 이용해서 정규화 한 뒤에, scale 및 shift를 감마, 베타를 통해 실행.
        device = feature_map.device 

        # i) 그냥 batch norm 구현
        if self.training:
            # (N, C, H, W)에서 C에 대해 평균과 분산 계산 (N: batch size)
            # 1. mini-batch mean 구하기
            mean = feature_map.mean(dim=(0, 2, 3), keepdim=True)
            # 2. mini-batch variance 구하기
            var = feature_map.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            
            # 러닝 평균/분산 업데이트
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)
            
            # 3. normalize (by mini-batch mean and variance)
            feature_norm = (feature_map - mean) / torch.sqrt(var + self.eps)

        else:
            # 러닝 평균/분산 업데이트
            mean = self.running_mean.view(1, -1, 1, 1).to(device)
            var = self.running_var.view(1, -1, 1, 1).to(device)
            feature_norm = (feature_map - mean) / torch.sqrt(var + self.eps)
        
        # 4. scale and shift (by gamma, beta: 학습 가능한 변수들, backpropagation을 통해서 학습함.)
        gamma = self.gamma.view(1, -1, 1, 1).to(device)
        beta = self.beta.view(1, -1, 1, 1).to(device)
        bn_output = gamma * feature_norm + beta

        # ii) pytorch 내장 함수 이용해서 구현
        #bn_norm_layer = nn.BatchNorm2d(self.num_of_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # image (N, C, H, W)
        #bn_output = bn_norm_layer(feature_map)
        
        return bn_output # 정규화된 값 -> 활성화 함수의 입력

class Basic_building_block(nn.Module):
    # Resnet 18 / 34
    def __init__(self):
        pass

class Bottleneck_building_block(nn.Module):
    # Resnet 50 / 101 / 152
    def __init__(self, input_channel, mid_channel, stride=1): # 3 / 4 / 6/ 3 / stride는 기본 1 고정
        super().__init__()
        #self.numbers_of_channel = numbers_of_channel  # channel of 1x1, channel of 3x3, channel of 1x1 
        self.input_channel = input_channel
        self.mid_channel = mid_channel
        self.output_channel = mid_channel * 4
        
        # [class 안에 class 객체 선언해서 돌리기] 
        # [2. convolution block + residual]
        # 1x1 conv
        self.conv_1 = Convolution_Layer(input_channel, mid_channel, (1, 1), 1) # 64(input channel), (1, 1)(filter size), 1(stride), x(padding size)
        self.conv_1_bn = Batch_normalization(mid_channel)
        self.conv_1_ac = Activation_Function()
        
        # 3x3 conv
        self.conv_2 = Convolution_Layer(mid_channel, mid_channel, (3, 3), stride, 1) # 64(input channel), (3, 3)(filter size), 1(stride), 1(padding size)
        self.conv_2_bn = Batch_normalization(mid_channel)
        self.conv_2_ac = Activation_Function()
        
        # class Bottle에서 블록 최종 출력 4배 키우기
        # 1x1 conv
        self.conv_3 = Convolution_Layer(mid_channel, self.output_channel, (1, 1), 1) # 64(input channel), (3, 3)(filter size), 1(stride), 1(padding size)
        self.conv_3_bn = Batch_normalization(mid_channel * 4)
        self.conv_3_ac = Activation_Function()

        # shortcut
        # Projection shortcut (x 크기 != F(x) 크기)
        # => (1 x 1)로 차원 맞춤.
        if input_channel != self.output_channel or stride != 1:
            self.shortcut_conv = Convolution_Layer(input_channel, self.output_channel, (1, 1), stride)
            self.shortcut_bn = Batch_normalization(self.output_channel)

        # Identity Shortcut (x 크기 = F(x) 크기)
        else:
            self.shortcut_conv = None
    
    # Conv -> Batch Normalization(DLWP4.3.5, 5.4.4) -> Activation Function
    def residual_function(self, input):
        #F = some_layers(x)  # 복잡한 함수 근사
        #return F + x        # residual + skip (identity)
        # [2. convolution block + residual = 3]
        
        residual = input # [N, 64, H, W]
        # 1x1 conv
        x = self.conv_1(input)
        x = self.conv_1_bn(x)
        x = self.conv_1_ac.ReLU(x)
        
        # 3x3 conv
        x = self.conv_2(x)
        x = self.conv_2_bn(x)
        x = self.conv_2_ac.ReLU(x)
        # 1x1 conv
        x = self.conv_3(x)
        x = self.conv_3_bn(x)

        # shortcut
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)
            residual = self.shortcut_bn(residual)
        
        # identity mapping
        x = x + residual #=> input, output이 맞나? tensor 크기가 일치하나?
        x = self.conv_3_ac.ReLU(x)
    
        return x
        
    # resnet50/152
    # 1x1 convolution 연산을 통해 차원을 스케일링 해주는 projection 방식 채택
    # 이유: 입력/출력 차원이 다를 때 스켕일링 해주기 위해서
    # projection 방식(Bottleneck building block): 네트워크를 깊게 만들어줄 때 학습 시간을 줄이기 위해 사용하는데, 위 두 구조다 비슷한 시간 복잡도를 가짐.
    # 차원 증가 시,
    # 1. 제로 패딩으로 차원 증가에 대응: 추가적인 파라미터 없어서 좋음.
    # 2. 1x1 convolution으로 차원 스케일링 해준 뒤 다시 원래 차원으로 스케일링 진행
class ResNet(nn.Module):
    def __init__(self, type_of_block, numbers_of_blocks): # type_of_block = basic / bottleneck
        super().__init__()
        # [1. convolution = 1]
        self.conv1 = Convolution_Layer(3, 64, (7, 7), 2, 3) # 64(output channel), (3, 3)(filter size), 1(stride), 3(padding size)
        self.conv1_bn = Batch_normalization(64)
        self.conv1_ac = Activation_Function()
        self.max_pool = Pooling_Layer((3, 3), 2, 1) 
        
        # [4. conv block stack block = 50]
        self.input_channel = 64 # conv1의 출력 채널
        self.conv2 = self.stacking_conv_blocks(type_of_block, numbers_of_blocks[0], 64, stride = 1) # conv2의 첫번째 output_channel = 64, 마지막은 *4 한 값
        self.conv3 = self.stacking_conv_blocks(type_of_block, numbers_of_blocks[1], 128, stride = 2) # conv3의 첫번째 output_channel = 128
        self.conv4 = self.stacking_conv_blocks(type_of_block, numbers_of_blocks[2], 256, stride = 2) # conv4의 첫번째 output_channel = 256
        self.conv5 = self.stacking_conv_blocks(type_of_block, numbers_of_blocks[3], 512, stride = 2) # conv5의 첫번째 output_channel = 512
        
        # global average => (7x7)입력 -> (1x1)출력
        self.aver_pool = Pooling_Layer((7, 7), 1)
        
        self.flatten = Flatten_Layer()
        self.fc = Fully_Connected_Layer(((512 if type_of_block == "basic" else 512 * 4), 1000))
        self.ac = Activation_Function()
        
    # (1x1conv->3x3conv->1x1conv) x 3,4,6,3
    def stacking_conv_blocks(self, type_of_block, num_of_blocks, output_channel, stride): # 64, 128, 256, 512
        blocks = []
        strides = [stride] + [1] * (num_of_blocks - 1)
        # [3. conv block stack = 9 / 12 / 18 / 6]
        for stride in strides:
            if type_of_block == "basic":
                blocks.append(Basic_building_block(self.input_channel, output_channel, stride)) 
                self.input_channel = output_channel
            if type_of_block == "bottleneck":
                blocks.append(Bottleneck_building_block(self.input_channel, output_channel, stride)) # 64,256 -> 256,256 -> 256,256 => 
                self.input_channel = output_channel * 4 # 앞 블록에서 출력을 4배 키워서 다음 블록의 입력은, 앞블록의 입력의 4배임.
        return nn.Sequential(*blocks)
    
    def forward(self, input):
        device = input.device
        
        # conv1 - output size: 112 x 112
        x = self.conv1(input)
        x = self.conv1_bn(x)
        x = self.conv1_ac.ReLU(x)
        print("#1 conv1의 'activation map' shape: ", x.shape)
        
        # max pooling
        x = self.max_pool.max_pooling_layer(x)
        print("#1 max pooling의 'activation map' shape: ", x.shape)
        
        # [conv2_x, conv3_x, conv4_x, conv5_x는 Bottleneck으로 구조가 같다]
        # conv2_x - output size: 56 x 56
        for block in self.conv2:
            x = block.residual_function(x) # 1x1conv->3x3conv->1x1conv
        print("#2 conv2의 block의 개수: ", len(self.conv2))
        print("#2 conv2의 'activation map' shape: ", x.shape)
            
        # conv3_x - output size: 56 x 56
        for block in self.conv3:
            x = block.residual_function(x) # 1x1conv->3x3conv->1x1conv
        print("#3 conv3의 block의 개수: ", len(self.conv3))    
        print("#3 conv3의 'activation map' shape: ", x.shape)
        
        # conv4_x - output size: 56 x 56
        for block in self.conv4:
            x = block.residual_function(x) # 1x1conv->3x3conv->1x1conv
        print("#4 conv4의 block의 개수: ", len(self.conv4))
        print("#4 conv4의 'activation map' shape: ", x.shape)
        
        # conv5_x - output size: 56 x 56
        for block in self.conv5:
            x = block.residual_function(x) # 1x1conv->3x3conv->1x1conv
        print("#5 conv5의 block의 개수: ", len(self.conv5))
        print("#5 conv5의 'activation map' shape: ", x.shape)
        
        # average pooling
        x = self.aver_pool.average_pooling_layer(x)
        print("#6 average pooling의 'activation map' shape: ", x.shape)
        
        # Flatten Layer
        x = self.flatten.flatten_layer(x)
        print("Flatten의 result shape: ", x.shape)

        # Fully Connected Layer + Softmax
        x = self.fc.fully_connected_layer(x)
        x = self.fc.Softmax(x)
        print("Fully Connected Layer의 result shape: ", x.shape)

        return x

def main(): # 
    # image preprocessing - crop from 
    img1 = Image.open('/data2/jeongsj/ComputerVision/BADUK.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(224), # 가로/세로 비율을 유지하면서, 작은 변의 길이로 크기를 줄임. 
        transforms.CenterCrop(224), # 이미지의 중앙 부분을 입력 크기로 잘라냄.
        transforms.ToTensor(), # (type) NumPy, PIL -> Tensor
    ])
    print(img1)
    
    # input
    img1_t = preprocess(img1)
    print(img1_t.shape)

    img2 = Image.open('/data2/jeongsj/ComputerVision/CATS.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(224), # 가로/세로 비율을 유지하면서, 작은 변의 길이로 크기를 줄임. 
        transforms.CenterCrop(224), # 이미지의 중앙 부분을 입력 크기로 잘라냄.
        transforms.ToTensor(), # (type) NumPy, PIL -> Tensor
    ])
    print(img2)
    
    # input
    img2_t = preprocess(img2)
    print(img2_t.shape)

    # 3차원 이미지 각 두 장 -> 4차원 이미지들 텐서 더미
    img_batch = torch.stack([img1_t, img2_t]).to(device)
    print(img_batch.shape)


    
    #resnet18 = ResNet("basic", [2, 2, 2, 2]).to(device)
    #resnet18_output = resnet18.forward(img_batch)
    #print("ResNet18 최종 output shape: ")
    #print(resnet18_output.shape) 
    
    #resnet34 = ResNet("basic", [3, 4, 6, 3]).to(device)
    #resnet34_output = resnet34.forward(img_batch)
    ##print("ResNet34 최종 output shape: ")
    #print(resnet34_output.shape) 
    
    resnet50 = ResNet("bottleneck", [3, 4, 6, 3]).to(device)
    resnet50_output = resnet50.forward(img_batch)
    print("ResNet50 최종 output shape: ")
    print(resnet50_output.shape)
    
    #resnet101 = ResNet("bottleneck", [3, 4, 23, 3]).to(device)
    #resnet101_output = resnet101.forward(img_batch)
    #print("ResNet101 최종 output shape: ")
    #print(resnet101_output.shape)  
    
    #resnet152 = ResNet("bottleneck", [3, 8, 36, 3]).to(device)
    #resnet152_output = resnet152.forward(img_batch)
    #print("ResNet152 최종 output shape: ")
    #print(resnet152_output.shape) 
    
if __name__ == '__main__':
    main()