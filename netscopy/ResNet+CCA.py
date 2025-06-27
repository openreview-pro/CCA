'''
ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=1, k=9):
        super(SparseAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.k = k  # 每个节点保留前 K 个注意力最大的邻居
        self.head_dim = dim // num_heads

        # 定义 Q, K, V 的线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.scale = (self.head_dim) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

        # 输出线性层
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: 输入形状为 (batch_size, dim, H, W)
        """
        batch_size, num_nodes, H, W = x.shape

        # 将x变换维度以方便在dim*3,同时拉平H*W
        x = x.view(batch_size, num_nodes, H * W).permute(0, 2, 1)

        # 计算 Q, K, V
        qkv = self.qkv(x)  # 形状 (batch_size, H*W, 3 * dim)
        # qkv = qkv.view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(batch_size, H * W, 3, self.dim)
        # qkv = qkv.permute(2, 0, 3, 1, 4)  # 形状 (3, batch_size, num_heads, num_nodes, head_dim)
        qkv = qkv.permute(2, 0, 3, 1)  # 形状（3, batch_size, self.dim, H*W）
        q, k, v = qkv[0], qkv[1], qkv[2]  # 获取 Q, K, V

        # 计算注意力分数
        attn_scores = torch.matmul(q,
                                   k.transpose(-2, -1)) * self.scale  # 形状 (batch_size, num_heads, num_nodes, num_nodes)

        # 为每个节点挑出前 K 个注意力系数
        top_k_values, top_k_indices = torch.topk(attn_scores, self.k, dim=-1)  # 获取每个节点前 K 个最大的注意力分数及其索引
        mask = torch.zeros_like(attn_scores)  # 创建一个与 attn_scores 形状相同的全 0 掩码
        mask.scatter_(-1, top_k_indices, 1.0)  # 在掩码的前 K 个注意力位置赋值为 1

        # 添加自身节点
        identity_indices = torch.arange(attn_scores.size(-1), device=attn_scores.device)
        mask[:, identity_indices, identity_indices] = 1.0  # 调整索引维度

        # 稀疏化注意力分数，将不属于前 K 和本身的位置置为 0
        sparse_attn_scores = attn_scores * mask

        # 将掩码中为零的位置的分数置为负无穷大
        sparse_attn_scores = sparse_attn_scores.masked_fill(mask == 0, float('-inf'))

        # 重新归一化稀疏化后的注意力分数
        sparse_attn_scores = self.softmax(sparse_attn_scores)

        # 聚合邻居节点信息
        out = torch.matmul(sparse_attn_scores, v)  # 形状 (batch_size, num_heads, num_nodes, head_dim)

        # 重新调整形状并通过线性层输出
        # out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, self.dim)
        out = out.view(batch_size, self.dim, H, W).permute(0, 2, 3, 1)
        out = self.proj(out)
        out = out.permute(0, 3, 1, 2)

        return out


class BasicBlock(nn.Module):
    """
    对于浅层网络，如ResNet-18/34等，用基本的Block
    基础模块没有压缩,所以expansion=1
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.sparseattention = SparseAttention(dim=out_channels, num_heads=1, k=9)
        # 如果输入输出维度不等，则使用1x1卷积层来改变维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = self.features(x)
        #         print(out.shape)
        out = self.sparseattention(out)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    对于深层网络，我们使用BottleNeck，论文中提出其拥有近似的计算复杂度，但能节省很多资源
    zip_channels: 压缩后的维数，最后输出的维数是 expansion * zip_channels
    针对ResNet50/101/152的网络结构,主要是因为第三层是第二层的4倍的关系所以expansion=4
    """
    expansion = 4

    def __init__(self, in_channels, zip_channels, stride=1):
        super(Bottleneck, self).__init__()
        out_channels = self.expansion * zip_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip_channels, zip_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.features(x)
        #         print(out.shape)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """
    不同的ResNet架构都是统一的一层特征提取、四层残差，不同点在于每层残差的深度。
    对于cifar10，feature map size的变化如下：
    (32, 32, 3) -> [Conv2d] -> (32, 32, 64) -> [Res1] -> (32, 32, 64) -> [Res2]
 -> (16, 16, 128) -> [Res3] -> (8, 8, 256) ->[Res4] -> (4, 4, 512) -> [AvgPool]
 -> (1, 1, 512) -> [Reshape] -> (512) -> [Linear] -> (10)
    """

    def __init__(self, block, num_blocks, num_classes=10, verbose=False, init_weights=True):
        super(ResNet, self).__init__()
        self.verbose = verbose
        self.in_channels = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 使用_make_layer函数生成上表对应的conv2_x, conv3_x, conv4_x, conv5_x的结构
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # cifar10经过上述结构后，到这里的feature map size是 4 x 4 x 512 x expansion
        # 所以这里用了 4 x 4 的平均池化
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifer = nn.Linear(512 * block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个block要进行降采样
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # 如果是Bottleneck Block的话需要对每层输入的维度进行压缩，压缩后再增加维数
            # 所以每层的输入维数也要跟着变
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        if self.verbose:
            print('block 1 output: {}'.format(out.shape))
        out = self.layer1(out)
        if self.verbose:
            print('block 2 output: {}'.format(out.shape))
        out = self.layer2(out)
        if self.verbose:
            print('block 3 output: {}'.format(out.shape))
        out = self.layer3(out)
        if self.verbose:
            print('block 4 output: {}'.format(out.shape))
        out = self.layer4(out)
        if self.verbose:
            print('block 5 output: {}'.format(out.shape))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)


def ResNet18(verbose=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], verbose=verbose)


def ResNet34(verbose=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], verbose=verbose)


def ResNet50(verbose=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], verbose=verbose)


def ResNet101(verbose=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], verbose=verbose)


def ResNet152(verbose=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], verbose=verbose)


def test():
    net = ResNet34()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net, (2, 3, 32, 32))

# test()