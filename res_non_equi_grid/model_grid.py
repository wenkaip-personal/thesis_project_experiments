import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_add_pool, radius_graph, knn, radius
from functools import partial

class equivariant_layer(nn.Module):
    def __init__(self, hidden_features):
        super(equivariant_layer, self).__init__()
        self.hidden_features = hidden_features
        self.message_net1 = nn.Sequential(
            nn.Linear(2*hidden_features+1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )
        self.message_net2 = nn.Sequential(
            nn.Linear(2*hidden_features+1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )
        self.p = 5

    def forward(self, x, pos, batch):
        edge_index = radius_graph(pos, r=4.5, batch=batch, loop=True)
        dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1, keepdim=True)
        vec1, vec2 = self.message(x[edge_index[0]], x[edge_index[1]], dist, pos[edge_index[0]], pos[edge_index[1]])
        vec1_out, vec2_out = global_add_pool(vec1, edge_index[0]), global_add_pool(vec2, edge_index[0])
        vec1_out = global_mean_pool(vec1_out, batch)
        vec2_out = global_mean_pool(vec2_out, batch)
        return self.gram_schmidt_batch(vec1_out, vec2_out)

    def gram_schmidt_batch(self, v1, v2):
        n1 = v1 / (torch.norm(v1, dim=-1, keepdim=True)+1e-8)
        n2_prime = v2 - (n1 * v2).sum(dim=-1, keepdim=True) * n1
        n2 = n2_prime / (torch.norm(n2_prime, dim=-1, keepdim=True)+1e-8)
        n3 = torch.cross(n1, n2, dim=-1)
        return torch.stack([n1, n2, n3], dim=-2)
    
    def omega(self, dist):
        out = 1 - (self.p+1)*(self.p+2)/2 * (dist/4.5)**self.p + self.p*(self.p+2) * (dist/4.5)**(self.p+1) - self.p*(self.p+1)/2 * (dist/4.5)**(self.p+2)
        return out
    
    def message(self, x_i, x_j, dist, pos_i, pos_j):
        x_ij = torch.cat([x_i, x_j, dist], dim=-1)
        mes_1 = self.message_net1(x_ij)
        mes_2 = self.message_net2(x_ij)
        coe = self.omega(dist)
        norm_vec = (pos_i - pos_j) / (torch.norm(pos_i - pos_j, dim=-1, keepdim=True)+1e-8)
        return norm_vec * coe * mes_1, norm_vec * coe * mes_2

class MPNNLayer(nn.Module):
    """ Message Passing Layer """
    def __init__(self, edge_features=6, hidden_features=128, act=nn.SiLU):
        super().__init__()
        self.edge_model = nn.Sequential(nn.Linear(edge_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, hidden_features))
        
        self.message_model = nn.Sequential(nn.Linear(hidden_features*2, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, hidden_features))

        self.update_net =  nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, hidden_features))
        
    def forward(self, node_embedding, node_pos, grid_pos, edge_index):
        message = self.message(node_embedding, node_pos, grid_pos, edge_index)
        x = self.update(message, edge_index[1])
        return x

    def message(self, node_embedding, node_pos, grid_pos, edge_index):
        index_i, index_j = edge_index[0], edge_index[1]
        pos_nodes, pos_grids = node_pos[index_i], grid_pos[index_j]
        edge_attr = torch.cat((pos_nodes, pos_grids), dim=-1)
        pos_embedding = self.edge_model(edge_attr)
        node_embedding = node_embedding[index_i]
        message = torch.cat((node_embedding, pos_embedding), dim=-1)
        message = self.message_model(message)
        return message

    def update(self, message, index_j):
        """ Update node features """
        num_messages = torch.bincount(index_j)
        message = global_add_pool(message, index_j) / num_messages.unsqueeze(-1)
        update = self.update_net(message)

        return update

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm3d(out_channel)
        )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = nn.ReLU()(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers: list, num_classes: int = 20, in_channels: int = 256):
        super(ResNet3D, self).__init__()

        self.instance_norm1 = nn.BatchNorm3d(in_channels)

        self.in_channels = in_channels

        self.layer1 = self._make_layer(block, in_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, in_channels * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, in_channels * 4, layers[2], stride=1)

        self.softmax = nn.functional.softmax
        self.fc = nn.Linear(in_channels * 4, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.instance_norm1(x)  # (bs, 128, 15, 15, 15)

        x1 = self.layer1(x)  # (bs, 128, 11, 11, 11)
        x2 = self.layer2(x1)  # (bs, 256, 7, 7, 7 )
        x3 = self.layer3(x2) # (bs, 512, 3, 3, 3)
        x_out = F.max_pool3d(x3, kernel_size=x3.shape[-1], stride=3)
        
        # Changed: squeeze only the spatial dimensions (2, 3, 4), not batch dimension (0)
        out = x_out.view(x_out.size(0), -1)  # Reshape to (batch_size, features)
        out = self.fc(out)
        return out

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class PatchEmbed_3d(nn.Module):
    """
    3D Volume to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=1, embed_dim=4096, norm_layer=None):
        # embed_dim = 16*16*16 = 4096 token在flatten之后的长度
        super().__init__()
        img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        # 一共有多少个token(patche)  (224/16)*(224/16)*(224/16) = 14*14*14 = 2744

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, P = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 输入图片的大小必须是固定的
        
        # x 大小224*224*224 经过k=16,s=16,c=4096的卷积核之后大小为 14*14*14*4096
        # flatten: [B, C, H, W, P] -> [B, C, HWP]   [B, 4096, 14, 14, 14] -> [B, 4096, 2744]
        # 对于Transfoemer模块，要求输入的是token序列，即 [num_token,token_dim] = [2744,4096]
        # transpose: [B, C, HWP] -> [B, HWP, C]   [B, 4096, 2744] -> [B, 2744, 4096]
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] 调整顺序
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale # q dot-product k的转置，只对最后两个维度进行操作
        attn = attn.softmax(dim=-1) # 对每一行进行softmax
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim] 将多头的结果拼接在一起
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4., # 第一个全连接层节点个数是输入的四倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(ViTBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class VisionTransformer_stage(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=None, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_stage, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 # num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) # 默认参数
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # token/patch的个数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # parameter构建可训练参数，第一个1是batch size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # 位置编码的大小和加入分类token之后的大小相同
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 构建等差序列，dropout率是递增的
#         self.blocks = nn.Sequential(*[
#             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
#                   norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(depth)
#         ])
        self.stage1 = nn.Sequential(ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[0],norm_layer=norm_layer, act_layer=act_layer),
                                    ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[1],norm_layer=norm_layer, act_layer=act_layer),
                                    ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[2],norm_layer=norm_layer, act_layer=act_layer))
        
        
        self.stage2 = nn.Sequential(ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[3],norm_layer=norm_layer, act_layer=act_layer),
                                    ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[4],norm_layer=norm_layer, act_layer=act_layer),
                                    ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[5],norm_layer=norm_layer, act_layer=act_layer))
        
        self.stage3 = nn.Sequential(ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[6],norm_layer=norm_layer, act_layer=act_layer),
                                    ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[7],norm_layer=norm_layer, act_layer=act_layer),
                                    ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[8],norm_layer=norm_layer, act_layer=act_layer))
        
        self.stage4 = nn.Sequential(ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[9],norm_layer=norm_layer, act_layer=act_layer),
                                    ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[10],norm_layer=norm_layer, act_layer=act_layer),
                                    ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, 
                                          attn_drop_ratio=attn_drop_ratio, 
                                          drop_path_ratio=dpr[11],norm_layer=norm_layer, act_layer=act_layer))
        
        

        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

        

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # 把cls_token复制batch_size份
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]        

        x = self.pos_drop(x + self.pos_embed) 
        #x = self.blocks(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.norm(x)
        
        x = self.pre_logits(x[:, 0])
        x = self.head(x) # 执行这里
        
        return x

class ProteinGrid(nn.Module):
    def __init__(self, patch_size=2, grid_size=10, node_types=4, res_types=21, on_bb=2, hidden_features=128, out_features=20, act=nn.SiLU):
        super().__init__()
        self.atom_embedding = nn.Embedding(node_types, node_types)
        self.res_embedding = nn.Embedding(res_types, res_types)
        self.on_bb_embedding = nn.Embedding(on_bb, on_bb)
        self.feature_embedding = nn.Sequential(
            nn.Linear(node_types + res_types + on_bb, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features),
        )
        
        self.equi_layer = equivariant_layer(hidden_features)
        self.mpnn_layer = MPNNLayer(hidden_features=hidden_features, act=act)

        self.cnn_model = ResNet3D(
            block=Block, layers=[1, 1, 1, 1], in_channels=hidden_features, num_classes=out_features
        )
        
        self.transformer_model = VisionTransformer_stage(
                                            img_size=grid_size,
                                            in_c=hidden_features,
                                            patch_size=patch_size,
                                            embed_dim=hidden_features*2,
                                            depth=12,
                                            num_heads=4,
                                            num_classes=out_features,
                                            embed_layer=PatchEmbed_3d)
        
        self.fusion = nn.Linear(out_features*2, out_features)
        self.hidden_features = hidden_features

    def forward(self, batch):
        batch_size = batch.ptr.shape[0] - 1
        node_pos = batch.coords.to(torch.float32)
        grid_pos = batch.grid_coords.to(torch.float32)
        physical_feats = torch.stack([batch.sasa, batch.charges], dim=-1).to(torch.float32)
        physical_feats[torch.isinf(physical_feats)] = 0
        physical_feats[torch.isnan(physical_feats)] = 0
        atom_types = batch.atom_types.to(torch.long)   
        atom_on_bb = batch.atom_on_bb.to(torch.long)
        res_types = batch.res_types.to(torch.long)
        atom_embedding = self.atom_embedding(atom_types)
        res_embedding = self.res_embedding(res_types)
        on_bb_embedding = self.on_bb_embedding(atom_on_bb)
        atom_feature = self.feature_embedding(
            torch.cat((atom_embedding, res_embedding, on_bb_embedding), dim=-1)
        )
        
        atom_batch = batch.batch[batch.is_atom_mask.bool() == True]  # Get batch assignments for atoms only
        # frame = self.equi_layer(atom_feature, node_pos, atom_batch)
        # node_pos = torch.bmm(node_pos.unsqueeze(1), frame[atom_batch].permute(0, 2, 1)).squeeze()
        # Get the number of grid points per sample
        grid_points_per_sample = batch.grid_size[0]**3
        
        # Create batch indices for grid points
        grid_batch_idx = torch.arange(batch_size, device=grid_pos.device).repeat_interleave(grid_points_per_sample)
        
        # Since grid_pos contains concatenated grid coordinates from all samples,
        # we need to select the appropriate grid coordinates for transformation
        # The first grid_points_per_sample points are the actual grid coordinates
        row_1, col_1 = knn(node_pos, grid_pos, k=3, batch_x=atom_batch, batch_y=grid_batch_idx)
        row_2, col_2 = knn(grid_pos, node_pos, k=3, batch_x=grid_batch_idx, batch_y=atom_batch)

        edge_index_knn = torch.stack(
            (torch.cat((col_1, row_2)),
            torch.cat((row_1, col_2)))
        )

        row_1, col_1 = radius(node_pos, grid_pos, r=4, batch_x=atom_batch, batch_y=grid_batch_idx)
        row_2, col_2 = radius(grid_pos, node_pos, r=4, batch_x=grid_batch_idx, batch_y=atom_batch)

        edge_index_radius = torch.stack(
            (torch.cat((col_1, row_2)),
            torch.cat((row_1, col_2)))
        )
        edge_index = torch.cat((edge_index_knn, edge_index_radius), dim=-1)

        edge_index = torch_geometric.utils.coalesce(edge_index)

        cnn_input = self.mpnn_layer(atom_feature, node_pos, grid_pos, edge_index)
        cnn_input = cnn_input.reshape(
            batch_size, 
            int(batch.grid_size[0]), int(batch.grid_size[0]), int(batch.grid_size[0]), 
            self.hidden_features
        ).permute(0, 4, 1, 2, 3)
        preds_1 = self.cnn_model(cnn_input)
        # preds_2 = self.transformer_model(cnn_input)
        # preds = torch.cat((preds_1, preds_2), dim=-1)
        # preds = F.log_softmax(self.fusion(preds), dim=-1)
        preds = F.log_softmax(preds_1, dim=-1)
        loss = F.cross_entropy(preds, batch.y, reduction='none')
        pred_labels = torch.max(preds, dim=-1)[1]
        acc = (pred_labels == batch.y).float()
        backprop_loss = loss.mean()  # []

        correct_counts = torch.zeros(20).to(batch.y.device)
        correct_counts.scatter_add_(0, batch.y, (pred_labels == batch.y).float())
        total_counts = torch.zeros(20).to(batch.y.device)
        total_counts.scatter_add_(0, batch.y, torch.ones_like(batch.y).float())
        acc_per_class = correct_counts / total_counts
        log_dict = dict()
        log_dict["loss"] = loss
        log_dict["acc"] = acc
        for i, acc_cls in enumerate(acc_per_class):
            log_dict[f"acc_{i}"] = torch.ones(batch_size).to(batch.y.device)*acc_cls

        return backprop_loss, log_dict