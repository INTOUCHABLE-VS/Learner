###### 做了多个stage

```python
class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 # num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        # self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
                'freeze_bn': spec['FREEZE_BN'],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = nn.Linear(dim_embed, 1000)
        trunc_normal_(self.head.weight, std=0.02)

    def forward(self, template, online_template, search):
        for i in range(self.num_stages):
            template, online_template, search = getattr(self, f'stage{i}')(template, online_template, search)
        return template, search

```

------

### 具体每个MAM

####  **第一步**   给每个模板和搜索加上位置编码

###### 输入 template、 search

```python
self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )        
        template = self.patch_embed(template)
        online_template = self.patch_embed(online_template)
        t_B, t_C, t_H, t_W = template.size()
        search = self.patch_embed(search)
        s_B, s_C, s_H, s_W = search.size()
```

###### 具体的位置编码

```python
class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """
 
    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
 
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None
 
    def forward(self, x):
        x = self.proj(x)
 
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
 
        return x
```

###### concat 位置编码后的tokens

```python
        template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
        online_template = rearrange(online_template, 'b c h w -> b (h w) c').contiguous()
        search = rearrange(search, 'b c h w -> b (h w) c').contiguous()
        x = torch.cat([template, online_template, search], dim=1)
 
        x = self.pos_drop(x)
```

#### **第二步**   计算MAM 注意力

###### 整体是一个残差结构

```python
def forward(self, x, t_h, t_w, s_h, s_w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, t_h, t_w, s_h, s_w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
```

###### mul head attention function（用于生成Q、K、V）

**先看Q K V 怎么来的**

```python
def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method,
                          norm):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', norm(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))

def forward_conv(self, x, t_h, t_w, s_h, s_w):
        template, online_template, search = torch.split(x, [t_h * t_w, t_h * t_w, s_h * s_w], dim=1)
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_h, w=t_w).contiguous()
        online_template = rearrange(online_template, 'b (h w) c -> b c h w', h=t_h, w=t_w).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_h, w=s_w).contiguous()

        if self.conv_proj_q is not None:
            t_q = self.conv_proj_q(template)
            ot_q = self.conv_proj_q(online_template)
            s_q = self.conv_proj_q(search)
            q = torch.cat([t_q, ot_q, s_q], dim=1)
        else:
            t_q = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            ot_q = rearrange(online_template, 'b c h w -> b (h w) c').contiguous()
            s_q = rearrange(search, 'b c h w -> b (h w) c').contiguous()
            q = torch.cat([t_q, ot_q, s_q], dim=1)

```

```python
"""
        Asymmetric mixed attention.
        """
if (
    self.conv_proj_q is not None
    or self.conv_proj_k is not None
    or self.conv_proj_v is not None
):
    q, k, v = self.forward_conv(x, t_h, t_w, s_h, s_w)
    
	self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
    self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
    self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
    
    q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()
    k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()
    v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()

```

###### 利用Q K V计算attention

```python
### Attention!: k/v compression，1/4 of q_size（conv_stride=2）
q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
# k_t, k_ot, k_s = torch.split(k, [t_h*t_w//4, t_h*t_w//4, s_h*s_w//4], dim=2)
# v_t, v_ot, v_s = torch.split(v, [t_h * t_w // 4, t_h * t_w // 4, s_h * s_w // 4], dim=2)
k_mt, k_s = torch.split(k, [((t_h + 1) // 2) ** 2 * 2, s_h * s_w // 4], dim=2)
v_mt, v_s = torch.split(v, [((t_h + 1) // 2) ** 2 * 2, s_h * s_w // 4], dim=2)


self.scale = dim_out ** -0.5
# template attention
attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_mt, k_mt]) * self.scale
attn = F.softmax(attn_score, dim=-1)
attn = self.attn_drop(attn)
x_mt = torch.einsum('bhlt,bhtv->bhlv', [attn, v_mt])
x_mt = rearrange(x_mt, 'b h t d -> b t (h d)')

# search region attention
attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_s, k]) * self.scale
attn = F.softmax(attn_score, dim=-1)
attn = self.attn_drop(attn)
x_s = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
x_s = rearrange(x_s, 'b h t d -> b t (h d)')

x = torch.cat([x_mt, x_s], dim=1)

self.proj = nn.linear(dim_in, dim_out)
self.proj_drop = nn.Dropout(proj_drop)

x = self.proj(x)
x = self.proj_drop(x)

return x
```

#### **第三步**   Corner Head

```python
def soft_argmax(self, score_map, return_dist=False, softmax=True):
    """ get soft-argmax coordinate for a given heatmap """
    score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
    prob_vec = nn.functional.softmax(score_vec, dim=1)
    exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
    exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
    if return_dist:
        if softmax:
            return exp_x, exp_y, prob_vec
        else:
            return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y
        
def get_score_map(self, x):
    # top-left branch
    x_tl1 = self.conv1_tl(x)
    x_tl2 = self.conv2_tl(x_tl1)
    x_tl3 = self.conv3_tl(x_tl2)
    x_tl4 = self.conv4_tl(x_tl3)
    score_map_tl = self.conv5_tl(x_tl4)

    # bottom-right branch
    x_br1 = self.conv1_br(x)
    x_br2 = self.conv2_br(x_br1)
    x_br3 = self.conv3_br(x_br2)
    x_br4 = self.conv4_br(x_br3)
    score_map_br = self.conv5_br(x_br4)
    return score_map_tl, score_map_br

def forward(self, x, return_dist=False, softmax=True):
    """ Forward pass with input x. """
    score_map_tl, score_map_br = self.get_score_map(x)
    if return_dist:
        coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
        coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
    else:
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz
```

