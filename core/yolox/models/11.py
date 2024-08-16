class Temporal_Active_Focus_3D(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="gelu"):
        super().__init__()
        time_channels = int(in_channels/2)
        reduce_times = int(log2(time_channels))

        embed_dim = 32

        self.embed_dim = embed_dim
        self.time_channels = time_channels
        self.convs = nn.ModuleList()
        self.convs.append(BaseConv(in_channels, int(time_channels / 2 * embed_dim), 3, 2, int(time_channels/2), True, act))
        for i in range(1, reduce_times):
            self.convs.append(BaseConv(int(time_channels / (2 ** i) * embed_dim), int(time_channels / (2 ** (i + 1)) * embed_dim), 3, 1, int(time_channels/(2 ** (i + 1))), True, act))


        self.conv2 = BaseConv(reduce_times * embed_dim, out_channels, ksize=1, stride=1, act = act, dropout=0.25)

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        for i in range(len(self.convs)):
            self.convs[i].weight.data.normal_(0, 0.01)

    def forward(self, x):

        xout = []
        for i in range(len(self.convs)):
            x = self.convs[i](x)

            xout.append(x[:,:self.embed_dim])
        x = self.conv2(torch.cat(xout, dim = 1))
        return x

    class Darknet(nn.Module):
        # number of blocks from dark2 to dark5.
        depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

        def __init__(
                self,
                depth,
                shape,
                stem=Focus,
                in_channels=3,
                stem_out_channels=64,
                out_channels=[256, 512, 1024],
                out_features=("dark3", "dark4", "dark5"),
                act="silu",
        ):
            """
            Args:
                depth (int): depth of darknet used in model, usually use [21, 53] for this param.
                in_channels (int): number of input channels, for example, use 3 for RGB image.
                stem_out_channels (int): number of output chanels of darknet stem.
                    It decides channels of darknet layer2 to layer5.
                out_features (Tuple[str]): desired output layer name.
            """
            super().__init__()
            assert out_features, "please provide output features of Darknet"
            self.out_features = out_features
            '''self.stem = nn.Sequential(
                BaseConv(in_channels, stem_out_channels, ksize=3, stride=1,act=act),
                *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,act=act),
            )'''

            self.stem = stem(in_channels, stem_out_channels, ksize=3, act=act)
            base_channels = stem_out_channels

            num_blocks = Darknet.depth2blocks[depth]
            # create darknet with `stem_out_channels` and `num_blocks` layers.
            # to make model structure more clear, we don't use `for` statement in python.

            # self.dark1 = nn.Sequential(
            #     *self.make_group_layer(base_channels, base_channels * 2, num_blocks[0], stride=2,act=act)
            # )

            self.dark2 = nn.Sequential(
                *self.make_group_layer(base_channels * 1, base_channels * 2, num_blocks[0], stride=2, act=act)
            )

            self.dark3 = nn.Sequential(
                *self.make_group_layer(base_channels * 2, out_channels[0], num_blocks[1], stride=2, act=act)
            )

            self.dark4 = nn.Sequential(
                *self.make_group_layer(out_channels[0], out_channels[1], num_blocks[2], stride=2, act=act)
            )

            self.dark5 = nn.Sequential(
                *self.make_group_layer(out_channels[1], out_channels[2], num_blocks[3], stride=2, act=act),
                *self.make_spp_block([out_channels[2], out_channels[2]], base_channels * 4, act=act),
            )

            self.shape = shape

        def make_group_layer(self, in_channels, out_channels, num_blocks, stride, act="silu"):
            "starts with conv layer then has `num_blocks` `ResLayer`"
            return [
                BaseConv(in_channels, out_channels, ksize=3, stride=stride, act=act),
                *[(ResLayer(out_channels, act=act)) for _ in range(num_blocks)],
            ]

        def make_spp_block(self, filters_list, in_filters, act="silu"):
            m = nn.Sequential(
                *[
                    BaseConv(in_filters, filters_list[0], 1, stride=1, act=act),
                    BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=act),
                    SPPBottleneck(
                        in_channels=filters_list[1],
                        out_channels=filters_list[0],
                        activation=act,
                    ),
                    BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=act),
                    BaseConv(filters_list[1], filters_list[0], 1, stride=1, act=act),
                ]
            )
            return m

        def forward(self, input):
            outputs = {}
            x = self.stem(input)  # 64, 128, 160, 0.9490G (+0.0679G)
            outputs["stem"] = x
            # x = self.dark1(x)
            # outputs["dark1"] = x
            x = self.dark2(x)  # 128, 64, 80, 1.5584G
            outputs["dark2"] = x
            x = self.dark3(x)  # 256, 32, 40, 2.1853G
            outputs["dark3"] = x
            x = self.dark4(x)  # 256, 16, 20, 0.6779G
            outputs["dark4"] = x
            x = self.dark5(x)  # 256, 8, 10, 0.1281G 0.1114G
            outputs["dark5"] = x
            # 4.7300G
            # 5.2454G 1.0915G 0.1807G
            # 16.8602G
            return [outputs[k] for k in self.out_features]
            # return {k: v for k, v in outputs.items() if k in self.out_features}