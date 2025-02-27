import torch
import torch.nn as nn

from src.models.layers import ResBlock, AttentionBlock3D

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, filters, kernel_size):
        super().__init__()


        # Encoder block
        self.encoder = nn.ModuleList()

        for f,k in zip(filters, kernel_size):

            self.encoder.append(
                ResBlock(in_channels, f, k)
            )

            in_channels = f

        self.pool = nn.MaxPool3d(kernel_size=(2,2,1), stride = (2,2,1))

        self.bottom = ResBlock(filters[-1], filters[-1], kernel_size=kernel_size[-1])


        # Decoder block

        self.decoder = nn.ModuleList()
        self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear')

        decoder_filters = filters[::-1][1:]

        prev_channels = filters[-1]

        reversed_kernels = kernel_size[::-1][1:]


        for f, k in zip(decoder_filters, reversed_kernels):

            self.decoder.append(
                ResBlock(prev_channels + f, f, k)
            )

            prev_channels = f

        self.final_resblock = ResBlock(prev_channels, filters[0], kernel_size[0])

        self.output_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)


    def forward(self, x):

        skips = []

        # Encoder pass
        for resblock in self.encoder:
            x = resblock(x)
            skips.append(x)
            x = self.pool(x)

        # Bottom pass
        x = self.bottom(x)

        # Decoder pass
        for resblock in self.decoder:
            skip = skips.pop()
            x = self.up(x)
            x = torch.cat((x, skip), dim = 1)
            x = resblock(x)

        # Final pass
        x = self.final_resblock(x)
        x = self.output_conv(x)

        return x



class AttentionUNet(nn.Module):

    def __init__(self, in_channels, out_channels, filters, att_channels, kernel_size, return_att_map = False):
        super().__init__()

        self.return_att_map = return_att_map

        # Encoder block
        self.encoder = nn.ModuleList()
        for f,k in zip(filters, kernel_size):

            self.encoder.append(
                ResBlock(in_channels, f, k)
            )
            in_channels = f

        self.pool = nn.MaxPool3d(kernel_size=(2,2,1), stride = (2,2,1))

        # Bottom block 
        self.bottom = ResBlock(filters[-1], filters[-1], kernel_size[-1])

        # Decoder block with attention
        self.attentions = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear')
        prev_channel = filters[-1]
        
        decoder_filters = filters[::-1][1:]
        decoder_kernels = kernel_size[::-1][1:]

        for f,k in zip(decoder_filters, decoder_kernels):

            self.decoder.append(
                ResBlock(prev_channel + f, f, k)
            )

            self.attentions.append(
                AttentionBlock3D(prev_channel, f, att_channels, kernel_size=3)
            )

            prev_channel = f

        self.final_resblock = ResBlock(prev_channel, filters[0], kernel_size[0])

        self.output_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)


    def forward(self, x):

        attention_maps = []
        skips = []

        # Encoder pass
        for resblock in self.encoder:
            x = resblock(x)
            skips.append(x)
            x = self.pool(x)

        # Bottom pass
        x = self.bottom(x) 

        # Decoder pass
        for resblock, attn_block in zip(self.decoder, self.attentions):
            skip = skips.pop()
            attn_skip, attn_map = attn_block(x, skip)
            attention_maps.append(attn_map)

            x = torch.cat((x, attn_skip), dim = 1)

            x = self.up(x)
            x = resblock(x)


        # Final pass
        x = self.final_resblock(x)
        x = self.output_conv(x)

        if self.return_att_map:
            return x, attention_maps

        return x


