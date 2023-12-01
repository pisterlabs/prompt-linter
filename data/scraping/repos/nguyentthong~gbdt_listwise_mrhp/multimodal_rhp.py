import torch
import torch.nn.functional as F

from torch import nn
from matchzoo.preprocessors.units import Vocabulary
from matchzoo.pipeline.rhp_pipeline import RHPPipeline

from matchzoo.modules.cnn import ConvEncoder
from matchzoo.modules.embedding_layer import EmbeddingLayer
from matchzoo.modules.transformer import TransformerEncoderLayer
from matchzoo.modules.cross_match import CrossMatchLayer, BertCrossMatchLayer
from matchzoo.modules.cross_modal_match import CrossModalMatchLayer
from matchzoo.modules.coherent import CoherentEncoder
from matchzoo.modules.kernel_max import KernelMaxPooling
from matchzoo.modules.utils import generate_seq_mask, flatten_all
import math


class TextCNNEncoder(nn.Module):
    def __init__(self, config, vocab: Vocabulary, vocab_name: str, stage: str):
        super().__init__()
        self.token_embedding = EmbeddingLayer(
            vocab_map=vocab.v2i,
            embedding_dim=config.embedding.embed_dim,
            vocab_name=vocab_name,
            dropout=config.embedding.dropout,
            embed_type=config.embedding.embed_type,
            padding_index=vocab.pad_index,
            pretrained_dir=config.embedding.pretrained_file,
            stage=stage,
            initial_type=config.embedding.init_type
        )

        self.seq_encoder = ConvEncoder(
            input_size=config.embedding.embed_dim,
            kernel_size=config.encoder.kernel_size,
            kernel_num=config.encoder.hidden_dimension,
            padding_index=vocab.pad_index
        )

    def forward(self, input, input_length):
        input = self.token_embedding(input)
        input, unpadding_mask = self.seq_encoder(input, input_length)
        return input, unpadding_mask


class TextLSTMEncoder(nn.Module):
    def __init__(self, config, vocab, vocab_name, stage):
        super().__init__()
        self.token_embedding = EmbeddingLayer(
            vocab_map=vocab.v2i,
            embedding_dim=config.embedding.embed_dim,
            vocab_name=vocab_name,
            dropout=config.embedding.dropout,
            embed_type=config.embedding.embed_type,
            padding_index=vocab.pad_index,
            pretrained_dir=config.embedding.pretrained_file,
            stage=stage,
            initial_type=config.embedding.init_type
        )

        self.hidden_dim = 128
        self.lstm_text_encoder = nn.LSTM(config.embedding.embed_dim, self.hidden_dim, batch_first=True)


    def forward(self, input):
        input = self.token_embedding(input)
        mask = txt_attn_mask.unsqueeze(-1)
        prd_hidden_vectors, _ = self.lstm_text_encoder(prd_hidden_vectors)
        prd_hidden_vectors = prd_hidden_vectors * mask
        return prd_hidden_vectors


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout

        self.map = nn.Linear(config.input_dim, config.encoder_embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                encoder_embed_dim=config.encoder_embed_dim,
                dropout=config.dropout,
                relu_dropout=config.relu_dropout,
                encoder_attention_heads=config.encoder_attention_heads,
                attention_dropout=config.attention_dropout,
                encoder_ffn_embed_dim=config.encoder_ffn_embed_dim
            ) for _ in range(config.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)

    def forward(self, input, input_length):
        input = self.map(input)
        input = F.dropout(input, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        input = input.transpose(0, 1)

        # Compute padding mask
        unpadding_mask = generate_seq_mask(
            input_length, max_length=input.size(0))
        encoder_padding_mask = unpadding_mask.eq(0)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Encoder layer
        for layer in self.layers:
            input = layer(input, encoder_padding_mask)
        input = self.layer_norm(input)

        # T x B x C -> B x T x C
        unpadding_mask = unpadding_mask.float()
        input = input.transpose(0, 1)
        input = input * unpadding_mask.unsqueeze(-1)
        return input, unpadding_mask


class CoherentReasoning(nn.Module):
    def __init__(self,
                 config,
                 prd_txt_dim,
                 prd_img_dim,
                 rvw_txt_dim,
                 rvw_img_dim,
                 max_seq_len):
        super().__init__()
        self.prd_coherent = CoherentEncoder(
            prd_img_dim, prd_txt_dim, config.hidden_dim, max_seq_len, config.nlayer, 'mean')
        self.rvw_coherent = CoherentEncoder(
            rvw_img_dim, rvw_txt_dim, config.hidden_dim, max_seq_len, config.nlayer, 'att')

    def forward(self,
                rvw_txt,
                rvw_txt_unpadding_mask,
                rvw_img,
                rvw_img_unpadding_mask,
                prd_txt,
                prd_txt_unpadding_mask,
                prd_img,
                prd_img_unpadding_mask):
        prd_repr = self.prd_coherent(
            prd_txt,
            prd_txt_unpadding_mask,
            prd_img,
            prd_img_unpadding_mask
        )
        coherent_match = self.rvw_coherent(
            rvw_txt,
            rvw_txt_unpadding_mask,
            rvw_img,
            rvw_img_unpadding_mask,
            claims=prd_repr
        )
        return coherent_match


class ProductAwareAttention(nn.Module):
    def __init__(self, hidden_dimension):
        super(ProductAwareAttention, self).__init__()
        self.w = nn.Parameter(torch.randn(hidden_dimension, hidden_dimension))
        self.b = nn.Parameter(torch.randn(1, 1, hidden_dimension))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.b)

    def forward(self,
                product_repr,
                product_unpadding_mask,
                review_repr,
                review_unpadding_mask):
        '''
        :param product_repr: torch.FloatTensor (batch, hidden_dimension, product_seq_lens)
        :param product_seq_lens: torch.LongTensor, (batch, max_len)
        :param review_repr: torch.FloatTensor (batch, hidden_dimension, review_seq_lens)
        :param review_seq_lens: torch.LongTensor, (batch, max_len)
        '''

        # (batch, product_seq_lens, hidden_dimension)
        p = torch.matmul(product_repr.transpose(1, 2), self.w)
        p = p + self.b
        p = torch.relu(p)  # (batch, product_seq_lens, hidden_dimension)
        # (batch, product_seq_lens, review_seq_lens)
        q = torch.matmul(p, review_repr)

        # (batch, product_seq_lens)
        p_mask = product_unpadding_mask
        p_mask = p_mask.unsqueeze(-1)  # (batch, product_seq_lens, 1)
        try:
            q = q * p_mask.float() + (~p_mask).float() * (-1e23)
            # q = torch.matmul(q, p_mask.float()) + (~q_mask).float() * (-1e23)
        except:
            import pdb
            pdb.set_trace()
        q = torch.softmax(q, dim=1)

        r_add = torch.matmul(product_repr, q)
        r = r_add + review_repr   # (batch, hidden_dimension, review_seq_lens)

        r = r.transpose(1, 2)  # (batch, review_seq_lens, hidden_dimension)
        r_mask = review_unpadding_mask  # (batch, review_seq_lens)
        r_mask = r_mask.unsqueeze(-1)
        r = r * r_mask.float()  # (batch, review_seq_lens, hidden_dimension)
        return r


class MultimodalRHPNet(nn.Module):
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self.stage = stage
        self.use_image = config.input_setting.use_img

        # build encoder
        self.prd_txt_encoder, self.rvw_txt_encoder = self.build_text_encoder()
        if self.use_image:
            self.prd_img_encoder, self.rvw_img_encoder = self.build_image_encoder()

        # build cross matching
        self.cross_match = CrossMatchLayer(
            do_normalize=config.cross_match.do_normalize)

        # build cross modal matching
        if self.use_image:
            self.img2txt_match, self.txt2img_match = self.build_cross_modal_match()

        # build coherent
        if self.use_image:
            self.coherentor = self.build_coherentor()

        # build kernel pooling
        poolers = self.build_multisource_pooler()
        self.txt_pooler = poolers[0]
        if self.use_image:
            self.img_pooler, self.img2txt_pooler, self.txt2img_pooler = poolers[1:]

        # build score linear
        features_num = self.cal_features_nums()
        self.linear = nn.Sequential(
            nn.Linear(features_num, 128),
            nn.ReLU(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 1))

    def build_text_encoder(self):
        prd_vocab = self.pipeline.prd_text_field.vocab
        rvw_vocab = self.pipeline.rvw_text_field.vocab
        prd_txt_encoder = TextCNNEncoder(
            self.config.prd_txt_encoder, prd_vocab, 'prd_vocab', self.stage)
        rvw_txt_encoder = TextCNNEncoder(
            self.config.rvw_txt_encoder, rvw_vocab, 'rvw_vocab', self.stage)
        return prd_txt_encoder, rvw_txt_encoder

    def build_image_encoder(self):
        prd_img_encoder = ImageEncoder(self.config.prd_img_encoder)
        rvw_img_encoder = ImageEncoder(self.config.rvw_img_encoder)
        return prd_img_encoder, rvw_img_encoder

    def build_cross_modal_match(self):
        prd_txt_channel = len(self.config.prd_txt_encoder.encoder.kernel_size)
        rvw_txt_channel = len(self.config.rvw_txt_encoder.encoder.kernel_size)

        img2txt_match = CrossModalMatchLayer(
            left_dim=self.config.prd_img_encoder.encoder_embed_dim,
            right_dim=self.config.rvw_txt_encoder.encoder.hidden_dimension * rvw_txt_channel,
            hidden_dim=self.config.cross_modal_match.hidden_dim,
            do_normalize=self.config.cross_modal_match.do_normalize
        )
        txt2img_match = CrossModalMatchLayer(
            left_dim=self.config.prd_txt_encoder.encoder.hidden_dimension * prd_txt_channel,
            right_dim=self.config.rvw_img_encoder.encoder_embed_dim,
            hidden_dim=self.config.cross_modal_match.hidden_dim,
            do_normalize=self.config.cross_modal_match.do_normalize
        )
        return img2txt_match, txt2img_match

    def build_multisource_pooler(self):
        prd_txt_channel = len(self.config.prd_txt_encoder.encoder.kernel_size)
        rvw_txt_channel = len(self.config.rvw_txt_encoder.encoder.kernel_size)

        txt_pooler = KernelMaxPooling(
            num_conv_layers=self.config.pooling.txt_convs_num,
            input_channels=prd_txt_channel * rvw_txt_channel,
            filters_count=self.config.pooling.txt_filters_num,
            ns=self.config.pooling.txt_ns
        )

        outputs = (txt_pooler,)
        if self.use_image:
            img_pooler = KernelMaxPooling(
                num_conv_layers=self.config.pooling.img_convs_num,
                input_channels=1,
                filters_count=self.config.pooling.img_filters_num,
                ns=self.config.pooling.img_ns
            )
            img2txt_pooler = KernelMaxPooling(
                num_conv_layers=self.config.pooling.img2txt_convs_num,
                input_channels=1,
                filters_count=self.config.pooling.img2txt_filters_num,
                ns=self.config.pooling.img2txt_ns
            )
            txt2img_pooler = KernelMaxPooling(
                num_conv_layers=self.config.pooling.txt2img_convs_num,
                input_channels=1,
                filters_count=self.config.pooling.txt2img_filters_num,
                ns=self.config.pooling.txt2img_ns
            )
            outputs += (img_pooler, img2txt_pooler, txt2img_pooler)
        return outputs

    def build_coherentor(self):
        prd_txt_channel = len(self.config.prd_txt_encoder.encoder.kernel_size)
        rvw_txt_channel = len(self.config.rvw_txt_encoder.encoder.kernel_size)

        coherentor = CoherentReasoning(
            self.config.coherent_encoder,
            prd_txt_dim=self.config.prd_txt_encoder.encoder.hidden_dimension * prd_txt_channel,
            prd_img_dim=self.config.prd_img_encoder.encoder_embed_dim,
            rvw_txt_dim=self.config.rvw_txt_encoder.encoder.hidden_dimension * rvw_txt_channel,
            rvw_img_dim=self.config.rvw_img_encoder.encoder_embed_dim,
            max_seq_len=self.config.input_setting.txt_max_length + self.config.input_setting.img_max_length)
        return coherentor

    def cal_features_nums(self):
        pool_config = self.config.pooling
        features_size = pool_config.txt_convs_num * pool_config.txt_filters_num \
            * pool_config.txt_ns

        if self.use_image:
            features_size += (
                pool_config.img_convs_num *
                pool_config.img_filters_num *
                pool_config.img_ns
            )
            features_size += (
                pool_config.img2txt_convs_num *
                pool_config.img2txt_filters_num *
                pool_config.img2txt_ns
            )
            features_size += (
                pool_config.txt2img_convs_num *
                pool_config.txt2img_filters_num *
                pool_config.txt2img_ns
            )
            features_size += self.config.coherent_encoder.hidden_dim

        return features_size

    def forward(self, batch, wo_score=False):
        # encode part data
        prd_txt_repr, prd_txt_unpadding_mask = self.prd_txt_encoder(
            batch['text_left'], batch['text_left_length'])
        rvw_txt_repr, rvw_txt_unpadding_mask = self.rvw_txt_encoder(
            batch['text_right'], batch['text_right_length'])

        # cross match
        txt_cross_match = self.cross_match(
            prd_txt_repr, prd_txt_unpadding_mask,
            rvw_txt_repr, rvw_txt_unpadding_mask
        )

        if self.use_image:
            # img encode
            prd_img_repr, prd_img_unpadding_mask = self.prd_img_encoder(
                batch['image_left'].float(), batch['image_left_length'])
            rvw_img_repr, rvw_img_unpadding_mask = self.prd_img_encoder(
                batch['image_right'].float(), batch['image_right_length'])

            # image cross match
            img_cross_match = self.cross_match(
                [prd_img_repr], prd_img_unpadding_mask,
                [rvw_img_repr], rvw_img_unpadding_mask
            )

            # pooling text
            rvw_txt_repr = torch.cat(rvw_txt_repr, dim=-1)
            prd_txt_repr = torch.cat(prd_txt_repr, dim=-1)

            # cross modal match
            img2txt_match = self.img2txt_match(
                prd_img_repr, prd_img_unpadding_mask,
                rvw_txt_repr, rvw_txt_unpadding_mask
            )
            txt2img_match = self.txt2img_match(
                prd_txt_repr, prd_txt_unpadding_mask,
                rvw_img_repr, rvw_img_unpadding_mask
            )

            # coherent reasoning
            coherent_cross_match = self.coherentor(
                rvw_txt_repr, rvw_txt_unpadding_mask,
                rvw_img_repr, rvw_img_unpadding_mask,
                prd_txt_repr, prd_txt_unpadding_mask,
                prd_img_repr, prd_img_unpadding_mask
            )

        # pooling
        pool_result = []
        pool_result.append(self.txt_pooler(txt_cross_match))
        if self.use_image:
            pool_result.append(self.img_pooler(img_cross_match))
            pool_result.append(self.img2txt_pooler(img2txt_match.unsqueeze(1)))
            pool_result.append(self.txt2img_pooler(txt2img_match.unsqueeze(1)))
            pool_result.append(coherent_cross_match)

        # get score
        input = torch.cat(flatten_all(pool_result), dim=-1)
        score = self.linear(input)
        return score


class MultimodalLayernormRHPNet(MultimodalRHPNet):
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__(config, pipeline, stage)

        # build layer norm
        layernorms = self.build_multisource_layernorm()
        self.txt_layernorm = layernorms[0]
        if self.use_image:
            self.img_layernorm, self.img2txt_layernorm, self.txt2img_layernorm, self.coherent_layernorm = layernorms[
                1:]

    def build_multisource_layernorm(self):
        txt_layernorm = nn.LayerNorm([
            self.config.pooling.txt_convs_num,
            self.config.pooling.txt_ns,
            self.config.pooling.txt_filters_num
        ])

        outputs = (txt_layernorm,)
        if self.use_image:
            img_layernorm = nn.LayerNorm([
                self.config.pooling.img_convs_num,
                self.config.pooling.img_ns,
                self.config.pooling.img_filters_num
            ])
            img2txt_layernorm = nn.LayerNorm([
                self.config.pooling.img2txt_convs_num,
                self.config.pooling.img2txt_ns,
                self.config.pooling.img2txt_filters_num
            ])
            txt2img_layernorm = nn.LayerNorm([
                self.config.pooling.txt2img_convs_num,
                self.config.pooling.txt2img_ns,
                self.config.pooling.txt2img_filters_num
            ])
            coherent_layernorm = nn.LayerNorm([
                self.config.coherent_encoder.hidden_dim
            ])
            outputs += (img_layernorm, img2txt_layernorm,
                        txt2img_layernorm, coherent_layernorm)
        return outputs

    def forward(self, batch, wo_score=False):
        # encode part data
        prd_txt_repr, prd_txt_unpadding_mask = self.prd_txt_encoder(
            batch['text_left'], batch['text_left_length'])
        rvw_txt_repr, rvw_txt_unpadding_mask = self.rvw_txt_encoder(
            batch['text_right'], batch['text_right_length'])

        # cross match
        txt_cross_match = self.cross_match(
            prd_txt_repr, prd_txt_unpadding_mask,
            rvw_txt_repr, rvw_txt_unpadding_mask
        )

        if self.use_image:
            # img encode
            prd_img_repr, prd_img_unpadding_mask = self.prd_img_encoder(
                batch['image_left'].float(), batch['image_left_length'])
            rvw_img_repr, rvw_img_unpadding_mask = self.prd_img_encoder(
                batch['image_right'].float(), batch['image_right_length'])

            # image cross match
            img_cross_match = self.cross_match(
                [prd_img_repr], prd_img_unpadding_mask,
                [rvw_img_repr], rvw_img_unpadding_mask
            )

            # pooling text
            rvw_txt_repr = torch.cat(rvw_txt_repr, dim=-1)
            prd_txt_repr = torch.cat(prd_txt_repr, dim=-1)

            # cross modal match
            img2txt_match = self.img2txt_match(
                prd_img_repr, prd_img_unpadding_mask,
                rvw_txt_repr, rvw_txt_unpadding_mask
            )
            txt2img_match = self.txt2img_match(
                prd_txt_repr, prd_txt_unpadding_mask,
                rvw_img_repr, rvw_img_unpadding_mask
            )

            # coherent reasoning
            coherent_cross_match = self.coherentor(
                rvw_txt_repr, rvw_txt_unpadding_mask,
                rvw_img_repr, rvw_img_unpadding_mask,
                prd_txt_repr, prd_txt_unpadding_mask,
                prd_img_repr, prd_img_unpadding_mask
            )

        # pooling
        pool_result = []
        pool_result.append(self.txt_layernorm(
            self.txt_pooler(txt_cross_match)))
        if self.use_image:
            pool_result.append(self.img_layernorm(
                self.img_pooler(img_cross_match)))
            pool_result.append(self.img2txt_layernorm(
                self.img2txt_pooler(img2txt_match.unsqueeze(1))))
            pool_result.append(self.txt2img_layernorm(
                self.txt2img_pooler(txt2img_match.unsqueeze(1))))
            pool_result.append(self.coherent_layernorm(coherent_cross_match))

        # get score
        input = torch.cat(flatten_all(pool_result), dim=-1)
        score = self.linear(input)
        return score


class CrossModalProductAwareAttention(nn.Module):
    def __init__(self,
                 left_dim: int,
                 hidden_dimension):
        super(CrossModalProductAwareAttention, self).__init__()
        self.w = nn.Parameter(torch.randn(hidden_dimension, hidden_dimension))
        self.b = nn.Parameter(torch.randn(1, 1, hidden_dimension))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.b)

        self.left_fc = nn.Sequential(
            nn.Linear(left_dim, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, hidden_dimension))

    def forward(self,
                product_repr,
                product_unpadding_mask,
                review_repr,
                review_unpadding_mask):
        '''
        :param product_repr: torch.FloatTensor (batch, product_seq_lens, hidden_dimension)
        :param product_seq_lens: torch.LongTensor, (batch, max_len)
        :param review_repr: torch.FloatTensor (batch, review_seq_lens, hidden_dimension)
        :param review_seq_lens: torch.LongTensor, (batch, max_len)
        '''
        product_repr = self.left_fc(product_repr).transpose(1, 2)
        review_repr = review_repr.transpose(1, 2)

        # (batch, product_seq_lens, hidden_dimension)
        p = torch.matmul(product_repr.transpose(1, 2), self.w)
        p = p + self.b
        p = torch.relu(p)  # (batch, product_seq_lens, hidden_dimension)
        # (batch, product_seq_lens, review_seq_lens)
        q = torch.matmul(p, review_repr)

        # (batch, product_seq_lens)
        p_mask = product_unpadding_mask
        p_mask = p_mask.unsqueeze(-1)  # (batch, product_seq_lens, 1)
        q = q * p_mask.float() + (~p_mask).float() * (-1e23)
        q = torch.softmax(q, dim=1)

        r_add = torch.matmul(product_repr, q)
        r = r_add + review_repr   # (batch, hidden_dimension, review_seq_lens)

        r = r.transpose(1, 2)  # (batch, review_seq_lens, hidden_dimension)
        r_mask = review_unpadding_mask  # (batch, review_seq_lens)
        r_mask = r_mask.unsqueeze(-1)
        r = r * r_mask.float()  # (batch, review_seq_lens, hidden_dimension)
        return r


class MultimodalLayernormRHPNet3(MultimodalLayernormRHPNet):
    """Replace the img2txt and txt2img matching with cross-modal-aware
    """

    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__(config, pipeline, stage)

        # build product aware attention
        prd_aware = self.build_multisource_prd_aware_attention()
        self.txt_prd_aware = prd_aware[0]
        if self.use_image:
            self.img_prd_aware = prd_aware[1]

    def build_multisource_prd_aware_attention(self):
        txt_prd_aware = ProductAwareAttention(
            self.config.rvw_txt_encoder.encoder.hidden_dimension)

        outputs = (txt_prd_aware,)
        if self.use_image:
            img_prd_aware = ProductAwareAttention(
                self.config.rvw_img_encoder.encoder_embed_dim)
            outputs += (img_prd_aware,)
        return outputs

    def build_cross_modal_match(self):
        img2txt_match = CrossModalProductAwareAttention(
            left_dim=self.config.prd_img_encoder.encoder_embed_dim,
            hidden_dimension=self.config.rvw_txt_encoder.encoder.hidden_dimension
        )
        txt2img_match = CrossModalProductAwareAttention(
            left_dim=self.config.prd_txt_encoder.encoder.hidden_dimension,
            hidden_dimension=self.config.rvw_img_encoder.encoder_embed_dim
        )
        return img2txt_match, txt2img_match

    def build_coherentor(self):
        coherentor = CoherentReasoning(
            self.config.coherent_encoder,
            prd_txt_dim=self.config.prd_txt_encoder.encoder.hidden_dimension,
            prd_img_dim=self.config.prd_img_encoder.encoder_embed_dim,
            rvw_txt_dim=self.config.rvw_txt_encoder.encoder.hidden_dimension,
            rvw_img_dim=self.config.rvw_img_encoder.encoder_embed_dim,
            max_seq_len=self.config.input_setting.txt_max_length + self.config.input_setting.img_max_length)
        return coherentor

    def cal_features_nums(self):
        pool_config = self.config.pooling
        features_size = pool_config.txt_convs_num * pool_config.txt_filters_num \
            * pool_config.txt_ns + self.config.rvw_txt_encoder.encoder.hidden_dimension

        if self.use_image:
            features_size += (
                pool_config.img_convs_num *
                pool_config.img_filters_num *
                pool_config.img_ns
            )
            features_size += self.config.rvw_txt_encoder.encoder.hidden_dimension
            features_size += self.config.rvw_img_encoder.encoder_embed_dim
            features_size += self.config.coherent_encoder.hidden_dim
            features_size += self.config.rvw_img_encoder.encoder_embed_dim

        return features_size

    def forward(self, batch, wo_score=False):
        # encode part data
        prd_txt_repr, prd_txt_unpadding_mask = self.prd_txt_encoder(
            batch['text_left'], batch['text_left_length'])
        rvw_txt_repr, rvw_txt_unpadding_mask = self.rvw_txt_encoder(
            batch['text_right'], batch['text_right_length'])

        # pooling text
        pool_rvw_txt_repr = torch.stack(rvw_txt_repr, dim=1).sum(1)
        pool_prd_txt_repr = torch.stack(prd_txt_repr, dim=1).sum(1)

        rvw_txt_prd_attn_repr = self.txt_prd_aware(
            pool_prd_txt_repr.transpose(1, 2),
            prd_txt_unpadding_mask.eq(1.),
            pool_rvw_txt_repr.transpose(1, 2),
            rvw_txt_unpadding_mask.eq(1.)
        ).mean(dim=1)

        # cross match
        txt_cross_match = self.cross_match(
            prd_txt_repr, prd_txt_unpadding_mask,
            rvw_txt_repr, rvw_txt_unpadding_mask
        )

        if self.use_image:
            # img encode
            prd_img_repr, prd_img_unpadding_mask = self.prd_img_encoder(
                batch['image_left'].float(), batch['image_left_length'])
            rvw_img_repr, rvw_img_unpadding_mask = self.prd_img_encoder(
                batch['image_right'].float(), batch['image_right_length'])

            rvw_img_prd_attn_repr = self.img_prd_aware(
                prd_img_repr.transpose(1, 2),
                prd_img_unpadding_mask.eq(1.),
                rvw_img_repr.transpose(1, 2),
                rvw_img_unpadding_mask.eq(1.)
            ).mean(dim=1)

            # image cross match
            img_cross_match = self.cross_match(
                [prd_img_repr], prd_img_unpadding_mask,
                [rvw_img_repr], rvw_img_unpadding_mask
            )

            # cross modal aware
            img2txt_match = self.img2txt_match(
                prd_img_repr, prd_img_unpadding_mask.eq(1.),
                pool_rvw_txt_repr, rvw_txt_unpadding_mask.eq(1.)
            ).mean(dim=1)
            txt2img_match = self.txt2img_match(
                pool_prd_txt_repr, prd_txt_unpadding_mask.eq(1.),
                rvw_img_repr, rvw_img_unpadding_mask.eq(1.)
            ).mean(dim=1)

            # coherent reasoning
            coherent_cross_match = self.coherentor(
                pool_rvw_txt_repr, rvw_txt_unpadding_mask,
                rvw_img_repr, rvw_img_unpadding_mask,
                pool_prd_txt_repr, prd_txt_unpadding_mask,
                prd_img_repr, prd_img_unpadding_mask
            )

        # pooling
        pool_result = []
        pool_result.append(self.txt_layernorm(
            self.txt_pooler(txt_cross_match)))
        if self.use_image:
            pool_result.append(self.img_layernorm(
                self.img_pooler(img_cross_match)))

        # context repr
        contex_repr = []
        contex_repr.append(rvw_txt_prd_attn_repr)
        if self.use_image:
            contex_repr.append(rvw_img_prd_attn_repr)
            contex_repr.append(img2txt_match)
            contex_repr.append(txt2img_match)
            contex_repr.append(coherent_cross_match)

        # get score
        input = torch.cat(flatten_all(pool_result) + contex_repr, dim=-1)
        score = self.linear(input)
        return score

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x/math.sqrt(2.0)))

class BertAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, attention_probs_dropout_prob):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size/num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertInducedAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, attention_probs_dropout_prob):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size/num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttnOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossAttnLayer(nn.Module):
    def __init__(self, num_heads, hidden_states, dropout_prob):
        super().__init__()
        self.attn = BertAttention(num_heads, hidden_states, dropout_prob)
        self.output = BertAttnOutput(hidden_states, dropout_prob)

    def forward(self, input_tensor, ctx_tensor, ctx_attn_mask):
        output = self.attn(input_tensor, ctx_tensor, ctx_attn_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfAttnLayer(nn.Module):
    def __init__(self, num_heads, hidden_states, dropout_prob):
        super().__init__()
        self.attn = BertAttention(num_heads, hidden_states, dropout_prob)
        self.output = BertAttnOutput(hidden_states, dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.attn(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertInducedAttnLayer(nn.Module):
    def __init__(self, num_heads, hidden_states, dropout_prob):
        super().__init__()
        self.attn = BertInducedAttention(num_heads, hidden_states, dropout_prob)
        self.output = BertAttnOutput(hidden_states, dropout_prob)


    def forward(self, induced_tensor, input_tensor, attention_mask=None):
        induced_output = self.attn(induced_tensor, input_tensor, attention_mask)
        attention_output = self.output(induced_output, induced_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class InputAttentionBlock(nn.Module):
    def __init__(self, num_features, bert_dropout):
        super().__init__()
        self.input_self_attn = BertSelfAttnLayer(4, num_features, bert_dropout)
    
    def forward(self, input):
        return self.input_self_attn(input)


class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_self_attn = BertSelfAttnLayer(4, config.encoder.hidden_dimension, 0.2)
        self.input_inter = BertIntermediate(config.encoder.hidden_dimension, 3072)
        self.input_output = BertOutput(config.encoder.hidden_dimension, 3072, 0.2)
        
    def input_output_fc(self, input_hidden_states):
        input_inter_hidden_states = self.input_inter(input_hidden_states)
        input_output_states = self.input_output(input_inter_hidden_states, input_hidden_states)
        return input_output_states

    def forward(self, input, input_self_attn_mask):
        input = self.input_self_attn(input, input_self_attn_mask)
        input = self.input_output_fc(input)
        return input


class TextBertEncoder(nn.Module):
    def __init__(self, config, vocab, vocab_name, stage):
        super().__init__()
        self.token_embedding = EmbeddingLayer(
            vocab_map=vocab.v2i,
            embedding_dim=config.embedding.embed_dim,
            vocab_name=vocab_name,
            dropout=config.embedding.dropout,
            embed_type=config.embedding.embed_type,
            padding_index=vocab.pad_index,
            pretrained_dir=config.embedding.pretrained_file,
            stage=stage,
            initial_type=config.embedding.init_type
        )

        layer_list = []
        for _ in range(2):
            layer_list.append(SelfAttentionBlock(config))

        self.layer_list = nn.ModuleList(layer_list)

        self.positional_embedding = nn.Parameter(torch.empty(1, 128, config.embedding.embed_dim))
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.linear_map = nn.Linear(config.embedding.embed_dim, 128)
        self.tanh = nn.Tanh()

    def text_output_fc(self, text_hidden_states):
        text_inter_hidden_states = self.text_inter(text_hidden_states)
        text_output_states = self.text_output(text_inter_hidden_states, text_hidden_states)
        return text_output_states

    def get_self_attn_mask(self, mask):
        mask = (~mask).long()*(-1000)
        text_self_attn_mask = mask.unsqueeze(1).unsqueeze(2)
        return text_self_attn_mask


    def forward(self, input, input_length):
        input = self.token_embedding(input)
        input = input + self.positional_embedding[:, :input.shape[1], :]
        input = self.tanh(self.linear_map(input))

        unpadding_mask = generate_seq_mask(input_length, max_length=input.size(1))
        text_self_attn_mask = self.get_self_attn_mask(unpadding_mask.bool())
        for layer in self.layer_list:
            input = layer(input, text_self_attn_mask)

        return input, unpadding_mask



class ImageBertEncoder(nn.Module):
    def __init__(self, config, stage):
        super().__init__()

        layer_list = []
        for _ in range(2):
            layer_list.append(SelfAttentionBlock(config))

        self.layer_list = nn.ModuleList(layer_list)

        self.linear_map = nn.Linear(2048, 128)
        self.tanh = nn.Tanh()

    def image_output_fc(self, image_hidden_states):
        image_inter_hidden_states = self.image_inter(image_hidden_states)
        image_output_states = self.image_output(image_inter_hidden_states, image_hidden_states)
        return image_output_states

    def get_self_attn_mask(self, mask):
        mask = (~mask).long()*(-1000)
        image_self_attn_mask = mask.unsqueeze(1).unsqueeze(2)
        return image_self_attn_mask


    def forward(self, input, input_length):
        input = self.tanh(self.linear_map(input))

        unpadding_mask = generate_seq_mask(input_length, max_length=input.size(1))
        image_self_attn_mask = self.get_self_attn_mask(unpadding_mask.bool())
        for layer in self.layer_list:
            input = layer(input, image_self_attn_mask)

        return input, unpadding_mask

class SDT(nn.Module):
    def __init__(self, input_dim, output_dim, depth=5, lamda=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.depth = depth
        self.lamda = lamda
        self.internal_node_num = 2 ** self.depth - 1
        self.leaf_node_num = 2 ** self.depth

        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim, self.internal_node_num, bias=False),
            nn.Sigmoid()
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num, self.output_dim, bias=False)

    def forward(self, X):
        mu = self._forward(X)
        y_pred = self.leaf_nodes(mu)
        return y_pred

    def _forward(self, X):
        batch_size = X.size()[0]
        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1-path_prob), dim=2)

        mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            mu = mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            mu = mu * _path_prob

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = mu.view(batch_size, self.leaf_node_num)
        return mu


class CommonSpaceMultimodalLayernormRHPNet3(MultimodalLayernormRHPNet3):
    """The best performance in out dataset.
    """
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__(config, pipeline, stage)
        if self.use_image:
            img_dim = self.config.rvw_img_encoder.encoder_embed_dim
            txt_dim = self.config.rvw_txt_encoder.encoder.hidden_dimension
            hidden_dim = self.config.common_space.hidden_dim
            self.img_linear = nn.Sequential(
                nn.Linear(img_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim))

            self.txt_linear = nn.Sequential(
                nn.Linear(txt_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim))

        num_features = self.cal_features_nums()
        self.mse_linear = nn.Linear(1, 1)
        self.mse_loss = nn.MSELoss()
        self.listwise_hidden = nn.Linear(num_features, 1)
        self.listwise_linear = nn.Linear(1, 1)

        self.bert_dropout = 0.2
        self.input_induced_attn = BertInducedAttnLayer(4, num_features, self.bert_dropout)
        self.input_self_induced_attn = BertInducedAttnLayer(4, num_features, self.bert_dropout)
        self.input_inter = BertIntermediate(num_features, 3072)
        self.input_output = BertOutput(num_features, 3072, self.bert_dropout)

        self.cross_modal_self_attn = BertSelfAttnLayer(4, self.config.rvw_img_encoder.encoder_embed_dim, self.bert_dropout)
        self.cross_modal_inter = BertIntermediate(self.config.rvw_txt_encoder.encoder.hidden_dimension, 3072)
        self.cross_modal_output = BertOutput(self.config.rvw_txt_encoder.encoder.hidden_dimension, 3072, self.bert_dropout)

        self.num_induced_tensor = 20
        prd_vocab = self.pipeline.prd_text_field.vocab
        rvw_vocab = self.pipeline.rvw_text_field.vocab
        self.prd_txt_encoder = TextBertEncoder(self.config.prd_txt_encoder, prd_vocab, 'prd_vocab', self.stage)
        self.rvw_txt_encoder = TextBertEncoder(self.config.rvw_txt_encoder, rvw_vocab, 'rvw_vocab', self.stage)
        self.prd_img_encoder = ImageBertEncoder(self.config.prd_txt_encoder, self.stage)
        self.rvw_img_encoder = ImageBertEncoder(self.config.rvw_txt_encoder, self.stage)
        self.txt_cross_match = BertCrossMatchLayer(num_repeat=9)
        self.img_cross_match = BertCrossMatchLayer(num_repeat=1)

        self.sdt = SDT(num_features, 1, depth=self.config.train.tree_depth)


    def build_cross_modal_match(self):
        img2txt_match = ProductAwareAttention(
            self.config.common_space.hidden_dim)
        txt2img_match = ProductAwareAttention(
            self.config.common_space.hidden_dim)
        return img2txt_match, txt2img_match

    def build_coherentor(self):
        coherentor = CoherentReasoning(
            self.config.coherent_encoder,
            prd_txt_dim=self.config.common_space.hidden_dim,
            prd_img_dim=self.config.common_space.hidden_dim,
            rvw_txt_dim=self.config.common_space.hidden_dim,
            rvw_img_dim=self.config.common_space.hidden_dim,
            max_seq_len=self.config.input_setting.txt_max_length + self.config.input_setting.img_max_length)
        return coherentor

    def cal_features_nums(self):
        pool_config = self.config.pooling
        features_size = pool_config.txt_convs_num * pool_config.txt_filters_num \
            * pool_config.txt_ns + self.config.rvw_txt_encoder.encoder.hidden_dimension

        if self.use_image:
            features_size += (
                pool_config.img_convs_num *
                pool_config.img_filters_num *
                pool_config.img_ns
            )
            features_size += self.config.common_space.hidden_dim
            features_size += self.config.common_space.hidden_dim
            features_size += self.config.coherent_encoder.hidden_dim
            features_size += self.config.rvw_img_encoder.encoder_embed_dim

        return features_size

    def transform_features(self, x):
        x = torch.log(1 + torch.abs(x)) * torch.sign(x)
        return x

    def get_cross_attn_mask(self, mask):
        mask = (~mask).long()*(-1000)
        cross_attn_mask = mask.unsqueeze(1).unsqueeze(2)
        return cross_attn_mask

    def cross_modal_output_fc(self, cross_modal_hidden_states):
        cross_modal_inter_hidden_states = self.cross_modal_inter(cross_modal_hidden_states)
        cross_modal_output_states = self.cross_modal_output(cross_modal_inter_hidden_states, cross_modal_hidden_states)
        return cross_modal_output_states


    def forward(self, batch, labels=None, wo_score=False, stage='train', data_mode='list'):
        if data_mode == 'list':
            left, right = batch
            bs = len(left['text_left'])
            list_len = len(right['text_right']) // bs

            _, text_len = left['text_left'].shape
            left['text_left'] = left['text_left'].reshape(bs, 1, text_len).repeat(1, list_len, 1).reshape(bs*list_len, text_len)
            left['text_left_length'] = left['text_left_length'].reshape(bs, 1).repeat(1, list_len).reshape(bs*list_len)

            text_left = left['text_left']
            text_right = right['text_right']
            text_left_length = left['text_left_length']
            text_right_length = right['text_right_length']
            if labels is not None:
                labels = labels.reshape(bs, list_len).float()

            if self.use_image:
                _, shape1, shape2 = left['image_left'].shape
                left['image_left'] = left['image_left'].reshape(bs, 1, shape1, shape2).repeat(1, list_len, 1, 1).reshape(bs*list_len, shape1, shape2)
                left['image_left_length'] = left['image_left_length'].reshape(bs, 1).repeat(1, list_len).reshape(bs*list_len)
                image_left = left['image_left']
                image_right = right['image_right']
                image_left_length = left['image_left_length']
                image_right_length = right['image_right_length']

        else:
            text_left = batch['text_left']
            text_right = batch['text_right']
            text_left_length = batch['text_left_length']
            text_right_length = batch['text_right_length']
            if self.use_image:
                image_left = batch['image_left']
                image_right = batch['image_right']
                image_left_length = batch['image_left_length']
                image_right_length = batch['image_right_length']

        # encode part data
        prd_txt_repr, prd_txt_unpadding_mask = self.prd_txt_encoder(text_left, text_left_length)
        rvw_txt_repr, rvw_txt_unpadding_mask = self.rvw_txt_encoder(text_right, text_right_length)

        # pooling text
        pool_rvw_txt_repr = rvw_txt_repr
        pool_prd_txt_repr = prd_txt_repr

        if self.use_image:
            # img encode
            prd_img_repr, prd_img_unpadding_mask = self.prd_img_encoder(
                image_left.float(), image_left_length)
            rvw_img_repr, rvw_img_unpadding_mask = self.rvw_img_encoder(
                image_right.float(), image_right_length)

            cross_modal_inputs = torch.cat([pool_prd_txt_repr, pool_rvw_txt_repr, prd_img_repr, rvw_img_repr], 1)
            cross_modal_mask = torch.cat([prd_txt_unpadding_mask, rvw_txt_unpadding_mask, prd_img_unpadding_mask, rvw_img_unpadding_mask], 1)
            cross_modal_mask = self.get_cross_attn_mask(cross_modal_mask.bool())
            cross_modal_inputs = self.cross_modal_self_attn(cross_modal_inputs, cross_modal_mask)
            cross_modal_inputs = self.cross_modal_output_fc(cross_modal_inputs)

            pool_prd_txt_repr = cross_modal_inputs[:, :pool_prd_txt_repr.size(1), :]
            pool_rvw_txt_repr = cross_modal_inputs[:, pool_prd_txt_repr.size(1):pool_prd_txt_repr.size(1)+pool_rvw_txt_repr.size(1), :]
            prd_img_repr = cross_modal_inputs[:, pool_prd_txt_repr.size(1)+pool_rvw_txt_repr.size(1):pool_prd_txt_repr.size(1)+pool_rvw_txt_repr.size(1)+prd_img_repr.size(1), :]
            rvw_img_repr = cross_modal_inputs[:, pool_prd_txt_repr.size(1)+pool_rvw_txt_repr.size(1)+prd_img_repr.size(1):, :]

            rvw_img_prd_attn_repr = self.img_prd_aware(
                prd_img_repr.transpose(1, 2),
                prd_img_unpadding_mask.eq(1.),
                rvw_img_repr.transpose(1, 2),
                rvw_img_unpadding_mask.eq(1.)
            ).mean(dim=1)

            # image cross match
            img_cross_match = self.img_cross_match(
                prd_img_repr, prd_img_unpadding_mask,
                rvw_img_repr, rvw_img_unpadding_mask
            )

            # mapping to a common space
            common_prd_txt_repr = self.txt_linear(pool_prd_txt_repr)
            common_rvw_txt_repr = self.txt_linear(pool_rvw_txt_repr)
            common_prd_img_repr = self.img_linear(prd_img_repr)
            common_rvw_img_repr = self.img_linear(rvw_img_repr)

            # cross modal aware
            img2txt_match = self.img2txt_match(
                common_prd_img_repr.transpose(
                    1, 2), prd_img_unpadding_mask.eq(1.),
                common_rvw_txt_repr.transpose(
                    1, 2), rvw_txt_unpadding_mask.eq(1.)
            ).mean(dim=1)
            txt2img_match = self.txt2img_match(
                common_prd_txt_repr.transpose(
                    1, 2), prd_txt_unpadding_mask.eq(1.),
                common_rvw_img_repr.transpose(
                    1, 2), rvw_img_unpadding_mask.eq(1.)
            ).mean(dim=1)

            # coherent reasoning
            coherent_cross_match = self.coherentor(
                common_rvw_txt_repr, rvw_txt_unpadding_mask,
                common_rvw_img_repr, rvw_img_unpadding_mask,
                common_prd_txt_repr, prd_txt_unpadding_mask,
                common_prd_img_repr, prd_img_unpadding_mask
            )

        rvw_txt_prd_attn_repr = self.txt_prd_aware(
            pool_prd_txt_repr.transpose(1, 2),
            prd_txt_unpadding_mask.eq(1.),
            pool_rvw_txt_repr.transpose(1, 2),
            rvw_txt_unpadding_mask.eq(1.)
        ).mean(dim=1)

        # cross match
        txt_cross_match = self.txt_cross_match(
            prd_txt_repr, prd_txt_unpadding_mask,
            rvw_txt_repr, rvw_txt_unpadding_mask
        )


        # pooling
        pool_result = []
        pool_result.append(self.txt_layernorm(
            self.txt_pooler(txt_cross_match)))
        if self.use_image:
            pool_result.append(self.img_layernorm(
                self.img_pooler(img_cross_match)))

        # context repr
        contex_repr = []
        contex_repr.append(rvw_txt_prd_attn_repr)
        if self.use_image:
            contex_repr.append(rvw_img_prd_attn_repr)
            contex_repr.append(img2txt_match)
            contex_repr.append(txt2img_match)
            contex_repr.append(coherent_cross_match)

        # get score
        input = torch.cat(flatten_all(pool_result) + contex_repr, dim=-1)
        pairwise_score = self.linear(input)

        # list-wise score
        if stage == 'train':
            num_text_right = self.config.train.num_neg + 1
        else:
            num_text_right = text_right.shape[0]

        listwise_input = input.reshape(-1, num_text_right, input.shape[-1])
        # listwise_input = self.transform_features(listwise_input)
        induced_tensor = torch.empty(listwise_input.shape[0], self.num_induced_tensor, listwise_input.shape[-1]).normal_(0, 0.1).cuda()
        induced_input = self.input_induced_attn(induced_tensor, listwise_input)
        listwise_input = self.input_self_induced_attn(listwise_input, induced_input)
        # listwise_input = self.input_self_attn_seqs(listwise_input)
        listwise_inter_input = self.input_inter(listwise_input)
        listwise_input = self.input_output(listwise_inter_input, listwise_input)
        # listwise_input = self.transform_features(input)
        # listwise_input = input.reshape(-1, num_text_right, input.shape[-1])
        # listwise_attention = torch.softmax(torch.matmul(F.normalize(listwise_input, p=2, dim=-1), F.normalize(listwise_input, p=2, dim=-1).permute(0,2,1)), -1)
        # listwise_score = self.listwise_hidden(torch.matmul(listwise_attention, listwise_input)).reshape(-1, 1)

        if self.config.train.listwise_score_only:
            listwise_score = self.sdt(listwise_input.reshape(-1, listwise_input.shape[-1]))
            score = listwise_score
        elif self.config.train.latent_cross:
            listwise_score = self.listwise_hidden(listwise_input).reshape(-1, 1)
            score = (1 + listwise_score) * pairwise_score
        else:
            listwise_score = self.listwise_hidden(listwise_input).reshape(-1, 1)
            score = (listwise_score + pairwise_score)/2

        if stage == 'test':
            return score, pairwise_score, listwise_score
        else:        
            return score, pairwise_score, listwise_score

        # get mse_loss
        # if labels is not None:
        #     real_labels = self.mse_linear(labels).detach()
        #     mse_loss = self.mse_loss(score, real_labels)

        #     return score, mse_loss
        # else:
        #     return score
