import torch
import typing
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from matchzoo.preprocessors.units import Vocabulary
from matchzoo.pipeline.rhp_pipeline import RHPPipeline

from matchzoo.modules.cnn import ConvEncoder
from matchzoo.modules.embedding_layer import EmbeddingLayer
from matchzoo.modules.transformer import TransformerEncoderLayer
from matchzoo.modules.cross_match import CrossMatchLayer
from matchzoo.modules.cross_modal_match import CrossModalMatchLayer
from matchzoo.modules.kernel_max import KernelMaxPooling
from matchzoo.modules.utils import generate_seq_mask, flatten_all 
from matchzoo.modules.coherent import CoherentEncoder

class Conv2DEncoder(nn.Module):
	def __init__(self, input_size: int, kernel_size: typing.List[int], kernel_num:typing.List[int], activation: str = 'ReLU', padding_index: int = 0):
		super().__init__()
		self.conv_layer = nn.ModuleList()
		self.kernel_size = kernel_size
		self.kernel_num = kernel_num
		self.activation = activation
		activation_class: nn.Module = getattr(nn, activation, None)

		for ks in self.kernel_size:
			modules: typing.Tuple[nn.Module] = (
				Conv2d(
						in_channels = 1,
						out_channels = kernel_num,
						kernel_size = (ks, input_size),
						stride = (1, 1),
						padding = ((ks - 1) // 2, 0),
					),
			)
			if activation_class:
				modules = modules + (activation_class(),)
			self.conv_layer.append(
				nn.Sequential(*modules)
			)

	def forward(self, input: torch.Tensor, input_length: torch.Tensor) -> typing.Tuple[typing.List[torch.Tensor], torch.Tensor]:
		unpadding_mask = generate_seq_mask(input_length, max_length=input.size(1))
		mask = unpadding_mask.unsqueeze(-1)
		input_convs = []
		for layer in self.conv_layer:
			convs = layer(input.unsqueeze(1)).squeeze(-1).permute(0, 2, 1) # (bs, num_kernel, seq_len, 1) -> (bs, seq_len, num_kernel)
			convs = convs * mask
			input_convs.append(convs)
        
		return input_convs, unpadding_mask 


class TextCNNEncoder(nn.Module):
    def __init__(self, config, vocab: Vocabulary, vocab_name: str, stage: str):
        super().__init__()
#        self.token_embedding = EmbeddingLayer(
#            vocab_map=vocab.v2i,
#            embedding_dim=config.embedding.embed_dim,
#            vocab_name=vocab_name,
#            dropout=config.embedding.dropout,
#            embed_type=config.embedding.embed_type,
#            padding_index=vocab.pad_index,
#            pretrained_dir=config.embedding.pretrained_file,
#            stage=stage,
#            initial_type=config.embedding.init_type
#        )

        configuration = BertConfig()
        self.token_embedding = BertModel(configuration)

        self.seq_encoder = Conv2DEncoder(
            input_size=config.embedding.embed_dim,
            kernel_size=config.encoder.kernel_size,
            kernel_num=config.encoder.hidden_dimension,
            padding_index=1
        )
        for param in self.seq_encoder.parameters():
            param.requires_grad = False

    def forward(self, input, input_length):
        input = self.token_embedding(input)[0]
        input, unpadding_mask = self.seq_encoder(input, input_length)
        return input, unpadding_mask

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.GRU(
            input_size = input_size,
            hidden_size = output_size,
            num_layers = 1,
            dropout = dropout
        )

    def forward(self, input_text, input_lengths):
        packed_seq = pack_padded_sequence(input_text, input_lengths, batch_first=True, enforce_sorted=False)
        o, h = self.lstm(packed_seq)
        o = pad_packed_sequence(o, batch_first=True)
        # unpadding_mask = generate_seq_mask(input_lengths, max_length=input_text.size(1))
        # mask = unpadding_mask.unsqueeze(-1)
        return (h.squeeze(0), o[0])  

class TextLSTMEncoder(nn.Module):
    def __init__(self, config, embedding_layer, vocab: Vocabulary, vocab_name: str, stage: str):
        super().__init__()
       # self.token_embedding = EmbeddingLayer(
       #     vocab_map=vocab.v2i,
       #     embedding_dim=config.embedding.embed_dim,
       #     vocab_name=vocab_name,
       #     dropout=config.embedding.dropout,
       #     embed_type=config.embedding.embed_type,
       #     padding_index=vocab.pad_index,
       #     pretrained_dir=config.embedding.pretrained_file,
       #     stage=stage,
       #     initial_type=config.embedding.init_type
       # )

        self.token_embedding = embedding_layer
        self.seq_encoder = LSTMEncoder(
            input_size=config.embedding.embed_dim,
            output_size=config.encoder.hidden_dimension,
        )

    def forward(self, input, input_length):
        unpadding_mask = generate_seq_mask(input_length, max_length=input.size(1))
        input = self.token_embedding(input, unpadding_mask)[0]
        output = self.seq_encoder(input, input_length)
        unpadding_mask = unpadding_mask.unsqueeze(-1)
        return output, unpadding_mask


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

class ReviewSelfAttention(nn.Module):
    def __init__(self, hidden_dimension):
        super(ReviewSelfAttention, self).__init__()
        self.proj1 = nn.Linear(hidden_dimension, hidden_dimension)
        self.proj2 = nn.Linear(hidden_dimension, hidden_dimension)

    def forward(self, review_repr, attention_mask, unpadding_mask):
        q = self.proj1(review_repr) # (bs, seq_len, hidden)
        v = self.proj2(review_repr)
     
        # attention_mask (bs, seq_len)
        logits = torch.matmul(q, v.permute(0,2,1))
        # logits = logits * torch.matmul(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # (bs, seq_len, seq_len)
        logits = logits * unpadding_mask.float() + (~unpadding_mask).float()*(-1e23)
        att_weights = torch.softmax(logits, dim=-1)
        att_weights = torch.softmax(logits, dim=-1) * (torch.matmul(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)))
        r = torch.matmul(att_weights, review_repr) + review_repr
        return r

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
        q = q * p_mask.float() + (~p_mask).float() * (-1e23)
        q = torch.softmax(q, dim=1)


        r_add = torch.matmul(product_repr, q)
        r = r_add + review_repr   # (batch, hidden_dimension, review_seq_lens)

        r = r.transpose(1, 2)  # (batch, review_seq_lens, hidden_dimension)
        r_mask = review_unpadding_mask  # (batch, review_seq_lens)
        r_mask = r_mask.unsqueeze(-1)
        r = r * r_mask.float()  # (batch, review_seq_lens, hidden_dimension)
        return r

class CrossModalTransformer(nn.Module):
    def __init__(self, in_dim1, in_dim2, attn_dim, dropout=0.2, num_heads=4):
        super().__init__()
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(in_dim1, attn_dim)
        self.kv_proj = nn.Linear(in_dim2, attn_dim)

        self.out_proj = nn.Linear(attn_dim, attn_dim)
        self.num_heads = num_heads
        self.layer_norm1 = nn.LayerNorm(attn_dim)
        self.layer_norm2 = nn.LayerNorm(attn_dim)

        self.ffn = nn.Sequential(
                nn.Linear(attn_dim, attn_dim),
                nn.Tanh(),
                nn.Linear(attn_dim, attn_dim)
        )

    def forward(self, modal1, modal2, unpadding_mask1, unpadding_mask2):
        '''
        attn_mask -> torch.Tensor(bs, seq_len1)
        '''
        residual = modal1
        q = self.q_proj(modal1) # (bs, seq_len1, attn_dim)
        kv = self.kv_proj(modal2) # (bs, seq_len2, attn_dim)

        bs = q.size(0)
        q = q.view(bs, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = kv.view(bs, -1, self.num_heads, self.head_dim).permute(0,2,3,1)
        v = k.permute(0,1,3,2)

        attn_mask = torch.matmul(unpadding_mask1.float(), unpadding_mask2.float().unsqueeze(-1).transpose(1,2)) > 0 # (bs, seq_len1, seq_len2)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # (bs, num_heads, seq_len1, seq_len2)
        attn_logits = torch.matmul(q, k)/(float(self.attn_dim)**0.5)
        attn_logits = attn_logits + ((~attn_mask).float()*(-1e23)) # (bs, num_head, seq_len1, seq_len2)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_res = torch.matmul(attn_weights, v).permute(0,2,1,3).contiguous().view(bs, -1, self.attn_dim) # (bs, num_head, seq_len1, head_dim)
       
        x = F.dropout(attn_res, p=self.dropout)
        x = x + residual
        x = self.layer_norm1(x)

        residual = x
        x = self.ffn(x)       
        x = F.dropout(attn_res, p=self.dropout)
        x = x + residual
        x = self.layer_norm2(x)
        return x

class MultimodalRHPNet(nn.Module):
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self.stage = stage
        self.use_image = config.input_setting.use_img

        # build encoder
        # self.prd_txt_encoder, self.rvw_txt_encoder = self.build_text_encoder()
        if self.use_image:
            self.prd_img_encoder, self.rvw_img_encoder = self.build_image_encoder()

        # build cross matching
        self.cross_match = CrossMatchLayer(
            do_normalize=config.cross_match.do_normalize)

        # build cross modal matching
        # if self.use_image:
        #    self.img2txt_match, self.txt2img_match = self.build_cross_modal_match()

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
            # pool_result.append(self.img2txt_layernorm(img2txt_match))
            # pool_result.append(self.txt2img_layernorm(txt2img_match))
            # pool_result.append(self.coherent_layernorm(coherent_cross_match))

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

class CPC(nn.Module):
    def __init__(self, hidden_dim, mode='txt_img'):
        super(CPC,self).__init__()
        assert mode in ['txt_img', 'rvw_prd']
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.prd_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rvw_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def cm_forward(self, prd_txt, prd_img, rvw_txt, rvw_img, pos_mask, labels):
        preds_prd_img = self.prd_pred(prd_txt)
        preds_rvw_img = self.rvw_pred(rvw_txt)
        prd_scores = (F.normalize(preds_prd_img) * F.normalize(prd_img)).sum(-1)
        rvw_scores = (F.normalize(preds_rvw_img) * F.normalize(rvw_img)).sum(-1)    # bs
        masked_pos_rvw_scores = rvw_scores * torch.masked_fill(torch.full(pos_mask.size(), 1.0).cuda(), ~pos_mask, 0.0)
        # masked_neg_rvw_scores = rvw_scores * torch.masked_fill(torch.full(pos_mask.size(), 1.0).cuda(), pos_mask, -1e-15)
        neg_total = torch.logsumexp(rvw_scores, dim=-1) # (bs,)

        pos_total = prd_scores.mean() + masked_pos_rvw_scores.mean()
        neg_total = neg_total.mean()
        return pos_total - neg_total

    def rp_forward(self, prd_txt, prd_img, rvw_txt, rvw_img, pos_mask,labels):
        preds_rvw_txt = self.prd_pred(prd_txt)
        preds_rvw_img = self.rvw_pred(prd_img)
        rvw_img_scores = (F.normalize(preds_rvw_img)*F.normalize(rvw_img)).sum(-1)
        rvw_txt_scores = (F.normalize(preds_rvw_txt)*F.normalize(rvw_txt)).sum(-1)
        masked_pos_img_scores = rvw_img_scores * torch.masked_fill(torch.full(pos_mask.size(), 1.0).cuda(), ~pos_mask, -1e-15)
        masked_neg_img_scores = rvw_txt_scores * torch.masked_fill(torch.full(pos_mask.size(), 1.0).cuda(), pos_mask, -1e-15)
        
        pos_total = torch.logsumexp(masked_pos_img_scores, dim=0)
        neg_total = torch.logsumexp(masked_neg_img_scores, dim=0)
        return pos_total - neg_total

    def forward(self, prd_txt, prd_img, rvw_txt, rvw_img, labels):
        pos_mask = (labels >= 3.0)[:,0].cuda()
        if self.mode == 'txt_img':
            score = self.cm_forward(prd_img, prd_img, rvw_img, rvw_img, pos_mask, labels)  
        elif self.mode == 'rvw_prd':
            score = self.rp_forward(prd_img, prd_img, rvw_img, rvw_img, pos_mask, labels)
        return score


class CommonSpaceMultimodalLayernormRHPNet3(MultimodalLayernormRHPNet3):
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__(config, pipeline, stage)
        self.txt_rvw_self_attn = self.build_self_attention()
        self.beta_proj = self.build_beta_proj()
         
        if self.use_image:
            img_dim = self.config.rvw_img_encoder.encoder_embed_dim
            txt_dim = self.config.rvw_txt_encoder.encoder.hidden_dimension
            hidden_dim = self.config.common_space.hidden_dim
            self.img_linear_cm = nn.Sequential(
                nn.Linear(img_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim))

            self.txt_linear_cm = nn.Sequential(
                nn.Linear(txt_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim))

            self.img_linear_rp = nn.Sequential(
                nn.Linear(img_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim))

            self.txt_linear_rp = nn.Sequential(
                nn.Linear(txt_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim))

            self.cross_modal_contrast = self.build_contrast(mode='txt_img')
            self.rvw_prd_contrast = self.build_contrast(mode='rvw_prd')

    def build_beta_proj(self):
        hidden_dim = self.config.rvw_txt_encoder.encoder.hidden_dimension
        beta_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        return beta_proj

    def build_self_attention(self):
        txt_rvw_self_attn = ReviewSelfAttention(
            self.config.rvw_txt_encoder.encoder.hidden_dimension
        )
        return txt_rvw_self_attn

    def build_contrast(self, mode):
        cpc = CPC(self.config.common_space.hidden_dim, mode)
        return cpc

    def build_cross_modal_match(self):
        img2txt_match = ProductAwareAttention(
            self.config.common_space.hidden_dim)
        txt2img_match = ProductAwareAttention(
            self.config.common_space.hidden_dim)
        return img2txt_match, txt2img_match

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

        # return features_size
        return 8* self.config.common_space.hidden_dim

    def build_text_encoder(self):
        prd_vocab = self.pipeline.prd_text_field.vocab
        rvw_vocab = self.pipeline.rvw_text_field.vocab
        prd_txt_encoder = TextCNNEncoder(
            self.config.prd_txt_encoder, prd_vocab, 'prd_vocab', self.stage)
        rvw_txt_encoder = TextCNNEncoder(
            self.config.rvw_txt_encoder, rvw_vocab, 'rvw_vocab', self.stage)
        return prd_txt_encoder, rvw_txt_encoder
 
    def forward(self, batch, wo_score=False):
        # encode part data (List[Tensor(bs x seq_len x hidden_size x 1) 
        prd_txt_repr, prd_txt_unpadding_mask = self.prd_txt_encoder(
            batch['text_left'], batch['text_left_length'])
        rvw_txt_repr, rvw_txt_unpadding_mask = self.rvw_txt_encoder(
            batch['text_right'], batch['text_right_length'])

        txt_prd_length = batch['text_left_length'].unsqueeze(-1)
        txt_rvw_length = batch['text_right_length'].unsqueeze(-1)
        # pooling text (Tensor(bs x seq_len x hidden_size))
        pool_rvw_txt_repr = torch.stack(rvw_txt_repr, dim=3).mean(3)
        pool_prd_txt_repr = torch.stack(prd_txt_repr, dim=3).mean(3)
	
        # pad the attention mask
        attn_mask = [torch.from_numpy(x[:pool_rvw_txt_repr.size(1)]) for x in batch['attention_mask']]
        attn_mask = [torch.masked_fill(torch.ones_like(x)*0.5, x, 1.0) for x in attn_mask]
        attn_mask = pad_sequence(attn_mask, batch_first=True).cuda()
       
        # perform self attention on rvw
        rvw_txt_self_attn_repr = self.txt_rvw_self_attn(pool_rvw_txt_repr, attn_mask, rvw_txt_unpadding_mask)  # (bs, seq_len, hidden_size)      
        rvw_txt_prd_attn_repr = self.txt_prd_aware(
            pool_prd_txt_repr.transpose(1, 2),
            prd_txt_unpadding_mask.eq(1.),
            rvw_txt_self_attn_repr.transpose(1, 2),
            rvw_txt_unpadding_mask.eq(1.),
        )
       
        pool_rvw_txt_repr = (rvw_txt_prd_attn_repr * attn_mask.unsqueeze(-1)).sum(1)/attn_mask.sum(1, keepdim=True)  # (bs, hidden_size)
        pool_prd_txt_repr = (pool_prd_txt_repr * prd_txt_unpadding_mask.unsqueeze(-1)).sum(1)/txt_prd_length

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

           
            pool_prd_img_repr = prd_img_repr.sum(1)/batch['image_left_length'].unsqueeze(-1)
            pool_rvw_img_repr = rvw_img_prd_attn_repr
            # mapping to two common spaces
            common_prd_txt_repr_cm = self.txt_linear_cm(pool_prd_txt_repr)
            common_rvw_txt_repr_cm = self.txt_linear_cm(pool_rvw_txt_repr)
            common_prd_img_repr_cm = self.img_linear_cm(pool_prd_img_repr)
            common_rvw_img_repr_cm = self.img_linear_cm(pool_rvw_img_repr)
            common_prd_txt_repr_rp = self.txt_linear_rp(pool_prd_txt_repr)
            common_rvw_txt_repr_rp = self.txt_linear_rp(pool_rvw_txt_repr)
            common_prd_img_repr_rp = self.img_linear_rp(pool_prd_img_repr)
            common_rvw_img_repr_rp = self.img_linear_rp(pool_rvw_img_repr)

	# contrastive learning
        if self.training:
            labels = batch['labels']        
            cpc_score = self.cross_modal_contrast(common_prd_txt_repr_cm, common_prd_img_repr_cm, common_rvw_txt_repr_cm, common_rvw_img_repr_cm, labels)
            cpc_score += self.rvw_prd_contrast(common_prd_txt_repr_rp, common_prd_img_repr_rp, common_rvw_txt_repr_rp, common_rvw_img_repr_rp, labels)
        else: cpc_score = 0
        alpha = self.config.cpc.alpha

        # context repr
        contex_repr = []
        contex_repr.append(common_rvw_txt_repr_cm)
        contex_repr.append(common_rvw_txt_repr_rp)

        contex_repr.append(common_prd_txt_repr_cm)
        contex_repr.append(common_prd_txt_repr_rp)

        if self.use_image:
            contex_repr.append(common_rvw_img_repr_cm)
            contex_repr.append(common_rvw_img_repr_rp)

            contex_repr.append(common_prd_txt_repr_cm)
            contex_repr.append(common_prd_txt_repr_rp)

        # get score
        input = torch.cat(contex_repr, dim=-1)
        score = self.linear(input)
        
        if self.training:
            return score, cpc_score*alpha
        else:
            return score

class CommonSpaceMultimodalLayernormBertRHPNet3(CommonSpaceMultimodalLayernormRHPNet3):
	def __init__(self, config, pipeline: RHPPipeline, stage: str):
		super().__init__(config, pipeline, stage)
		self.bert_encoder = self.build_bert_encoder()
		self.prd_txt_encoder, self.rvw_txt_encoder = self.build_text_encoder()
		self.txt_rvw_self_attn = self.build_self_attention()
		self.beta_proj = self.build_beta_proj()
 
	def build_bert_encoder(self):
		configuration = BertConfig()
		bert_encoder = BertModel(configuration)

		for name, param in bert_encoder.named_parameters():
			if "bertmodel.encoder.layer" in name:
				layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
				if layer_num <= 7:
					param.requires_grad = False
		return bert_encoder
 
	def cal_features_nums(self):
		# return features_size
		return (8 if self.use_image else 4)*self.config.common_space.hidden_dim

	def build_text_encoder(self):
		prd_vocab = self.pipeline.prd_text_field.vocab
		rvw_vocab = self.pipeline.rvw_text_field.vocab
		prd_txt_encoder = TextLSTMEncoder(
			self.config.prd_txt_encoder, self.bert_encoder, prd_vocab, 'prd_vocab', self.stage)
		rvw_txt_encoder = TextLSTMEncoder(
			self.config.rvw_txt_encoder, self.bert_encoder, rvw_vocab, 'rvw_vocab', self.stage)
		return prd_txt_encoder, rvw_txt_encoder
 
	def forward(self, batch, wo_score=False):
		# encode part data (List[Tensor(bs x seq_len x hidden_size x 1) 
		(prd_txt_seqrepr, prd_txt_repr), prd_txt_unpadding_mask = self.prd_txt_encoder(
			batch['text_left'], batch['text_left_length'])
		(rvw_txt_seqrepr, rvw_txt_repr), rvw_txt_unpadding_mask = self.rvw_txt_encoder(
			batch['text_right'], batch['text_right_length'])

		txt_prd_length = batch['text_left_length'].unsqueeze(-1)
		txt_rvw_length = batch['text_right_length'].unsqueeze(-1)
		# pooling text (Tensor(bs x seq_len x hidden_size))
		beta = self.beta_proj(rvw_txt_seqrepr)

		# pad the attention mask
		attn_mask = [torch.from_numpy(x[:rvw_txt_repr.size(1)]) for x in batch['attention_mask']]
		oppo_attn_mask = [torch.from_numpy(~x[:rvw_txt_repr.size(1)]) for x in batch['attention_mask']]
		attn_mask = pad_sequence(attn_mask, batch_first=True).cuda()
		oppo_attn_mask = pad_sequence(oppo_attn_mask, batch_first=True).cuda()
		attn_mask = 1.0 * attn_mask + oppo_attn_mask * beta

		# perform self attention on rvw
		rvw_txt_self_attn_repr = self.txt_rvw_self_attn(rvw_txt_repr, attn_mask, rvw_txt_unpadding_mask)  # (bs, seq_len, hidden_size)     
		rvw_txt_prd_attn_repr = self.txt_prd_aware(
			prd_txt_repr.transpose(1, 2),
			prd_txt_unpadding_mask.eq(1.).squeeze(-1),
			rvw_txt_self_attn_repr.transpose(1, 2),
			rvw_txt_unpadding_mask.eq(1.).squeeze(-1),
		)

		pooled_rvw_txt_repr = (rvw_txt_prd_attn_repr * attn_mask.unsqueeze(-1)).sum(1)/attn_mask.sum(1, keepdim=True)  # (bs, hidden_size)
		pooled_prd_txt_repr = (prd_txt_repr * prd_txt_unpadding_mask).sum(1)/txt_prd_length

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
			)# .mean(dim=1)


			pooled_prd_img_repr = prd_img_repr.sum(1)/batch['image_left_length'].unsqueeze(-1)
			pooled_rvw_img_repr = rvw_img_prd_attn_repr.mean(1)
			# mapping to two common spaces
			common_prd_txt_repr_cm = self.txt_linear_cm(pooled_prd_txt_repr)
			common_rvw_txt_repr_cm = self.txt_linear_cm(pooled_rvw_txt_repr)
			common_prd_img_repr_cm = self.img_linear_cm(pooled_prd_img_repr)
			common_rvw_img_repr_cm = self.img_linear_cm(pooled_rvw_img_repr)
			common_prd_txt_repr_rp = self.txt_linear_rp(pooled_prd_txt_repr)
			common_rvw_txt_repr_rp = self.txt_linear_rp(pooled_rvw_txt_repr)
			common_prd_img_repr_rp = self.img_linear_rp(pooled_prd_img_repr)
			common_rvw_img_repr_rp = self.img_linear_rp(pooled_rvw_img_repr)

		# contrastive learning
		if self.training:
			labels = batch['labels']        
			cpc_score = self.cross_modal_contrast(common_prd_txt_repr_cm, common_prd_img_repr_cm, common_rvw_txt_repr_cm, common_rvw_img_repr_cm, labels)
			cpc_score += self.rvw_prd_contrast(common_prd_txt_repr_rp, common_prd_img_repr_rp, common_rvw_txt_repr_rp, common_rvw_img_repr_rp, labels)
		else: cpc_score = 0
		alpha = self.config.cpc.alpha

		# context repr
		contex_repr = []
		contex_repr.append(common_rvw_txt_repr_cm)
		contex_repr.append(common_rvw_txt_repr_rp)

		contex_repr.append(common_prd_txt_repr_cm)
		contex_repr.append(common_prd_txt_repr_rp)

		if self.use_image:
			contex_repr.append(common_rvw_img_repr_cm)
			contex_repr.append(common_rvw_img_repr_rp)

			contex_repr.append(common_prd_img_repr_cm)
			contex_repr.append(common_prd_img_repr_rp)

		# get score
		input = torch.cat(contex_repr, dim=-1)
		score = self.linear(input)
        
		if self.training:
			return score, cpc_score*alpha
		else:
			return score


class CommonSpaceMultimodalLayernormRHPNet3Old(MultimodalLayernormRHPNet3):
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
	
        # get score
        input = torch.cat(flatten_all(pool_result) + contex_repr, dim=-1)
        score = self.linear(input)
        return score
