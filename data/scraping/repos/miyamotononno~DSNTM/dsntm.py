from unicodedata import decimal
import torch
import torch.nn.functional as F 
import numpy as np
from torch import dtype, nn
import data
import math

from coherence import compute_coherence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TINY = 1e-10

class DSNTM(nn.Module):
    def __init__(self, args, embeddings):
        super(DSNTM, self).__init__()

        ## define hyperparameters
        self.num_topics = args.num_topics
        self.num_times = args.num_times
        self.vocab_size = args.vocab_size
        self.t_hidden_size = args.t_hidden_size
        self.eta_hidden_size = args.eta_hidden_size
        self.rho_size = args.rho_size
        self.emsize = args.emb_size
        self.enc_drop = args.enc_drop
        self.eta_nlayers = args.eta_nlayers
        self.t_drop = nn.Dropout(args.enc_drop)
        self.delta = args.delta
        self.citation = args.citation

        # define the topic embedding \alpha and attention 
        self.mu_0_alpha = nn.Parameter(torch.randn(1, args.num_topics, self.rho_size).to(device))
        self.logsigma_q_alpha = nn.Linear(self.rho_size, self.rho_size, bias=True)
        self.mu_q_alpha = nn.Linear(self.rho_size, self.rho_size, bias=True)
        self.num_heads = 10
        self.multi_attn = nn.MultiheadAttention(embed_dim=self.rho_size, num_heads=self.num_heads, batch_first=True)
        self.qkv_proj = nn.Linear(self.rho_size,3*self.rho_size)
        self.layer_norm = nn.LayerNorm(self.rho_size)
        
        self.cw = args.c_weight
        ## define the word embedding matrix \rho
        self.train_embeddings = args.train_embeddings
        if args.train_embeddings:
            self.rho = nn.Linear(args.rho_size, args.vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            rho.weight.data = embeddings
            self.rho = rho.weight.data.clone().float().to(device)
    
        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
                    nn.Linear(args.vocab_size+args.num_topics, args.t_hidden_size), 
                    nn.ReLU(inplace=False),
                    nn.Linear(args.t_hidden_size, args.t_hidden_size),
                    nn.ReLU(inplace=False),
                )
        self.mu_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(args.vocab_size, args.eta_hidden_size)
        self.q_eta = nn.LSTM(args.eta_hidden_size, args.eta_hidden_size, args.eta_nlayers, dropout=args.eta_dropout)
        self.mu_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)

        self.citation_loss = nn.BCELoss(reduction='sum')
        self.mat_cit = None
        self.bow_articles = None
        self.num_cum_articles = [0] * (self.num_times+1)
        self.dict_label_matric = [0] * self.num_times
        

    def update_citation_data(self, mat_cit, bow_articles):
        self.mat_cit = mat_cit
        self.bow_articles = bow_articles
        if self.mat_cit is None:
            self.num_cum_articles = [0] * (self.num_times+1)
            return

        for t in range(1, self.num_times):
            self.num_cum_articles[t] = self.num_cum_articles[t-1] + self.bow_articles[t-1].shape[0]
            mat_label = torch.zeros(self.bow_articles[t].shape[0], self.num_cum_articles[t]).to(device)
            for s in range(t):
                mat_label[:, self.num_cum_articles[s]:self.num_cum_articles[s+1]] = self.mat_cit[t][s]
            self.dict_label_matric[t] = mat_label

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        epsilon = torch.randn(mu.shape, device=device)
        return mu+epsilon*torch.exp(0.5 * logvar)

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def get_attention(self, predict=False):
        alpha = self.mu_0_alpha
        num_times = self.num_times + 1 if predict else self.num_times
        alphas = torch.empty((num_times, self.mu_0_alpha.shape[1], self.mu_0_alpha.shape[2]), device=device)
        alpha = self.layer_norm(alpha)
        alphas[0] = torch.squeeze(alpha)
        attentions = {}
        for t in range(1, num_times):
            qkv = self.qkv_proj(alpha)
            q_t, k_t, v_t = qkv.chunk(3, dim=-1)
            prev_alpha = alpha
            if t == 1:
                cul_k = k_t
                cul_v = v_t
            else:
                cul_k = torch.cat((cul_k, k_t), 1)
                cul_v = torch.cat((cul_v, v_t), 1) # 1(batch_size) * topic_num * emb_size

            alpha, attn_weights = self.multi_attn(q_t, cul_k, cul_v)
            attentions[t] = torch.squeeze(attn_weights)
            alpha = self.layer_norm(alpha+prev_alpha) # residual connection
            alphas[t] = torch.squeeze(alpha)
        return alphas, attentions

    def get_alpha(self):
        kl_alpha = []
        
        outputs, attentions = self.get_attention()
        alphas = torch.empty((self.num_times, self.num_topics, self.rho_size), device=device)
        q_mu_0 = self.mu_q_alpha(outputs[0])
        q_logsigma_0 = self.logsigma_q_alpha(outputs[0])
        alpha = self.reparameterize(q_mu_0, q_logsigma_0)
        alphas[0] = alpha
        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        p_logsigma_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        kl_t = self.get_kl(q_mu_0, q_logsigma_0, p_mu_0, p_logsigma_0)
        kl_alpha.append(kl_t)
        for t in range(1, self.num_times):
            q_mu_t = self.mu_q_alpha(outputs[t])
            q_logsigma_t = self.logsigma_q_alpha(outputs[t])
            alpha = self.reparameterize(q_mu_t, q_logsigma_t)
            alphas[t] = alpha
            p_mu_t = alphas[t-1]
            p_logsigma_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(device))
            kl_t = self.get_kl(q_mu_t, q_logsigma_t, p_mu_t, p_logsigma_t)
            kl_alpha.append(kl_t)
        
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, attentions, kl_alpha

    def get_eta(self, rnn_inp): ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.num_times, self.num_topics).to(device)
        kl_eta = []
        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_topics,).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics,).to(device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def get_theta(self, eta, bows, times=None): ## amortized inference
        """Returns the topic proportions.
        """
        if times is not None:
            eta_td = eta[times.type('torch.LongTensor')]
            inp = torch.cat([bows, eta_td], dim=1)
        else: # alredy speicfy time
            eta_td = eta.repeat((bows.shape[0],1))
            inp = torch.cat([bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(device))
        return theta, kl_theta

    def get_beta(self, alpha):
        """Returns the topic matrix beta of shape K x V
        """
        if self.train_embeddings:
            logit = self.rho(alpha.reshape(-1, self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, self.rho.permute(1, 0))
        logit = logit.reshape(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta 

    def get_nll(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = torch.log(loglik+TINY)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll

    def get_phi(self, theta):
        """paper proportion"""
        tmp = torch.transpose(theta, 0, 1)
        divisor = torch.sum(tmp, dim=1).unsqueeze(-1)
        phi = tmp / divisor # 行の和が1
        return phi

    def get_citation_loss_at_t(self, theta, attention, phi, mat_label, min_citation_count=4):
        """
        Args:
            theta: (num_article_at_t, num_topic)
            attention: (num_topic, num_topic*(t-1))
            phi: (num_topic, cum_num_article)
            mat_label: citation label (num_article_at_t, cum_num_article)
            min_citation_count: minimum count that each article cites other articles.
        Ret:
            loss: scaler
        """
        paper_to_prev_topics = torch.mm(theta, attention)
        atten_paper = torch.mm(paper_to_prev_topics, phi)
        
        sum_citation = torch.sum(mat_label, dim=1)
        idx_citation = torch.squeeze(torch.nonzero(sum_citation >= min_citation_count), dim=1)
        labels = mat_label[idx_citation]
        preds = atten_paper[idx_citation]
        loss = self.citation_loss(preds, labels)

        return loss, labels.shape[1]

    def get_citation_loss(self, atts, etas):
        """
            Args:
                atts: dict of attention. atts[t] indicates (num_topic, num_topic*(t-1))
                etas: 
            Ret:
                loss: scaler
        """
        kl_theta = 0.0
        phis = torch.zeros(self.num_times * self.num_topics, self.num_cum_articles[self.num_times-1]).to(device)
        dict_theta = {}
        dict_phi = {}
        for t in range(self.num_times):
            theta, kl_loss = self.get_theta(etas[t], self.bow_articles[t])
            dict_theta[t] = theta
            phi = self.get_phi(theta)
            dict_phi[t] = phi
            kl_theta += torch.sum(kl_loss)

        for t in range(1, self.num_times):
            phis[(t-1)*self.num_topics:t*self.num_topics, self.num_cum_articles[t-1]:self.num_cum_articles[t]] = dict_phi[t-1]
        
        total_loss = 0.0
        for t in range(1, self.num_times):
            att = atts[t] # (num_topic, (target_time-1) * num_topic) 
            cum_phi = phis[:t*self.num_topics, :self.num_cum_articles[t]].detach()
            label = self.dict_label_matric[t] # (self.bow_articles[t].shape[0], self.num_cum_articles[t])
            loss, sz = self.get_citation_loss_at_t(dict_theta[t], att, cum_phi, label)
            total_loss += loss

        return total_loss, kl_theta 

    def forward(self, bows, normalized_bows, times, rnn_inp, num_docs):
        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz
        alpha, alpha_attention, kl_alpha = self.get_alpha()
        eta, kl_eta = self.get_eta(rnn_inp)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times)
        kl_theta = kl_theta.sum() * coeff
        beta = self.get_beta(alpha)
        beta = beta[times.type('torch.LongTensor')]
        nll = self.get_nll(theta, beta, bows)
        nll = nll.sum() * coeff

        # citation loss
        if self.citation and self.cw > 0:
            citation_loss, kl_theta = self.get_citation_loss(alpha_attention, eta)
        else:
            citation_loss = torch.tensor(0.0, device=device)
        nelbo = nll  + kl_eta  + kl_theta + kl_alpha + citation_loss
        return nelbo, nll, kl_eta, kl_theta, kl_alpha, citation_loss

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))

class EvaluationModel(DSNTM):
    def __init__(self, args, embeddings, tokens, counts, times, rnn_inp, vocab, dict_citation=None, dict_bow=None):
        super().__init__(args, embeddings)
        self.eval()
        self.batch_size = args.eval_batch_size
        self.bow_norm = args.bow_norm
        self.tokens = tokens
        self.counts = counts
        self.times = times
        self.num_docs = len(tokens)
        self.rnn_inp = rnn_inp
        self.vocab = vocab
        self.dict_citation = dict_citation
        self.dict_bow = dict_bow
        self.args = args

    def update_bow_citation(self, citation, bow):
        self.dict_citation = citation
        self.dict_bow = bow
    
    def visualize_words(self, word_prop, num_top, is_string=False):
        top_words = list(word_prop.cpu().detach().numpy().argsort()[-num_top:][::-1])
        topic_words = [self.vocab[a] for a in top_words]
        if not is_string:
            return topic_words
        topics_words = []
        topics_words.append(' '.join(topic_words))  
        return topics_words

    def get_eta(self, rnn_inp):
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()
        etas = torch.zeros(self.num_times, self.num_topics).to(device)
        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(device)], dim=0)
        etas[0] = self.mu_q_eta(inp_0)
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            etas[t] = self.mu_q_eta(inp_t)
        return etas

    def get_theta(self, eta, bows):
        with torch.no_grad():
            inp = torch.cat([bows, eta], dim=1)
            q_theta = self.q_theta(inp)
            mu_theta = self.mu_q_theta(q_theta)
            theta = F.softmax(mu_theta, dim=-1)
            return theta

    def get_alpha(self, predict=False):
        num_times = self.num_times + 1 if predict else self.num_times
        outputs, _ = self.get_attention(predict)
        alphas = torch.empty((num_times, outputs.shape[1], self.rho_size), device=device)
        for t in range(num_times):
            alphas[t] = self.mu_q_alpha(outputs[t])
        return alphas

    def get_alpha_attention(self, selected_time, topic_idx=-1):
        _, attn_weights = self.get_attention()
        attn_weights = attn_weights[selected_time]
        if topic_idx >=0:
            attn_weights = attn_weights[topic_idx]
        attn_weights = torch.reshape(attn_weights, (-1, self.num_topics)) # (target topic, time t, source topic at t)
        return attn_weights

    def get_perplexity(self):
        """Returns document completion perplexity.
        """
        with torch.no_grad():
            alpha = self.get_alpha()
            indices = torch.split(torch.tensor(range(self.num_docs)), self.batch_size)
            eta = self.get_eta(self.rnn_inp)
            acc_loss = 0
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch, times_batch = data.get_batch(
                    self.tokens, self.counts, ind, self.vocab_size, self.emsize, temporal=True, times=self.times)
                sums = data_batch.sum(1).unsqueeze(1)
                if self.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                eta_td = eta[times_batch.type('torch.LongTensor')]
                theta = self.get_theta(eta_td, normalized_data_batch)
                alpha_td = alpha[times_batch.type('torch.LongTensor'), :, :]
                beta = self.get_beta(alpha_td)
                loglik = theta.unsqueeze(2) * beta
                loglik = loglik.sum(1)
                loglik = torch.log(loglik+TINY)
                nll = -loglik * data_batch
                nll = nll.sum(-1)
                loss = nll / sums.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_all = round(math.exp(cur_loss), 1)
            return ppl_all

    def get_topic_coherence(self, beta, dir_corpus, visualize=True):
        if len(beta.shape) != 2:
            print("shape size of beta must be 2")
            return
        topic_top_list = [0]*self.num_topics
        for topic_idx, k in enumerate(range(self.num_topics)):    
            topic_top_list[topic_idx] = self.visualize_words(beta[k, :], 15)
        mean_coherence_list = self._coherence(topic_top_list, dir_corpus)
        mean_coherence = np.mean(np.array(mean_coherence_list))
        dtype = [('words', list), ('coherence', float), ('index', int)]
        values = zip(topic_top_list, mean_coherence_list, range(self.num_topics))
        a = np.array(list(values), dtype=dtype)
        if visualize:
            for top_word, coherence, tdi in np.sort(a, order='coherence')[::-1]:
                print('Topic {}: coherence: {}, top words: {}'.format(tdi, round(coherence, 3), top_word[:10]))
            print("Coherence: {}".format(round(mean_coherence, 3)))
            print("==========================================================================")
        return mean_coherence

    def _coherence(self, topic_words_idx, dir_corpus):
        mean_coherence_list = compute_coherence(topic_words_idx, dir_corpus, topns=[10], window_size=0, verbose=False)
        return mean_coherence_list

    def get_topic_diversity(self, beta, num_tops=25, visualize=True):
        if len(beta.shape) != 2:
            print("shape size of beta must be 2")
            return
        list_w = np.zeros((self.num_topics, num_tops))
        for k in range(self.num_topics):
            gamma = beta[k, :]
            top_words = gamma.cpu().detach().numpy().argsort()[-num_tops:][::-1]
            list_w[k, :] = top_words
        list_w = np.reshape(list_w, (-1))
        list_w = list(list_w)
        n_unique = len(np.unique(list_w))
        diversity = n_unique / (self.num_topics * num_tops)
        if visualize:
            print("Diversity: {}".format(round(diversity, 3)))
        return diversity

    def get_c_loss(self, rnn_inp):
        with torch.no_grad():
            _, attns = self.get_attention()
            eta = self.get_eta(rnn_inp)
            loss, _ = self.get_citation_loss(attns, eta)
            return loss

    def visualize_one_topic(self, selected_time, topic_idx, count=20):
        """Visualizes topics and embeddings and word usage evolution.
        """
        self.eval()
        with torch.no_grad():
            alpha = self.get_alpha()
            beta = self.get_beta(alpha)
            gamma = beta[selected_time, topic_idx, :]
            topics_words = self.visualize_words(gamma, count, True)
            print('Time: {} .. Topic {} ===> {}'.format(selected_time, topic_idx, topics_words))
            return self.visualize_words(gamma, 20)

    def visualize_attention(self, target_time, topic_idx, include_time=None, include_topic=None, threshold=0.001, num_top=-1, visualize=False):
        attn_weight = self.get_alpha_attention(selected_time=target_time, topic_idx=topic_idx)
        sorted, indices = torch.sort(attn_weight.reshape(-1), descending=True)
        if not visualize:
            return sorted, indices
        
        topics = {}
        target_topics = self.visualize_one_topic(target_time, topic_idx, count=50)
        topics['target'] = target_topics
        if include_time is None:
            include_time = range(target_time)
        if include_topic is None:
            include_topic = range(self.num_topics)
        for i, (weight, topic_idx) in enumerate(zip(sorted, indices)):
            if weight < threshold:
                break
            if num_top == i:
                break
            tim, tix = self._topic_idx_to_time_topic(topic_idx)
            if tim in include_time and tix in include_topic:
                print("[time: {} & topic id: {}] {}".format(tim, tix, round(weight.item(), 3)))
                source_topics = self.visualize_one_topic(tim, tix)
                topics['source_{}'.format(i)] = source_topics

        return sorted, indices, topics
        

    def _topic_idx_to_time_topic(self, topic_idx):
        tim = torch.div(topic_idx, self.num_topics, rounding_mode='floor')
        tix = topic_idx % self.num_topics
        return tim, tix

    def topic_branching(self, target_time, topic_id, paper_ids_by_time, num_topic=5, check_paper=True):
        print("="*20)
        target_words = set(self.visualize_one_topic(target_time, topic_id))
        print()
        target_topic_id = target_time*self.num_topics + topic_id
        weights = torch.empty(self.num_topics*(self.num_times-target_time-1)).to(device)
        topics = torch.empty(self.num_topics*(self.num_times-target_time-1)).to(device)
        for t in range(target_time+1, self.num_times):
            topic_weights_to_paper, _ = self.check_topic_contribution_by_time(tt=t, is_sorted=False)
            for k in range(self.num_topics):
                sorted, indices = self.visualize_attention(target_time=t, topic_idx=k, threshold=0.0)
                idxes = (indices == target_topic_id).nonzero(as_tuple=True)
                weight = sorted[idxes[0]]
                weights[(t-target_time-1)*self.num_topics+k] = weight[0]*topic_weights_to_paper[k]
                topics[(t-target_time-1)*self.num_topics+k] = t*self.num_topics+k
        weights /= torch.sum(weights)
        sorted, indices = torch.sort(weights, descending=True)
        topics = topics[indices]
        target_paper_ids, _ = self.get_top_papers_by_topic(target_time=target_time, topic_id=topic_id, paper_ids_by_time=paper_ids_by_time, num_top=50)
        source_paper_ids = torch.empty((num_topic, 50), device=device)
        citing_paper_info = []
        for i, (s, t) in enumerate(zip(sorted[:num_topic], topics[:num_topic])):
            tim, tix = self._topic_idx_to_time_topic(int(t))
            source_words = set(self.visualize_one_topic(tim, tix))
            print("Weight: ", s.item())
            if len(target_words & source_words) > 0:
                print("Duplicate words: ", target_words & source_words)
            paper_ids, _ = self.get_top_papers_by_topic(target_time=tim.item(), topic_id=tix, paper_ids_by_time=paper_ids_by_time, num_top=50)
            citing_paper_info = citing_paper_info + [{
                "index": i,
                "paper_rank":pr,
                "paper_id": pid.item(),
            } for pr, pid in enumerate(paper_ids)]
            source_paper_ids[i] = paper_ids

        if check_paper:
            check_topic_branching(self.args, target_paper_ids, source_paper_ids, citing_paper_info)

    def topic_integrating(self, target_time, topic_id, paper_ids_by_time, num_topic=10, check_paper=True):
        _, indices, topics = self.visualize_attention(target_time=target_time, topic_idx=topic_id, num_top=num_topic, visualize=True)
        target_topics = set(topics['target'])
        target_paper_ids, _ = self.get_top_papers_by_topic(target_time=target_time, topic_id=topic_id, paper_ids_by_time=paper_ids_by_time, num_top=50)
        source_paper_ids = torch.empty((num_topic, 50), device=device)
        for i, topic_idx in enumerate(indices[:num_topic]):
            source_topics = set(topics['source_{}'.format(i)])
            if len(target_topics & source_topics) > 0:
                print("Duplicate words: ", target_topics & source_topics)
            else:
                print("No duplicate words")
            tim, tix = self._topic_idx_to_time_topic(topic_idx)
            paper_ids, _ = self.get_top_papers_by_topic(target_time=tim.item(), topic_id=tix, paper_ids_by_time=paper_ids_by_time, num_top=50)
            source_paper_ids[i] = paper_ids
        
        if check_paper:
            check_topic_integrating(self.args, target_paper_ids, source_paper_ids)

    def get_top_papers_by_topic(self, target_time, topic_id, paper_ids_by_time, num_top):
        etas = self.get_eta(self.rnn_inp)
        eta_td = etas[target_time].repeat((self.bow_articles[target_time].shape[0],1)) # 1時刻のみなのでrepeatするだけで良い
        theta = self.get_theta(eta_td, self.bow_articles[target_time])
        phi = self.get_phi(theta)
        papers = phi[topic_id]
        _, indices = torch.sort(papers, descending=True)
        paper_ids = torch.tensor(paper_ids_by_time[target_time]).to(device)
        
        return paper_ids[indices[:num_top]], indices[:num_top]

    def check_topic_contribution_by_time(self, tt, num_visualize=-1, is_sorted=True):
        """measure contribution rate of each topic to all papers at specific time"""
        with torch.no_grad():
            etas = self.get_eta(self.rnn_inp)
            eta_td = etas[tt].repeat((self.bow_articles[tt].shape[0],1))
            theta = self.get_theta(eta_td, self.bow_articles[tt])
            contribute_weight_to_article = torch.sum(theta, dim=0)
            if not is_sorted:
                contribute_weight_to_article /= torch.sum(contribute_weight_to_article)
                return contribute_weight_to_article, torch.zeros_like(contribute_weight_to_article).to(device)
            
            sorted, indices = torch.sort(contribute_weight_to_article, descending=True)
            sum_weighted = torch.sum(sorted)
            print("Time :", tt)
            for i, (topic_idx, v) in enumerate(zip(indices, sorted)):
                if i == num_visualize:
                    print("=====================")
                    break
                v /=sum_weighted
                print("topic {}: {}".format(topic_idx, round(v.item(), 3)))

        return sorted, indices

    def get_performance(self, visualize, cohrence_path):
        ppl = self.get_perplexity()
        beta = self.get_beta(self.get_alpha())
        avg_coherence = 0
        avg_diversity = 0
        for t in range(self.num_times):
            coherence = self.get_topic_coherence(beta[t], cohrence_path, visualize=visualize)
            diversity = self.get_topic_diversity(beta[t], visualize=visualize)
            avg_coherence += coherence
            avg_diversity += diversity
        avg_coherence /= self.num_times
        avg_diversity /= self.num_times
        return ppl, avg_coherence, avg_diversity

    def predict_topic(self, visualize=False):
        alphas = self.get_alpha(predict=True)
        
        latest_topic = torch.unsqueeze(alphas[self.num_times], 0)
        beta = self.get_beta(latest_topic)
        for k in range(self.num_topics):
            gamma = beta[0, k, :]
            topic_words = self.visualize_words(gamma, 20, visualize)
            print("{}th topic: {}".format(k, topic_words))
        
        return latest_topic

    def get_avg_kl_div(self, beta1, beta2, visualize=False):
        total_kl_div = 0.0
        for i in range(self.num_topics):
            b1 = beta1[i]
            min_dist = 1e9
            target_j = -1
            for j in range(self.num_topics):
                b2 = beta2[j]
                dist = F.kl_div(torch.log(b1+TINY), b2, reduction='sum')
                if min_dist > dist:
                    min_dist = dist
                    target_j = j

            if min_dist == 1e9:
                print("inf index: ", i, j)
            total_kl_div += min_dist
            if visualize:
                print()
                print("[predict topic: {}, actual topic: {}] KL Loss: {}".format(i, target_j, min_dist))
                top_words_predict = self.visualize_words(b1, 10, True)
                top_words_actual = self.visualize_words(b2, 10, True)
                print("predict topic at topic {}: {}".format(i, top_words_predict))
                print("actual topic at topic {}: {}".format(target_j, top_words_actual))
                
        avg_kl_div = total_kl_div / self.num_topics
        return avg_kl_div
