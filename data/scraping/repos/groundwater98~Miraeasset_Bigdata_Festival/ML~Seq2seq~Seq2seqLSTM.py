import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import math
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import openai

from Seq2seq.Train import train, evaluate, test
from Seq2seq.Dataset import CustomDataset, CustomInferenceDataset, collate_fn


base = os.path.abspath(__file__)
base = base.split('/')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        '''
        input_dim: 입력 차원, 일반적으로 단어 집합의 크기
        emb_dim: 임베딩 차원
        hid_dim: LSTM의 hidden state의 차원
        n_layers: LSTM 층의 개수
        dropout: dropout 비율
        '''
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim) 
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout) # 과적합 방지

    def forward(self, src): 
        '''
        forward 메서드는 모델이 학습 및 예측 시 호출하는 메서드
        src: 입력 데이터
        '''
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        # hidden 차원의 3배 크기를 가진 입력을 받아 hidden 차원 크기의 출력을 만드는 linear layer.
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        # hidden 차원의 입력을 받아 크기 1의 출력을 만드는 linear layer: enery에 대한 weights 생성
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        '''
        hidden: Decoder의 hidden state
        encoder_outputs: Incoder의 출력
        '''
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0] # 입력 시퀀스의 길이
        
        # hidden: [2, 64, 512] => [64, 2, 512]
        hidden = hidden.permute(1, 0, 2)
    #    print("hidden shape: {}".format(hidden.shape))

        # [64, 2, 512] => [64, 7, 1024] via repeating
        # [32, 2, 512] => [32, 2, 1, 512] => [32, 2, 7, 512] = [batch_size, seq_len, src_len, hidden_dim]
        hidden_repeated = hidden.unsqueeze(2).repeat(1, 1, src_len, 1)
    #    print("hidden_repeated shape: {}".format(hidden_repeated.shape))
        
        # reshape 결과: batch_size, src_len, seq_len*hid_dim
        hidden_repeated = hidden_repeated.reshape(batch_size, src_len, -1)
    #    print("hidden_repeated shape: {}".format(hidden_repeated.shape))

        # encoder_outputs: [7, 64, 512] => [64, 7, 512]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
    #    print("encoder_outputs shape: {}".format(encoder_outputs.shape))

        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2))) # hidden_repeated와 encoder_outputs가 연결되어 attention의 에너지를 계산한다.
    #    print("energy shape: {}".format(energy.shape))

        attention = self.v(energy).squeeze(2) # energy tensor을 입력으로 받아 attention 점수를 계산한다.
    #    print("attention shape: {}".format(attention.shape))
    #    print("attention: {}".format(attention))

        '''
        attention mechanism에서는 각 입력 시퀀스 위치에 대한 attention 점수를 계산하는데, 이 점수는 확률 분포 형태로 변환되어야 한다.
        softmax activation function을 사옹하면 attention 점수를 확률 값으로 변환할 수 있다. 여기서 반환된 확률 분포는 Decoder가 Encoder의 어떤 부분에 주의를 기울여야 하는지 나타낸다.

        * 오해한 부분 => Attention mechanism은 입력 시퀀스에 대한 Attention 점수를 자동으로 학습하기 때문에 어느 부분에 주의를 기울일지 사용자가 직접 지정하는 것은 일반적이지 않다.
        '''
        return nn.functional.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim # 출력 차원
        self.attention = attention # Attention 객체
        
        # output_dim 크기의 단어 집합에서 emb_dim 크기의 embedding vector 추출
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # LSTM => 입력은 emb_dim + hid_dim 크기이고 출력은 hid_dim 크기
        self.rnn = nn.LSTM((emb_dim + hid_dim), hid_dim, n_layers, dropout=dropout)
        # 최종 출력을 위한 Linear layer. 입력은 hid_dim * 2 크기, 출력은 output_dim 크기
        self.fc_out = nn.Linear((hid_dim * 2), output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input 단어의 차원을 확장
        input = input.unsqueeze(0)
        # 단어를 embedding vector로 변환 후 dropout 적용
        embedded = self.dropout(self.embedding(input))
        
        # Attention 가중치 계산 => 어느 부분에 집중할 것인지 자동으로 계산
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Attention 가중치 'a'와 Encoder 출력을 곱하여 주목해야 할 정보를 가중 평균으로 추출
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        
        # embedding bector와 가중 평균 vector을 결합하여 rnn 입력을 생성
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        # output과 weighted를 결합하여 최종 예측 생성
        prediction = self.fc_out(torch.cat((output, weighted), dim=1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # trg는 목표 문장 => [seq_len, batch_size]
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        
        # Decoder의 출력 차원 설정 => 목표 문장의 단어장 크기와 같다, 단어장 크기란 목표 언어에 있는 고유한 단어의 개수
        trg_vocab_size = self.decoder.output_dim
        
        # Decoder에서 예측된 출력을 저장할 tensor을 초기화
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # Encoder를 통해 src를 인코딩하고, 초기 hidden state와 cell state를 얻는다.
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # 첫 번째 Decoder 입력은 <sos> 토큰
        input = trg[0,:]
        
        # trg_len만큼 반복하며, Decoder을 time step마다 순차적으로 실행한다.
        for t in range(1, trg_len):
            '''
            Teacher Forcing: 학습 동안 Decoder의 예측을 다음 입력으로 사용하는 대신 실제 목표값을 다음 입력으로 사용하는 기법
            1. 학습 속도: 모델이 잘못된 예측을 하더라도 올바른 경로로 빠르게 되돌아올 수 있도록 한다
            2. 안정성: 초기에는 모델의 예측이 부정확할 가능성이 높기 때문에, 잘못된 예측이 연쇄적으로 발생하는 것을 방지하고 학습을 안정화 시킨다.
            '''
            # Decoder 결과
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            # Prediction 저장
            outputs[t] = output
            # torch.rand(1)은 0과 1사이의 무작위 값을 반환하고 .item()은 그 값을 스칼라 값으로 추출한다.
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            # 가장 가능성 있는 다음 단어 예측 => argmax(1) 함수는 가장 높은 확률의 index 즉, 가장 가능성 있는 다음 단어의 index를 반환한다.
            top1 = output.argmax(1)
            # 다음 입력 결정 (실제 목표값 or 예측값)
            input = trg[t] if teacher_force else top1
        
        return outputs


def generate_query(model, sentence, dataset, max_len=50):
    '''
    model: 학습된 모델
    sentence: 변환하거나 번역하고자 하는 입력 문장
    dataset: CustomInferenceDataset 클래스의 인스턴스
    max_len: 생성할 출력 문장의 최대 길이
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # 평가 모드
    with torch.no_grad(): # Gradient 계산을 비활성화
        tokens = dataset.tokenize_and_encode(sentence) 
        src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device) # 입력 문장을 텐서로 변환하고 텐서를 모델에 맞게 변형
        src_len = torch.LongTensor([len(tokens)])
        
        trg_indexes = [dataset.word2idx['<sos>']]  # Assuming you have <sos> token for start
        for _ in range(max_len): # 모델은 현재까지의 출력 토큰을 사용하여 다음 토큰을 예측한다.
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)
            output = model(src_tensor, trg_tensor)
            pred_token = output.argmax(2)[-1, :].item() # argmax(2)를 사용하여 가장 확률이 높은 토큰의 인덱스를 얻는다.
            trg_indexes.append(pred_token)
            if pred_token == dataset.word2idx['<eos>']:  # Assuming you have <eos> token for end
                break
        trg_tokens = trg_indexes[1:]
        
        # 출력 문장 끝에 <eos> 토큰이 있다면 제거
        if trg_tokens[-1] == dataset.word2idx['<eos>']:
            trg_tokens = trg_tokens[:-1]
        
        query = dataset.decode(trg_tokens)
        return query


def correct_sql_by_gpt3(user_input, input_query):
    # GPT-3로부터 SQL 쿼리 교정 요청
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 이제부터 Seq2seq LSTM에서 나온 SQL Query를 교정해주는 역할이야. 답변은 다 생략하고 무조건 수정된 SQL Query만 출력해줘."},
            {"role": "user", "content": "삼성전자의 최신 뉴스를 요약해줘."},
            {"role": "assistant", "content": "SELECT title, link, description, pubdate FROM 삼성전자 ORDER BY pubdate DESC LIMIT 1"},
            {"role": "user", "content": user_input},
            {"role": "user", "content": f"사용자의 입력인 '{user_input}'에 기반해서 Seq2seq LSTM Model이 만들어낸 SQL Query인 '{input_query}'을 올바른 형태로 수정해줘."}
            
        ],
    )
    corrected_query = response.choices[0]['message']['content'].strip()
    return corrected_query


def get_sql(question):
    global base

    tr_path = "/".join(base[:-3]+["data","train.csv"])
    val_path = "/".join(base[:-3]+["data","val.csv"])
    test_path = "/".join(base[:-3]+["data","test.csv"])
    model_path = "/".join(base[:-2]+["Seq2seq", f"{datetime.now().date()}_seq2seq-model.pt"])
    # training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and process the data
    train_dataset = CustomDataset(tr_path)
    val_dataset = CustomDataset(val_path)
    test_dataset = CustomDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


    # Hyperparameters
    INPUT_DIM = len(train_dataset.word2idx) + 1  # +1 for padding index
    OUTPUT_DIM = INPUT_DIM
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # Create the model instances
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    attn = Attention(HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    #optimizer = optim.SGD(model.parameters(),lr=0.01)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx) # loss function

    N_EPOCHS = 10
    CLIP = 1

    # matplotlib으로 시각화를 하기 위한 코드
    train_losses = []
    valid_losses = []

    # PPL->Perplexity: 낮을 수록 좋다.
    if not os.path.exists(model_path):
        for epoch in range(N_EPOCHS):
            train_loss = train(model, train_loader, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, val_loader, criterion)

            # Save the losses for plotting
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            #print(f'Epoch: {epoch + 1:02}')
            #print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


        test_loss = test(model, test_loader, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
        
        torch.save(model.state_dict(), model_path)

    # generate sql
    openai.api_key = "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"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inference를 위한 Dataset => trained model로 새로운 입력 문장을 생성할 때 이 Dataset을 사용하여 입력 문장을 적합한 형태로 변환하고, 모델의 출력을 다시 사람이 이해할 수 있는 형태로 변환한다.

    # Load vocabulary from training dataset
    train_dataset = CustomDataset(tr_path)
    vocab = train_dataset.word2idx
    inference_dataset = CustomInferenceDataset(vocab)

    # Load the trained model
    INPUT_DIM = len(vocab) + 1
    OUTPUT_DIM = INPUT_DIM
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5


    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    attn = Attention(HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    
    model.load_state_dict(torch.load(model_path))

    # Get user input and generate SQL query
    # sql 여러 개 생성해서 그 중에서 괜찮은 거 뽑는 식으로 진행해도 될 듯.
    for _ in range(1):
        user_input = question
        sql_query = generate_query(model, user_input, inference_dataset)
    #   print(f"Generated SQL Query: {sql_query}")
        corrected_sql_query = correct_sql_by_gpt3(user_input, sql_query)
        print(f"final SQL Query: {corrected_sql_query}")
    return corrected_sql_query