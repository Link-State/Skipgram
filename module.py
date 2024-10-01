import os
import time
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from model import dotModel

FREQ_THRESHOLD = 30     # 기준치
WIN_SZ = 2              # 윈도우 크기
VOCAB_SZ = 0            # 사전 크기
VOCAB = dict()          # 사전
ROULETTE = list()       # 룰렛
EMBEDDING_DIM = 100     # 임베딩 크기
BATCH_SIZE = 32         # 배치크기
EPOCH = 2               # 반복 횟수
TOTAL_DATA = 1049       # 훈련예제 데이터 갯수
TRAIN_DATA = 1049       # 훈련 데이터
TEST_DATA = ((TOTAL_DATA // 10) * 8) + 1    # 테스트 데이터

NO = '<NO>'
ENCODING = "utf-8"
CORPUS_DIR = "./corpus/"
VOCAB_DIR = "./vocabulary.txt"
TRAINING_FOLDER = "./training"
TRAINING_DIR = TRAINING_FOLDER + "/training_"
MODEL_DIR = TRAINING_FOLDER + "/wordvec_model.pth"
MODEL_IDX_DIR = TRAINING_FOLDER + "/wordvec_model_idx.txt"
EUCLID = 0
COSINE = 1


# 로그 출력
def log(msg, out=True, dir="./log.txt") :
    f = open(dir, "a", encoding=ENCODING)
    f.write(str(msg) + "\n")
    f.close()
    if out :
        print(msg)
    return

# 단어 정제
def cleaning_word(oword) :
    cleaned_oword = oword.replace(",", "")
    cleaned_oword = cleaned_oword.replace("$", "")
    cleaned_oword = cleaned_oword.replace(".", "")
    if cleaned_oword.isnumeric() :
        return NO
    word = cleaned_oword

    word_leng = len(word)
    if word_leng == 0 :
        return NO
    i = 0
    if word_leng > 1 and not word[0].isalpha() :
        i = 1
        if word_leng > 2 and not word[1].isalpha() :
            i = 2
    
    j = word_leng - 1
    if j >= i and j > 0 and not word[j].isalpha() :
        j = j - 1
        if j >= i and j > 0 and not word[j].isalpha() :
            j = j - 1
    if j < i :
        return NO
    
    word = word[i:j + 1]
    if len(word) == 0 :
        return NO

    if word.isalpha() :
        return word
    else :
        return NO


# 말뭉치 사전화
def readCorpus() :
    global VOCAB_SZ

    voc_tmp = dict()
    listdir = os.listdir(CORPUS_DIR)

    if len(listdir) <= 0 :
        return False

    for f in listdir :
        fp = open(CORPUS_DIR + f, "r")
        for line in fp :
            words = line.split()
            for word in words :
                key = cleaning_word(word.strip())
                if key != NO :
                    if key in voc_tmp :
                        voc_tmp[key][1] += 1
                    else :
                        voc_tmp[key] = [0, 1]
        fp.close()

    if len(voc_tmp) <= 0 :
        return False
    
    index = 0
    f_vocab = open(VOCAB_DIR, "w", encoding=ENCODING)
    keys = voc_tmp.keys()
    for key in keys :
        if voc_tmp[key][1] > FREQ_THRESHOLD :
            id = index
            freq = voc_tmp[key][1]
            f_vocab.write(key + "\t" + str(id) + "\t" + str(freq) + "\n")
            index += 1
            VOCAB_SZ = VOCAB_SZ + 1
    f_vocab.close()

    if index <= 0 :
        return False

    return True


# 사전 불러오기
def getVocab() :
    global VOCAB, VOCAB_SZ

    fp_v = open(VOCAB_DIR, "r", encoding=ENCODING)
    VOCAB = dict()

    while True :
        line = fp_v.readline()

        if len(line) == 0 :
            break

        line_splitted = line.split()
        word = line_splitted[0]
        id = int(line_splitted[1])
        freq = int(line_splitted[2])
        VOCAB[word] = [id, freq]
    fp_v.close()

    VOCAB_SZ = len(VOCAB)
    return VOCAB


# 해당 문장을 idx 리스트로 변환
def getSentenceIdx(line="") :
    sentence_idx = list()
    sentence = line.split()

    for _ in range(WIN_SZ) :
        sentence_idx.append(VOCAB_SZ)
    
    for oword in sentence :
        word = cleaning_word(oword.strip())
        idx = VOCAB_SZ
        if word != NO and word in VOCAB :
            idx_freq = VOCAB[word]
            idx = idx_freq[0]
        sentence_idx.append(idx)
    
    for _ in range(WIN_SZ) :
        sentence_idx.append(VOCAB_SZ)
    return sentence_idx


# Negative 예제 생성
def getNegativeSample(center_idx) :
    global ROULETTE
    # 룰렛 휠 생성
    if len(ROULETTE) <= 0 :
        tq = 1
        ROULETTE = list()
        for word in VOCAB :
            start = tq
            end = tq + int((VOCAB[word][1])**(3/4))
            ROULETTE.append([VOCAB[word][0], start, end])
            tq = end + 1
    
    # 뽑기
    r = random.randint(1, ROULETTE[-1][-1])
    left = 0
    right = len(ROULETTE)-1
    while True :
        mid = (left + right) // 2
        if ROULETTE[mid][1] > r :
            right = mid - 1
        elif ROULETTE[mid][2] < r :
            left = mid + 1
        else :
            if ROULETTE[mid][0] == center_idx :
                r = random.randint(1, ROULETTE[-1][-1])
                left = 0
                right = len(ROULETTE)-1
            else :
                break
    return ROULETTE[mid][0]


# 예제 생성
def generateSampling(sentence_idx=[], addr="./training.txt") :
    center_word = None
    out_word = None
    train_examples = list()

    training = open(addr, "a", encoding=ENCODING)

    for i in range(WIN_SZ, len(sentence_idx) - WIN_SZ) :
        center_word = sentence_idx[i]
        if center_word != VOCAB_SZ :
            
            for j in range(WIN_SZ) :
                out_word = sentence_idx[i-j-1]
                if out_word != VOCAB_SZ :
                    # train_examples.append([center_word, out_word, 1])
                    training.write(str(center_word) + "\t" + str(out_word) + "\t1\n")
                    neg_sampled_word = getNegativeSample(i)
                    training.write(str(center_word) + "\t" + str(neg_sampled_word) + "\t0\n")
                    # train_examples.append([center_word, neg_sampled_word, 0])
            
            for j in range(WIN_SZ) :
                out_word = sentence_idx[i+j+1]
                if out_word != VOCAB_SZ :
                    # train_examples.append([center_word, out_word, 1])
                    training.write(str(center_word) + "\t" + str(out_word) + "\t1\n")
                    neg_sampled_word = getNegativeSample(i)
                    training.write(str(center_word) + "\t" + str(neg_sampled_word) + "\t0\n")
                    # train_examples.append([center_word, neg_sampled_word, 0])
    training.close()
    return train_examples


# 룰렛 원소 모두 삭제 (메모리 확보용)
def clearRoulette() :
    global ROULETTE
    ROULETTE = list()
    return


# 파일로 저장된 훈련 예제를 텐서로 변환
def getTensorData(num) :
    path = TRAINING_DIR + str(num) + ".txt"
    train_x = list()
    train_y = list()

    f = open(path, "r", encoding=ENCODING)

    while True :
        line = f.readline()

        if len(line) == 0 :
            break

        line_splitted = line.split()
        wc = int(line_splitted[0])
        wo = int(line_splitted[1])
        label = int(line_splitted[2])

        w = list()
        w.append(wc)
        w.append(wo)

        train_x.append(w)
        train_y.append(label)

    f.close()

    data = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    return data


# 훈련 시작
def run() :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log("device : " + str(device))

    model = dotModel(voc_sz=VOCAB_SZ, d_emb=EMBEDDING_DIM)
    train_start = 1
    total_cal_time = 0
    if os.path.exists(MODEL_DIR) and os.path.exists(MODEL_IDX_DIR) :
        f_idx = open(MODEL_IDX_DIR, "r")
        line = f_idx.readline().split()
        train_start = int(line[0]) + 1
        total_cal_time = float(line[1])
        f_idx.close()
        model.load_state_dict(torch.load(MODEL_DIR))

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.9e-5)
    criterion = nn.BCELoss()

    for idx in range(train_start, TRAIN_DATA + 1) :
        log(str(idx) + "/" + str(TRAIN_DATA))
        data = getTensorData(idx)
        dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=BATCH_SIZE, drop_last=True)
        num_batches = len(dataloader)

        for i in range(EPOCH) :
            startT = time.perf_counter()
            total_loss = 0.0
            model.train()
            for j, batch in enumerate(dataloader) :
                batch = tuple(t.to(device) for t in batch)
                b_train_X, b_train_Y = batch
                optimizer.zero_grad()
                model.zero_grad()
                pred = model(b_train_X)
                pred = torch.reshape(pred, (BATCH_SIZE, 1))
                b_train_Y = torch.reshape(b_train_Y.float(), (len(b_train_Y), 1))

                loss = criterion(pred, b_train_Y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                if j % 2000 == 0 :
                    log("number of batches done = " + str(j))
            avg_loss_per_batch = total_loss / num_batches
            log("epoch = " + str(i) + " has finished. average loss per example = " + str(avg_loss_per_batch))

            endT = time.perf_counter()
            total_cal_time += endT - startT
            loop_count = ((i + 1) + ((idx - 1) * EPOCH))
            avg_cal_time = total_cal_time / loop_count
            remain_loop = (TRAIN_DATA * EPOCH) - loop_count
            log("est. time : " + str(int(avg_cal_time * remain_loop)) + "s")
        
        torch.save(model.state_dict(), MODEL_DIR)
        f_idx = open(MODEL_IDX_DIR, "w")
        f_idx.write(str(idx) + "\t" + str(total_cal_time))
        f_idx.close()
    return model


# 유사 단어 출력
def getSimilarWord(model=dotModel(voc_sz=VOCAB_SZ, d_emb=EMBEDDING_DIM), word="", count=5, method=COSINE, detail=False) :
    result = list()

    if word not in VOCAB :
        return []
    
    weights = model.embedding2.weight.data
    widx = VOCAB[word][0]
    word2vec = weights[widx]
    listed_vocab = sorted(VOCAB, key= lambda x:VOCAB[x][0])

    if method == EUCLID :
        # (1) 유클리드 내적
        dists = torch.sub(weights, word2vec).float()
        norms = torch.linalg.vector_norm(x=dists, dim=1)
        listed_norms = enumerate(norms)
        sorted_norms = sorted(listed_norms, key = lambda x:x[1].item())

        for _ in range(1, count+1) :
            value = ""
            if detail :
                distance = sorted_norms[_][1].item()
                value = " (" + f"{distance:.9f}" + ")"
            result.append(listed_vocab[sorted_norms[_][0]] + value)
        
    else :
        # (2) 코사인 유사도
        inner = weights.inner(word2vec)
        em_norm = torch.linalg.vector_norm(x=weights, dim=1)
        wv_norm = torch.linalg.vector_norm(x=word2vec)
        norm = torch.mul(em_norm, wv_norm)
        cos_sim = torch.div(inner, norm)
        listed_sim = enumerate(cos_sim)
        sorted_sim = sorted(listed_sim, key = lambda x:x[1].item(), reverse=True)

        for _ in range(1, count+1) :
            value = ""
            if detail :
                similarity = sorted_sim[_][1].item()
                value = " (" + f"{similarity:.9f}" + ")"
            result.append(listed_vocab[sorted_sim[_][0]] + value)

    return result
