# Python 3.12.6 (venv)
# pytorch 2.4.1 + cuda 12.4

import os
import time
from datetime import datetime
from module import *

def main() :
    log("\n------------------------" + str(datetime.now()) + "------------------------\n", False)
    if not os.path.exists(CORPUS_DIR) :
        os.makedirs(CORPUS_DIR, exist_ok=True)

    if not os.path.exists(TRAINING_FOLDER) :
        os.makedirs(TRAINING_FOLDER, exist_ok=True)

    # 사전 파일 없을 경우 corpus에서 사전 생성
    if not os.path.exists(VOCAB_DIR) :
        log("reading corpus...")
        start_time = time.perf_counter()
        if not readCorpus() :
            log("no data!")
            return
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log("create vocabulary.txt (" + f"{elapse:.2f}" + "s)\n")

    # 사전 불러오기
    log("loading vocabulary...")
    start_time = time.perf_counter()
    if len(getVocab()) <= 0 :
        log("no data!")
        return
    end_time = time.perf_counter()
    elapse = end_time - start_time
    log("loaded vocabulary (" + f"{elapse:.2f}" + "s)\n")

    # 훈련 예제 생성
    if not os.path.exists(TRAINING_DIR + "1.txt") :
        log("generating train example...")
        start_time = time.perf_counter()
        listdir = os.listdir(CORPUS_DIR)
        count = 0
        # corpus 파일마다 훈련 예제 생성
        for f in listdir :
            fp = open(CORPUS_DIR + f, "r")
            count += 1
            addr = TRAINING_DIR + str(count) + ".txt"
            for line in fp :
                sentence_idx = getSentenceIdx(line)
                generateSampling(sentence_idx, addr)
            fp.close()
        clearRoulette()
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log("generated train example (" + f"{elapse:.2f}" + "s)\n")
    
    # 훈련 시작
    log("start training")
    start_time = time.perf_counter()
    model = run()
    end_time = time.perf_counter()
    elapse = end_time - start_time
    log("end training (" + f"{elapse:.2f}" + "s)\n")

    # 테스트
    log("start test")
    log(" >> word, count, distance:Euclid=0|Cosine=1, detail:false=0|true=1\n")
    while True :
        test = input(" >> ")
        log(" >> " + test, out=False)
        if test == "" :
            break

        test = test.split(",")
        word = ""
        count = 5
        method = COSINE
        detail = False
        if len(test) == 4 :
            word = test[0].strip().lower()
            count = int(test[1].strip())
            method = int(test[2].strip())
            detail = bool(int(test[3].strip()))
        elif len(test) == 3 :
            word = test[0].strip().lower()
            count = int(test[1].strip())
            method = int(test[2].strip())
        elif len(test) == 2 :
            word = test[0].strip().lower()
            count = int(test[1].strip())
        else :
            word = test[0].strip().lower()

        start_time = time.perf_counter()
        testResult = getSimilarWord(model=model, word=word, count=count, method=method, detail=detail)
        end_time = time.perf_counter()
        elapse = end_time - start_time

        spliter = "\n" + (" " * (len(word) + 10))
        log("    " + word + "  ->  " + spliter.join(testResult) + "\n (" + f"{elapse:.2f}" + "s)\n")
    return

if __name__ == "__main__" :
    main()
