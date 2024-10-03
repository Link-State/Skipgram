# Skipgram 및 Negative Sampling을 이용한 Word Vector 개발
### [2024 2학기 자연어처리 과제1]

### 개발 기간
> 2024.09.09 ~ 2024.09.29

### 개발 환경
> Python 3.12.6 (venv)<br>
> Pytorch 2.4.1 + CUDA 12.4<br>
> GTX1060 6GB<br>

### 설명
+ 동기
    + 자연어처리 수업 과제
+ 기획
    + Skipgram 방식을 이용하여 word vector를 개발한다.
    + 개발한 word vector로 단어 간 유사도를 계산하여 가장 유사한 단어를 출력한다.
    + 두 벡터 간 유사도 계산은 유클리드 내적과 코사인 유사도 두 가지 방법을 사용한다.
    + 임베딩 크기 : 100
    + 배치 사이즈 : 32
    + Epoch : 2

#### 시작 Loss
<img width="429" alt="init_loss" src="https://github.com/user-attachments/assets/c8f975ba-e642-49b3-a11a-4850fc8b5fe0">

#### 최종 Loss
<img width="431" alt="final_loss" src="https://github.com/user-attachments/assets/feed15fa-24ae-46b1-9795-7b6563faf459">

#### 유사 단어 출력 결과 (유클리드 내적)
<img width="262" alt="1049_others_0" src="https://github.com/user-attachments/assets/d1c0669e-0987-4663-a879-31ce741eef80">

#### 유사 단어 출력 결과 (코사인 유사도)
<img width="296" alt="1049_others_1" src="https://github.com/user-attachments/assets/55cef629-e0c1-4ca9-b435-ec72b32d7ef3">
