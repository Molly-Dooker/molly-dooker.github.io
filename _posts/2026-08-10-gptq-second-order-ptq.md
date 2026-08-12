---
title: "GPTQ"
date: 2026-08-10 10:58:14 +0900
categories: [DeepLearning, Quantization]
tags: [quantization]
description: "GPTQ가 OBQ의 greedy quantization을 고정 순서, Cholesky factor, lazy batch-update로 재구성해 175B LLM까지 확장한 과정을 설명합니다."
math: true
render_with_liquid: false
---

## 문서 범위

- 핵심: 레이어별 재구성, OBQ 갱신, 고정된 열 순서, Cholesky 재구성, lazy batch-update
- 논문 기준: Elias Frantar et al., [GPTQ](https://arxiv.org/pdf/2210.17323) arXiv v2·ICLR 2023
- 구현 대조: IST-DASLab의 [공식 GPTQ repository](https://github.com/IST-DASLab/gptq)
- 선행 글: [Optimal Brain Compression](/posts/optimal-brain-compression/)
- 배경 자료: [OBC 논문](https://arxiv.org/pdf/2208.11580)의 OBQ
- 확인일: 2026-08-11

이 글에서 별다른 수식어 없이 “GPTQ”라고 하면 논문 Algorithm 1의 fixed-order weight-only
quantization을 가리킨다. `act-order`, `true-sequential` 등 실제 구현에 추가된 변형은
다루지 않고, 논문이 제안한 본질적인 알고리즘과 수식에 집중한다.

## 동기와 목표

GPTQ의 목표는 각 레이어의 가중치를 quantize하면서 원래 레이어와 압축한 레이어의 출력
재구성 오차를 최소화하는 것이다. 단순히 가장 가까운 grid point로 반올림하는 데서 끝나지
않고, 한 열에서 생긴 quantization 오차를 second-order 정보로 나머지 열에 보정한다.

출발점인 Optimal Brain Quantization(OBQ)은 ResNet-50의 2,500만 매개변수를 처리하는 데도
GPU 약 한 시간이 걸린다. 복잡도가 $O(d_{\mathrm{row}}d_{\mathrm{col}}^3)$이므로 수십억 개에서
수천억 개의 매개변수를 가진 언어 모델에는 적용하기 어렵다.

GPTQ는 greedy 순서를 고정된 열 순서로 바꾸고, 반복할 역헤시안 갱신을 Cholesky factor에
미리 담으며, 작은 갱신을 블록 행렬 곱으로 모은다. 이 세 변경으로 계산량을
$O(\max\{d_{\mathrm{row}}d_{\mathrm{col}}^2,d_{\mathrm{col}}^3\})$으로 낮춘다. 논문은 이
구조로 OPT-175B를 단일 NVIDIA A100에서 약 4.2시간에 quantize했다.

이 글은 논문 Algorithm 1의 수식과 처리 흐름을 기준으로 한다.

### Optimal Brain family에서의 위치

Optimal Brain Damage(OBD), Optimal Brain Surgeon(OBS), Optimal Brain Compression(OBC),
GPTQ는 모두 compression으로 생기는 오차를 second-order 정보로 줄이는 계보에 속한다.
Hessian을 어느 수준까지 사용하는지, pruning과 quantization 중 무엇을 다루는지,
계산을 어느 규모까지 확장하는지가 서로 다르다.

| Method | 대상 | 핵심 아이디어 |
| --- | --- | --- |
| OBD | pruning | diagonal Hessian으로 saliency 근사 |
| OBS | pruning | 전체 역헤시안으로 제거 오차 보정 |
| OBC/OBQ | pruning/quantization | 레이어별 exact greedy OBS |
| GPTQ | 가중치 quantization | OBQ를 수십억 매개변수 규모로 재구성 |

GPTQ의 새로움은 second-order 보정 그 자체에 있지 않다. 큰 레이어에서 이득이 작아지는
OBQ의 greedy 순서 탐색을 덜어내고, 같은 보정량을 GPU에 잘 맞고 수치적으로 안정적인
형태로 계산한 것이 핵심이다.

### 레이어별 재구성

Linear 레이어의 원래 가중치와 quantized 가중치를 각각
$\mathbf{W},\widehat{\mathbf{W}}\in\mathbb{R}^{d_{\mathrm{row}}\times d_{\mathrm{col}}}$,
calibration 입력을 $\mathbf{X}\in\mathbb{R}^{d_{\mathrm{col}}\times N}$이라 하자.
GPTQ가 직접 최소화하는 값은 end-to-end language-model loss가 아니라 다음 레이어별
재구성 오차다.

$$
\widehat{\mathbf{W}}^{*}
=
\arg\min_{\widehat{\mathbf{W}}\in\mathcal{Q}}
\left\lVert
\mathbf{W}\mathbf{X}-\widehat{\mathbf{W}}\mathbf{X}
\right\rVert_F^2,
\tag{1}
$$

여기서 $\mathcal{Q}$는 선택한 bit-width와 scale, zero-point로 정의한 quantization grid다.
Calibration sample에서 원래 레이어와 quantized 레이어의 출력이 최대한 비슷해지도록 가중치를
정하는 셈이다. 논문은 알고리즘을 시작하기 전에 grid가 정해져 있고, 아직 고정하지 않은
가중치는 보정 과정에서 자유롭게 움직일 수 있다고 가정한다. 아래 표에서는
$d_r=d_{\mathrm{row}}$, $d_c=d_{\mathrm{col}}$로 줄여 쓴다.

| 기호 | shape | 의미 |
| --- | ---: | --- |
| $\mathbf{W}$ | $d_r\times d_c$ | 원본 가중치 |
| $\widehat{\mathbf{W}},\mathbf{Q}$ | $d_r\times d_c$ | 양자화 결과 |
| $\mathbf{X}$ | $d_c\times N$ | calibration 입력 activation |
| $\mathbf{H}$ | $d_c\times d_c$ | $2\mathbf{X}\mathbf{X}^{T}$ |
| $F$ | 가변 | 아직 quantize하지 않은 열 index 집합 |

Frobenius norm은 행별 제곱 $L_2$ norm의 합이므로 식 (1)은 다음처럼 정확히 나뉜다.

$$
\left\lVert
\mathbf{W}\mathbf{X}-\widehat{\mathbf{W}}\mathbf{X}
\right\rVert_F^2
=
\sum_{r=1}^{d_{\mathrm{row}}}
\left\lVert
\left(\mathbf{W}_{r,:}-\widehat{\mathbf{W}}_{r,:}\right)\mathbf{X}
\right\rVert_2^2.
\tag{2}
$$

따라서 가중치 행은 서로 독립적으로 최적화할 수 있다. 모든 행이 같은 입력
$\mathbf{X}$를 사용하므로 $d_{\mathrm{col}}\times d_{\mathrm{col}}$ Hessian도 공유한다.
이 구조가 OBQ와 GPTQ 모두의 출발점이다.

### 왜 RTN만으로는 부족한가

Round-to-Nearest(RTN)는 각 가중치를 독립적으로 가장 가까운 grid point로 옮긴다. 하지만 한
가중치의 rounding 오차가 레이어 출력에 미치는 영향은 입력 feature 사이의 상관관계에 따라
달라진다. GPTQ는 $\mathbf{H}^{-1}$로 이 상관관계를 반영하고, 이미 quantize한
열에서 생긴 오차를 아직 full precision인 열에 나눠 준다.

## 제안 방법과 핵심 기호: OBQ에서 시작하기

가중치 행 $\mathbf{w}$에서 아직 full precision인 index 집합을 $F$라 하자.
$\operatorname{quant}(w)$는 integer code가 아니라 quantize한 뒤 다시 dequantize한
floating-point grid 값이다. 이때

$$
\mathbf{H}_F=2\mathbf{X}_F\mathbf{X}_F^T
$$

이고, OBQ가 다음으로 quantize할 index $q$의 score와 보정량은 다음과 같다.

$$
q^{*}
=
\arg\min_{q\in F}
\frac{
\left(\operatorname{quant}(w_q)-w_q\right)^2
}{
\left[\mathbf{H}_F^{-1}\right]_{qq}
},
\tag{3}
$$

$$
\boldsymbol{\delta}_F
=
-\frac{
w_q-\operatorname{quant}(w_q)
}{
\left[\mathbf{H}_F^{-1}\right]_{qq}
}
\left(\mathbf{H}_F^{-1}\right)_{:,q}.
\tag{4}
$$

식 (3)은 quantization 오차가 작고 Hessian 관점의 민감도도 낮은 가중치를 먼저 고른다.
식 (4)는 해당 가중치를 grid에 고정한 뒤 나머지 full-precision 가중치를 움직여 출력 오차를
최소화한다. $\mathbf{H}_F^{-1}$은 대칭 행렬이므로 구현에서는 열 대신 같은 값을 가진
행을 쓸 수 있다.

가중치 하나를 고정한 뒤에는 해당 행과 열을 제거한 역헤시안이 필요하다. OBQ는
다음 rank-1 downdate를 매 단계 수행한다.

$$
\left(\mathbf{H}_{-q}\right)^{-1}
=
\left(
\mathbf{H}^{-1}
-
\frac{
\mathbf{H}^{-1}_{:,q}\mathbf{H}^{-1}_{q,:}
}{
\left[\mathbf{H}^{-1}\right]_{qq}
}
\right)_{-q}.
\tag{5}
$$

OBQ는 행마다 greedy 순서가 다르다. 그러므로 식 (5)를
$d_{\mathrm{row}}d_{\mathrm{col}}$번 적용하며, 전체 복잡도는
$O(d_{\mathrm{row}}d_{\mathrm{col}}^3)$이 된다. ResNet-50 정도에서는 약 한 시간이지만,
수십억 개가 넘는 매개변수를 가진 LLM에는 지나치게 비싸다.

## 제안 방법: GPTQ의 세 가지 핵심 변경

아래에서는 원본 정리의 설명 순서에 맞춰 arbitrary order, Cholesky, lazy batch-update를
차례로 다룬다. GPTQ 논문 자체는 lazy batch-update를 Step 2, Cholesky를 Step 3으로 소개하지만
두 변경을 모두 적용한 최종 Algorithm 1은 같다.

### 1. Greedy 순서를 고정된 열 순서로 바꾼다

GPTQ 논문의 첫 번째 관찰은 레이어가 클수록 OBQ의 greedy 순서가 arbitrary 순서보다 주는
이득이 작아진다는 것이다. Greedy 순서는 quantization 오차가 큰 가중치를 뒤로 미룬다.
하지만 뒤로 갈수록 그 오차를 보상해 줄 full-precision 가중치도 줄어든다. 논문은 두 효과가
어느 정도 상쇄된다고 경험적으로 설명한다.

GPTQ는 모든 행을 같은 열 순서로 처리한다.

| 항목 | OBQ | 원 논문의 GPTQ |
| --- | --- | --- |
| 다음 대상 | 행별 최소 score 가중치 | 모든 행의 같은 열 |
| active set $F$ | 행마다 다름 | 모든 행이 동일 |
| 역행렬 갱신 횟수 | $d_{\mathrm{row}}d_{\mathrm{col}}$ | $d_{\mathrm{col}}$ |
| Hessian 비용 | $O(d_r d_c^3)$ | $O(d_c^3)$ |
| 가중치 보정 | 행별 수행 | 열 전체를 vectorize |

이 변경으로 식 (3)의 선택 score는 사라진다. 하지만 선택한 열을 quantize한 뒤 식 (4)의
보정량을 적용하는 second-order 구조는 그대로 남는다.

가중치 갱신 자체에는 여전히 $O(d_{\mathrm{row}}d_{\mathrm{col}}^2)$이 필요하다. 따라서 전체
복잡도는 다음과 같다.

$$
O\!\left(
\max\left\{
d_{\mathrm{row}}d_{\mathrm{col}}^2,
d_{\mathrm{col}}^3
\right\}
\right).
\tag{6}
$$

OBQ 대비 이론적인 가속 비율은
$\min\{d_{\mathrm{row}},d_{\mathrm{col}}\}$이다.

### 2. 반복할 역행렬 갱신을 Cholesky factor로 미리 계산한다

식 (5)를 수천 번 반복하면 floating-point 오차가 쌓여 $\mathbf{H}_F^{-1}$이 더는 양의
정부호가 아닐 수 있다. 이때 큰 보정값이 잘못된 방향으로 적용되면 레이어 전체의
quantization이 무너진다.

GPTQ는 먼저 damped Hessian을 만든다.

$$
\mathbf{H}_{\lambda}
=
2\mathbf{X}\mathbf{X}^{T}+\lambda\mathbf{I},
\qquad
\lambda
=
0.01\cdot
\operatorname{mean}\!\left(\operatorname{diag}(\mathbf{H})\right).
\tag{7}
$$

그다음 역행렬을 구하고 upper-triangular Cholesky factor $\mathbf{U}$를 한 번 계산한다.

$$
\mathbf{H}_{\lambda}^{-1}
=
\mathbf{U}^{T}\mathbf{U}.
\tag{8}
$$

공식 구현은 다음 세 연산을 사용한다. 변수 이름을 분명히 하려고 논문처럼 `H`를 재사용하지 않고
$\mathbf{L}$, $\mathbf{H}^{-1}$, $\mathbf{U}$로 나눠 적었다.

```python
L = torch.linalg.cholesky(H)
Hinv = torch.cholesky_inverse(L)
U = torch.linalg.cholesky(Hinv, upper=True)
```

입력 feature가 calibration sample에서 항상 0이면 $\mathbf{H}$의 해당 대각 원소도 0이다.
공식 구현은 이런 dead feature의 대각 원소를 1로 바꾸고 대응하는 가중치 열을 0으로 만든 뒤
dampening과 Cholesky를 적용한다.

이후에는 $\mathbf{U}$를 갱신하지 않는다. 각 단계에 필요한 역헤시안 행 정보가 이미
$\mathbf{U}$의 행에 순서대로 들어 있기 때문이다.

#### Cholesky가 Lemma 1의 update를 담는 이유

GPTQ 논문은 두 연산이 본질적으로 같다고 설명하지만 정식 증명은 싣지 않았다. 다음 블록
전개로 둘의 관계를 확인할 수 있다.

[상세 수치 예제와 증명으로 이동](#cholesky-lemma-1-proof)

먼저 Algorithm을 이해하는 데 필요한 일반식만 살펴보자.

현재 단계의 역헤시안을 첫 active 열을 기준으로

$$
\mathbf{M}
=
\begin{bmatrix}
a & \mathbf{b}^{T} \\
\mathbf{b} & \mathbf{C}
\end{bmatrix},
\qquad
\mathbf{U}
=
\begin{bmatrix}
u & \mathbf{r}^{T} \\
\mathbf{0} & \widehat{\mathbf{U}}
\end{bmatrix}
$$

라 놓고 $\mathbf{M}=\mathbf{U}^{T}\mathbf{U}$를 전개하면

$$
a=u^2,
\qquad
\mathbf{b}=u\mathbf{r},
\qquad
\mathbf{C}
=
\mathbf{r}\mathbf{r}^{T}
+
\widehat{\mathbf{U}}^{T}\widehat{\mathbf{U}}.
$$

따라서 남은 Cholesky 문제는

$$
\widehat{\mathbf{U}}^{T}\widehat{\mathbf{U}}
=
\mathbf{C}
-
\frac{\mathbf{b}\mathbf{b}^{T}}{a}.
\tag{9}
$$

식 (9)의 우변은 식 (5)로 첫 행과 열을 제거한 residual 행렬과 정확히 같다. 이 관계를
귀납적으로 반복하면 Cholesky factor의 각 행이 모든 단계의 residual 역행렬 정보를 담는다.
단, Cholesky 행은 해당 단계의 대각 원소 제곱근으로 정규화되어 있다.

$$
\mathbf{U}_{q,q:}
=
\frac{
\mathbf{M}^{(q)}_{q,q:}
}{
\sqrt{\mathbf{M}^{(q)}_{qq}}
},
\qquad
\mathbf{U}_{qq}
=
\sqrt{\mathbf{M}^{(q)}_{qq}}.
\tag{10}
$$

열 $q$의 오차 벡터를

$$
\mathbf{r}_q
=
\mathbf{W}_{:,q}-\mathbf{Q}_{:,q}
$$

라 하면 원래 OBQ 갱신은 다음처럼 Cholesky 형태로 정리된다.

$$
-\frac{\mathbf{r}_q}{\mathbf{M}^{(q)}_{qq}}
\mathbf{M}^{(q)}_{q,q:}
=
-\frac{\mathbf{r}_q}{\mathbf{U}_{qq}}
\mathbf{U}_{q,q:}.
\tag{11}
$$

즉 square-root scaling 하나가 약분되므로, 역행렬을 반복해 갱신하지 않고도 같은 보정
정보를 사용할 수 있다.

### 3. Lazy batch-update로 메모리 traffic을 줄인다

열 하나를 quantize할 때마다 남은 거대한 가중치 행렬 전체를 갱신하면, 연산량에 비해
메모리 접근이 지나치게 많아진다. 이런 low arithmetic-intensity 갱신은 GPU compute unit보다
메모리 bandwidth의 영향을 더 크게 받는다.

GPTQ는 기본적으로 연속한 $B=128$개 열을 하나의 quantization block으로 묶는다.

1. 블록 안에서는 열을 하나씩 quantize하고 같은 블록에서 아직 처리하지 않은 열만 즉시 갱신한다.
2. 블록이 끝나면 누적 오차를 행렬 곱 한 번으로 블록 바깥 전체에
   전파한다.

열 $j$의 rounding 결과는 이미 $j$에 반영된 앞선 갱신에만 의존한다. 아직 처리하지 않은
열에 보정량을 언제 전파하더라도 현재 $j$의 quantization 결과는 바뀌지 않는다. 이 성질 덕분에
lazy scheduling이 가능하다.

이 방법이 점근적인 FLOP 수를 줄이는 것은 아니다. 작은 rank-1 갱신을 GPU가 효율적으로
처리할 수 있는 행렬 곱으로 바꾼다. 논문은 큰 모델에서 실제 실행 시간이 약 10배 빨라졌다고
보고한다.

## 4. Calibration data가 흐르는 방식

Lazy-update의 $B=128$ 열 블록과 Transformer block은 서로 다른 개념이다.

| 용어 | 범위 | 목적 |
| --- | --- | --- |
| quantization block | linear 가중치의 128개 열 | 메모리 접근을 GEMM으로 묶음 |
| Transformer block | attention·MLP linear 레이어를 포함한 decoder 블록 하나 | 최대 메모리 제한과 누적 오차 전달 |

### OBC의 독립 레이어 흐름

OBC와 GPTQ의 차이는 대상 모델이 vision인지 LLM인지에 그치지 않는다. Quantization 중
calibration 데이터가 모델 안에서 흐르는 방식, 앞 레이어의 quantization 오차가 뒤 레이어의
Hessian에 반영되는 방식도 다르다. 다만 아래 OBC 절차는 일반적인 레이어별 PTQ 흐름을
정리한 것이다. GPTQ 논문이 OBC의 calibration·quantization pipeline을 정확히 규정한 것은 아니다.

1. Calibration 데이터로 밀집 모델 전체를 한 번 추론한다.
2. 각 레이어의 입력 $\mathbf{X}$를 수집하고 레이어별로
   $\mathbf{H}=2\mathbf{X}\mathbf{X}^{T}$를 계산한다.
3. 수집한 밀집 activation을 기준으로 모든 레이어를 서로 독립적으로 quantize한다.
4. 레이어별 quantization 결과를 합쳐 최종 모델을 만든다.

각 레이어의 Hessian은 원래 밀집 모델에서 얻은 activation으로 계산한다. 따라서 한 레이어에서
생긴 quantization 오차는 뒤 레이어의 Hessian에 반영되지 않는다. ResNet 같은 vision
모델에서는 이런 독립 처리만으로도 충분히 좋은 결과를 얻었다.

### GPTQ의 sequential Transformer-block 흐름

논문 실험은 C4에서 무작위로 고른 2,048-token segment 128개만 calibration 데이터로 사용한다.
모델 전체를 full precision으로 GPU에 올리지 않고 다음 순서로 처리한다.

1. Transformer block 하나를 GPU에 올린다.
2. 현재 calibration 입력으로 각 linear 레이어의 $\mathbf{H}=2\mathbf{X}\mathbf{X}^{T}$를
   누적한다.
3. 블록 안의 가중치를 GPTQ로 quantize한다.
4. Calibration 입력을 완전히 quantize된 현재 블록에 다시 통과시킨다.
5. 그 출력을 다음 Transformer block의 실제 calibration 입력으로 사용한다.

논문에서 이 Transformer block은 attention과 MLP의 선형 가중치 6개로 구성된 처리 단위다.
현재 블록의 quantization 오차가 다음 블록의 입력으로 전달되는 흐름은 다음과 같다.

```text
+-------------------+     +----------------------+     +----------------------+
| input for block t |---->| collect H + quantize |---->| replay quantized t   |
+-------------------+     +----------------------+     +-----------+----------+
                                                                    |
                                                                    v
                                                        +----------------------+
                                                        | input for block t+1  |
                                                        +----------------------+
```

따라서 뒤쪽 블록의 Hessian은 원래 밀집 모델의 activation이 아니라, 앞쪽 블록에서 이미
생긴 quantization 오차가 반영된 activation으로 계산된다. 모든 레이어를 독립적으로
quantize한 뒤 합치는 OBC의 기본 pipeline과 구별되는 실용적인 차이다.

| 비교 | OBC의 독립 레이어 pipeline | GPTQ 논문의 LLM pipeline |
| --- | --- | --- |
| 처리 순서 | 레이어 결과를 독립적으로 만든 뒤 결합 | Transformer block을 앞에서 뒤로 처리 |
| 다음 블록 입력 | 밀집 모델에서 수집한 activation | 이미 quantize된 앞 블록의 출력 |
| 이전 quantization 오차 반영 | 기본적으로 없음 | 있음 |
| 주된 목적 | compression 선택의 독립성과 재사용 | 거대 모델의 메모리 절감과 누적 오차 반영 |

## OBQ와 GPTQ의 차이 요약

| 항목 | OBQ | GPTQ Algorithm 1 |
| --- | --- | --- |
| 순서 | 행별 greedy | 공통 fixed 열 |
| 처리 단위 | 한 행의 가중치 하나 | 모든 행의 같은 열 |
| Hessian active set | 행마다 다름 | 모든 행에서 같음 |
| 역행렬 | 단계마다 rank-1 갱신 | Cholesky factor 1회 계산 |
| 갱신 | 가중치마다 전체 갱신 | 블록 내부 + lazy GEMM |
| 기본 블록 크기 | 해당 없음 | $B=128$개 열 |
| Calibration 흐름 | 독립적인 레이어 결과를 결합 | Transformer block을 순서대로 재실행 |
| 이론 복잡도 | 행마다 세제곱 비용 | 식 (6) |
| 주 대상 | 기존 vision·language model | 수십억 매개변수 이상 LLM |
| 포기한 것 | 해당 없음 | greedy order의 작은 이득 |
| 유지한 것 | second-order 보정 | second-order 보정 |

GPTQ의 근사는 Hessian 정보를 버려서 생기지 않는다. 원 논문에서 가장 큰 알고리즘상의
trade-off는 행별 greedy 순서를 포기하고 모두 같은 순서를 쓴다는 점이다. Cholesky 재구성과
lazy scheduling은 필요한 보정량을 더 안정적이고 효율적으로 계산하도록 연산 순서를 바꾼 것이다.

## 5. GPTQ Algorithm 1

[논문의 Algorithm 1](https://arxiv.org/pdf/2210.17323)은 다음과 같다.

![GPTQ Algorithm 1의 열 단위 양자화, 블록 내부 보정, 나머지 열에 대한 lazy update 순서](/assets/img/posts/gptq-second-order-ptq/gptq-algorithm-1.png)

그림은 Cholesky 변환 결과도 $\mathbf{H}^{-1}$ 기호에 덮어써서 표현한다. 이 행렬은 이 글에서
식 (8)의 upper-triangular factor $\mathbf{U}$라고 부른 값과 같다. $B$는 lazy-update 블록
크기다. 뒤의 설명에서는 블록 끝 인덱스를 $i_2=\min(i+B,d_{\mathrm{col}})$로 두며, 마지막
블록은 $d_{\mathrm{col}}$을 넘지 않게 잘라 처리한다.

### 줄마다 shape 확인하기

| 표현 | shape | 역할 |
| --- | ---: | --- |
| $\mathbf{Q}_{:,j}$ | $d_{\mathrm{row}}$ | 현재 열의 quantized-dequantized 값 |
| $\mathbf{E}_{:,j-i}$ | $d_{\mathrm{row}}$ | scaled 오차 |
| $\mathbf{U}_{j,j:i_2}$ | $i_2-j$ | 같은 블록 안으로 보낼 보정 계수 |
| $\mathbf{E}$ | $d_{\mathrm{row}}\times(i_2-i)$ | 한 블록에서 누적한 scaled 오차 |
| $\mathbf{U}_{i:i_2,i_2:}$ | $(i_2-i)\times(d_{\mathrm{col}}-i_2)$ | 블록 밖 계수 |

현재 위치까지 포함해

$$
\mathbf{W}_{:,j:i_2}
\leftarrow
\mathbf{W}_{:,j:i_2}
-
\mathbf{E}_{:,j-i}\mathbf{U}_{j,j:i_2}
\tag{12}
$$

를 적용하면 $j$번째 성분에서는

$$
\mathbf{W}_{:,j}
-
\frac{\mathbf{W}_{:,j}-\mathbf{Q}_{:,j}}{\mathbf{U}_{jj}}
\mathbf{U}_{jj}
=
\mathbf{Q}_{:,j}
$$

가 된다. 따라서 현재 열은 정확히 quantization grid에 고정되고, 나머지 성분만 오차를
나눠 받는다.

블록이 끝난 뒤에는

$$
\mathbf{W}_{:,i_2:}
\leftarrow
\mathbf{W}_{:,i_2:}
-
\mathbf{E}\mathbf{U}_{i:i_2,i_2:}
\tag{13}
$$

를 수행한다. 식 (13)은 $d_{\mathrm{row}}\times B$와
$B\times(d_{\mathrm{col}}-i_2)$의 행렬 곱이므로 GPU를 훨씬 효율적으로 사용한다.

## 6. 실험 결과

아래 표는 GPTQ 논문 v2의 수치를 대조해 Markdown으로 옮겼다.
Perplexity는 낮을수록, vision top-1 accuracy와 LAMBADA accuracy는 높을수록 좋다.

### 실험 설정

논문은 C4 training set에서 무작위로 뽑은 2,048-token segment 128개를 calibration 데이터로
사용한다. 모든 모델은 단일 NVIDIA A100 80GB에서 standard uniform per-row asymmetric
min-max grid로 quantize한다. 주요 결과는 activation을 FP16으로 유지하는 weight-only
quantization이며, Transformer block을 앞에서부터 하나씩 처리한다.

### 작은 vision model에서 accuracy 검증

| Method | RN18 4-bit | RN18 3-bit | RN50 4-bit | RN50 3-bit |
| --- | ---: | ---: | ---: | ---: |
| AdaRound | 69.34 | 68.37 | 75.84 | 75.14 |
| AdaQuant | 68.12 | 59.21 | 74.68 | 64.98 |
| BRECQ | 69.37 | 68.47 | 75.88 | 75.32 |
| OBQ | 69.56 | 68.69 | 75.72 | 75.24 |
| GPTQ | 69.37 | 67.88 | 75.71 | 74.87 |

밀집 baseline은 RN18 69.76%, RN50 76.13%다. GPTQ는 4-bit에서 기존 정밀 PTQ와 거의 같은
accuracy를 내지만, 3-bit에서는 OBQ보다 RN18 0.81%p, RN50 0.37%p 낮다. 고정 순서에 따른
작은 accuracy 손실이 보이는 대신, 한 시간가량 걸리는 비교 방법과 달리 GPTQ는 1분 안에
처리됐다고 논문은 보고한다.

### OBQ와 작은 Transformer model 비교

논문 Appendix A.1은 OBQ를 적용할 수 있는 BERT-base와 OPT-125M에서도 두 방법을 직접
비교한다.

| Method | BERT 4-bit F1 ↑ | BERT 3-bit F1 ↑ | OPT-125M 4-bit PPL ↓ | OPT-125M 3-bit PPL ↓ |
| --- | ---: | ---: | ---: | ---: |
| OBQ | 88.23 | 85.29 | 32.52 | 69.32 |
| GPTQ | 88.18 | 86.02 | 31.12 | 53.85 |

밀집 baseline은 BERT-base F1 88.53, OPT-125M perplexity 27.66이다. 4-bit에서는 두 방법이
비슷하고, 3-bit에서는 이 실험의 GPTQ가 오히려 더 낫다. 논문은 OBQ의 early outlier
rounding 같은 heuristic이 비전 모델이 아닌 경우 별도 조정이 필요할 가능성을 원인으로
추정한다.

### 전체 모델 quantization 실행 시간

모든 시간은 단일 NVIDIA A100 80GB에서 측정했다.

| Model | Parameters | Runtime |
| --- | ---: | ---: |
| OPT | 13B | 20.9 min |
| OPT | 30B | 44.9 min |
| OPT | 66B | 1.6 h |
| OPT | 175B | 4.2 h |
| BLOOM | 1.7B | 2.9 min |
| BLOOM | 3B | 5.2 min |
| BLOOM | 7.1B | 10.0 min |
| BLOOM | 176B | 3.8 h |

### OPT family: WikiText2 perplexity

넓은 원본 표를 블로그에서 읽기 쉽도록 두 표로 나눴다.

| Method | Bits | 125M | 350M | 1.3B | 2.7B | 6.7B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full precision | 16 | 27.65 | 22.00 | 14.63 | 12.47 | 10.86 |
| RTN | 4 | 37.28 | 25.94 | 48.17 | 16.92 | 12.10 |
| GPTQ | 4 | 31.12 | 24.24 | 15.47 | 12.87 | 11.39 |
| RTN | 3 | 1.3e3 | 64.57 | 1.3e4 | 1.6e4 | 5.8e3 |
| GPTQ | 3 | 53.85 | 33.79 | 20.97 | 16.88 | 14.86 |

| Method | Bits | 13B | 30B | 66B | 175B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full precision | 16 | 10.13 | 9.56 | 9.34 | 8.34 |
| RTN | 4 | 11.32 | 10.98 | 110 | 10.54 |
| GPTQ | 4 | 10.31 | 9.63 | 9.55 | 8.37 |
| RTN | 3 | 3.4e3 | 1.6e3 | 6.1e3 | 7.3e3 |
| GPTQ | 3 | 11.61 | 10.27 | 14.16 | 8.68 |

OPT-175B의 4-bit GPTQ는 FP16 대비 perplexity가 8.34에서 8.37로 0.03만 늘어난다. 반면
3-bit RTN은 모든 OPT 크기에서 크게 무너진다. GPTQ 3-bit도 작은 모델에서는 손실이 있지만,
가장 큰 175B 모델에서는 8.68을 유지한다.

### BLOOM family: WikiText2 perplexity

| Method | Bits | 560M | 1.1B | 1.7B | 3B | 7.1B | 176B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full precision | 16 | 22.42 | 17.69 | 15.39 | 13.48 | 11.37 | 8.11 |
| RTN | 4 | 25.90 | 22.00 | 16.97 | 14.76 | 12.10 | 8.37 |
| GPTQ | 4 | 24.03 | 19.05 | 16.48 | 14.20 | 11.73 | 8.21 |
| RTN | 3 | 57.08 | 50.19 | 63.59 | 39.36 | 17.38 | 571 |
| GPTQ | 3 | 32.31 | 25.08 | 21.11 | 17.40 | 13.47 | 8.64 |

BLOOM에서도 같은 경향이 나타난다. 176B의 4-bit GPTQ는 8.11에서 8.21, 3-bit GPTQ는
8.64로 유지되는 반면, 3-bit RTN은 571까지 증가한다.

### 175B models와 group-wise quantization

원 논문의 `g1024`, `g128`은 각각 연속한 1024개, 128개 weight마다 별도의 quantization
parameter를 사용하는 3-bit GPTQ다.

#### OPT-175B

| Method | Bits | WikiText2 ↓ | PTB ↓ | C4 ↓ | LAMBADA ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 16 | 8.34 | 12.01 | 10.13 | 75.59 |
| RTN | 4 | 10.54 | 14.22 | 11.61 | 71.34 |
| GPTQ | 4 | 8.37 | 12.26 | 10.28 | 76.80 |
| RTN | 3 | 7.3e3 | 8.0e3 | 4.6e3 | 0.00 |
| GPTQ | 3 | 8.68 | 12.68 | 10.67 | 76.19 |
| GPTQ, group 1024 | 3 | 8.45 | 12.48 | 10.47 | 77.39 |
| GPTQ, group 128 | 3 | 8.45 | 12.37 | 10.36 | 76.42 |

#### BLOOM-176B

| Method | Bits | WikiText2 ↓ | PTB ↓ | C4 ↓ | LAMBADA ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 16 | 8.11 | 14.59 | 11.71 | 67.40 |
| RTN | 4 | 8.37 | 15.00 | 12.04 | 66.70 |
| GPTQ | 4 | 8.21 | 14.75 | 11.81 | 67.71 |
| RTN | 3 | 571 | 107 | 598 | 0.17 |
| GPTQ | 3 | 8.64 | 15.57 | 12.27 | 65.10 |
| GPTQ, group 1024 | 3 | 8.35 | 15.01 | 11.98 | 67.47 |
| GPTQ, group 128 | 3 | 8.26 | 14.89 | 11.85 | 67.86 |

Group size를 줄이면 scale과 zero-point metadata가 늘어 실제 평균 bit 수도 조금 증가한다.
논문 기준 group 1024는 가중치당 약 0.02 bit, group 128은 약 0.15 bit를 추가하는 대신 3-bit
perplexity를 FP16에 훨씬 가깝게 만든다.

### 실제 generative inference latency

논문은 3-bit OPT-175B를 위한 양자화 행렬·FP16 벡터 전용 kernel도 평가했다. 다음
수치는 batch size 1로 길이 128의 sequence를 생성할 때 token당 평균 latency다.

| GPU | FP16 latency | 3-bit latency | Speedup | GPU 수 변화 |
| --- | ---: | ---: | ---: | ---: |
| A6000 48GB | 589 ms | 130 ms | 4.53× | 8 → 2 |
| A100 80GB | 230 ms | 71 ms | 3.24× | 5 → 1 |

3-bit OPT-175B의 가중치는 embedding과 FP16 output layer를 포함해 약 63GB이며, 최대 2,048
token의 KV cache 약 9GB를 더해도 A100 80GB 한 장에 들어간다. Latency 향상은 quantized
checkpoint만으로 자동으로 생기는 결과가 아니다. 가중치를 필요할 때 동적으로 dequantize하며
메모리 이동량을 줄이는 논문의 전용 kernel이 함께 있어야 한다.

### 2-bit와 ternary까지 낮춘 경우

논문은 grouping을 더 세분화한 extreme quantization도 별도로 평가한다.

| Model | FP16 | 2-bit g128 | 2-bit g64 | 2-bit g32 | 3-bit |
| --- | ---: | ---: | ---: | ---: | ---: |
| OPT-175B | 8.34 | 9.58 | 9.18 | 8.94 | 8.68 |
| BLOOM-176B | 8.11 | 9.55 | 9.17 | 8.83 | 8.64 |

표는 WikiText2 perplexity다. Scale과 zero-point metadata를 포함한 실제 평균 정밀도는 g128이
약 2.2 bit, g32가 약 2.6 bit다. Group size 8의 ternary $\{-1,0,+1\}$ quantization은
OPT-175B에서 perplexity 9.20을 기록했지만, group metadata 때문에 평균 compression rate는
위 2-bit 설정보다 나빠진다. 따라서 이 결과는 3~4 bit 주 결과와 구분해 해석해야 한다.

## 핵심 정리

1. GPTQ는 레이어 출력 재구성 오차를 second-order 정보로 줄이는 weight-only PTQ다.
2. 모든 행이 같은 열 순서를 사용하게 해 역헤시안 처리를
   $d_{\mathrm{row}}d_{\mathrm{col}}$번에서 $d_{\mathrm{col}}$번으로 줄인다.
3. Cholesky factor는 고정 순서에서 필요한 residual 역행렬 행을 안정적으로 미리
   담는다.
4. Lazy batch-update는 메모리 병목인 rank-1 갱신을 블록 GEMM으로 바꾼다.
5. 논문은 이 조합으로 OPT-175B와 BLOOM-176B를 단일 A100에서 약 4시간에 3~4 bit로
   quantize하면서 FP16에 가까운 perplexity를 유지했다.
6. 성능은 calibration 데이터, group size, quantization grid, kernel과 hardware에 따라 달라지므로
   “GPTQ 4-bit”라는 이름만으로 동일한 결과를 가정해서는 안 된다.

## 부록: Cholesky와 Lemma 1의 관계 {#cholesky-lemma-1-proof}

논문이 설명한 범위와 이 글에서 보충한 유도를 혼동하지 않도록 먼저 둘의 경계를 짚는다.

### A.1 논문이 실제로 말하는 범위

GPTQ 논문은 Cholesky와 반복적인 역헤시안 갱신이 왜 동치인지 정식으로 증명하지 않는다.
논문 5쪽의 “Step 3: Cholesky Reformulation”은 다음 두 가지 관찰만 제시한다.

1. 가중치 $q$를 quantize할 때
   $\left(\mathbf{H}_{F_q}^{-1}\right)$ 전체가 필요한 것은 아니다. 필요한 정보는 $q$번째
   행에서 대각 원소부터 시작하는 부분뿐이다.
2. symmetric $\mathbf{H}^{-1}$에 논문의 식 (3), 즉 이 글의 식 (5)를 적용해 row를 제거하는
   과정은 Cholesky decomposition과 본질적으로 대응한다. 차이는 Cholesky가 그 행을
   $\sqrt{[\mathbf{H}_{F_q}^{-1}]_{qq}}$로 나눈다는 점뿐이다.

논문은 이 관찰을 토대로 반복 갱신을 한 번의 Cholesky로 대체할 수 있다고 결론 내린다.
다만 그 이유를 수식으로 전개하지는 않는다. 아래 증명은 생략된 연결고리를 채운다.

공식 구현의 실제 계산 흐름은 다음과 같다. 같은 변수 `H`가 입력 Hessian, 그 역행렬,
마지막 upper-triangular factor를 차례로 가리킨다는 점에 주의하자.

```python
H = torch.linalg.cholesky(H)             # H 변수 = L, 원래 Hessian = L L^T
H = torch.cholesky_inverse(H)            # H 변수 = Hessian^{-1} = L^{-T} L^{-1}
H = torch.linalg.cholesky(H, upper=True) # H 변수 = U, Hessian^{-1} = U^T U
Hinv = H                                 # 실제 저장값은 inverse가 아니라 U
```

Lemma 1 갱신을 반복하면 floating-point 오차가 쌓일 수 있다. 위 방식은 필요한 정보를
upper-triangular $\mathbf{U}$에 한 번 계산해 둔 뒤 더는 갱신하지 않는다. 열 순서대로
처리하므로 각 단계에 필요한 행도 $\mathbf{U}$에서 차례로 꺼낼 수 있다.

### A.2 배경: Schur complement

Schur complement는 블록 행렬에서 한 블록을 제거한 뒤 남는 행렬로 이해할 수 있다.
$n\times n$ 대칭 행렬 $\mathbf{M}$에서 첫 행과 열을 분리하면

$$
\mathbf{M}
=
\begin{bmatrix}
m_{11} & \mathbf{m}_{1}^{T} \\
\mathbf{m}_{1} & \mathbf{M}_{22}
\end{bmatrix}
$$

이고, $m_{11}$을 제거한 Schur complement는

$$
\mathbf{M}_{22}
-
\frac{1}{m_{11}}
\mathbf{m}_{1}\mathbf{m}_{1}^{T}
\tag{A.1}
$$

이다. 이 글의 식 (5)와 같은 형태다. 아래에서는 Schur-complement 정리를 그대로 가져다
쓰는 대신 $\mathbf{U}^{T}\mathbf{U}=\mathbf{M}$을 블록 형태로 직접 전개해 같은 결론을
유도한다.

### A.3 한 번의 Cholesky가 반복 update를 대체하는 이유

#### A.3.1 설정

다음 역헤시안 $\mathbf{M}=\mathbf{H}^{-1}$을 예로 사용한다.
알고리즘의 배열 index와 맞추려고 이 절의 행렬 원소 첨자는 0부터 시작한다.

$$
\mathbf{M}
=
\begin{bmatrix}
4 & 2 & 6 \\
2 & 5 & 7 \\
6 & 7 & 22
\end{bmatrix},
\qquad
\mathbf{U}
=
\begin{bmatrix}
u_{00} & u_{01} & u_{02} \\
0 & u_{11} & u_{12} \\
0 & 0 & u_{22}
\end{bmatrix}.
\tag{A.2}
$$

목표는 다음 조건을 만족하는 upper-triangular $\mathbf{U}$를 찾는 일이다.

$$
\mathbf{U}^{T}\mathbf{U}=\mathbf{M}.
\tag{A.3}
$$

이 글에서는 이 형태의 upper-triangular Cholesky decomposition을 사용한다.

#### A.3.2 $\mathbf{U}$와 $\mathbf{U}^{T}$의 블록 분할

$\mathbf{U}$를 첫 행과 나머지 부분으로 나눈다.

$$
\mathbf{U}
=
\begin{bmatrix}
u_{00} & \mathbf{r}^{T} \\
\mathbf{0} & \widehat{\mathbf{U}}
\end{bmatrix},
\qquad
\mathbf{U}^{T}
=
\begin{bmatrix}
u_{00} & \mathbf{0}^{T} \\
\mathbf{r} & \widehat{\mathbf{U}}^{T}
\end{bmatrix},
\tag{A.4}
$$

각 block은 다음과 같다.

$$
\begin{aligned}
u_{00}&:\text{ scalar}, \\
\mathbf{r}^{T}&=[u_{01},u_{02}], \\
\mathbf{0}&=[0,0]^{T}, \\
\widehat{\mathbf{U}}
&=
\begin{bmatrix}
u_{11} & u_{12} \\
0 & u_{22}
\end{bmatrix}.
\end{aligned}
\tag{A.5}
$$

$\mathbf{r}^{T}$는 행 벡터이고 $\mathbf{r}$은 이를 transpose한 열 벡터다.
$\widehat{\mathbf{U}}$는 첫 행과 열을 제외하고 남은 upper-triangular 행렬이다.

#### A.3.3 블록 행렬 곱

식 (A.4)의 두 블록 행렬을 직접 곱한다.

$$
\begin{aligned}
\mathbf{U}^{T}\mathbf{U}
&=
\begin{bmatrix}
u_{00} & \mathbf{0}^{T} \\
\mathbf{r} & \widehat{\mathbf{U}}^{T}
\end{bmatrix}
\begin{bmatrix}
u_{00} & \mathbf{r}^{T} \\
\mathbf{0} & \widehat{\mathbf{U}}
\end{bmatrix} \\
&=
\begin{bmatrix}
u_{00}u_{00}+\mathbf{0}^{T}\mathbf{0}
&
u_{00}\mathbf{r}^{T}
+\mathbf{0}^{T}\widehat{\mathbf{U}} \\
\mathbf{r}u_{00}
+\widehat{\mathbf{U}}^{T}\mathbf{0}
&
\mathbf{r}\mathbf{r}^{T}
+\widehat{\mathbf{U}}^{T}\widehat{\mathbf{U}}
\end{bmatrix} \\
&=
\begin{bmatrix}
u_{00}^{2} & u_{00}\mathbf{r}^{T} \\
u_{00}\mathbf{r}
&
\mathbf{r}\mathbf{r}^{T}
+\widehat{\mathbf{U}}^{T}\widehat{\mathbf{U}}
\end{bmatrix}
=\mathbf{M}.
\end{aligned}
\tag{A.6}
$$

이 한 번의 곱셈으로 첫 Cholesky 행과 다음 단계에 넘길 residual 행렬을 함께 얻는다.

#### A.3.4 첫 행 풀기

식 (A.6)과 식 (A.2)의 첫 행과 열을 비교하자. 먼저 $(0,0)$ block은

$$
u_{00}^{2}=M_{00}=4
\quad\Longrightarrow\quad
u_{00}=\sqrt{4}=2
\tag{A.7}
$$

를 준다. Cholesky에서는 양의 대각 원소를 택한다. 이어서 $(0,1)$ block은

$$
u_{00}\mathbf{r}^{T}
=
\mathbf{M}_{0,1:}
=
\begin{bmatrix}
2 & 6
\end{bmatrix}
\tag{A.8}
$$

이므로

$$
\mathbf{r}^{T}
=
\frac{[2,6]}{2}
=
[1,3].
\tag{A.9}
$$

따라서 첫 행의 일반식은 다음과 같다.

$$
U_{00}=\sqrt{M_{00}},
\qquad
\mathbf{U}_{0,1:}
=
\frac{\mathbf{M}_{0,1:}}{U_{00}}.
\tag{A.10}
$$

#### A.3.5 Residual 행렬 유도: 핵심 단계

식 (A.6)의 오른쪽 아래 block만 떼어 쓰면

$$
\mathbf{r}\mathbf{r}^{T}
+
\widehat{\mathbf{U}}^{T}\widehat{\mathbf{U}}
=
\mathbf{M}_{1:,1:}
\tag{A.11}
$$

이므로, $\widehat{\mathbf{U}}$가 풀어야 할 문제는

$$
\widehat{\mathbf{U}}^{T}\widehat{\mathbf{U}}
=
\mathbf{M}_{1:,1:}
-
\mathbf{r}\mathbf{r}^{T}
=
\mathbf{M}_{1:,1:}
-
\mathbf{U}_{0,1:}^{T}\mathbf{U}_{0,1:}
\tag{A.12}
$$

가 된다. 위 예제의 숫자를 그대로 넣어 계산하면

$$
\mathbf{U}_{0,1:}^{T}\mathbf{U}_{0,1:}
=
\begin{bmatrix}
1 \\
3
\end{bmatrix}
\begin{bmatrix}
1 & 3
\end{bmatrix}
=
\begin{bmatrix}
1 & 3 \\
3 & 9
\end{bmatrix}.
\tag{A.13}
$$

따라서

$$
\begin{aligned}
\widehat{\mathbf{U}}^{T}\widehat{\mathbf{U}}
&=
\begin{bmatrix}
5 & 7 \\
7 & 22
\end{bmatrix}
-
\begin{bmatrix}
1 & 3 \\
3 & 9
\end{bmatrix} \\
&=
\begin{bmatrix}
4 & 4 \\
4 & 13
\end{bmatrix}
=
\mathbf{M}^{(1)}.
\end{aligned}
\tag{A.14}
$$

$\mathbf{M}^{(1)}$은 $\widehat{\mathbf{U}}$가 풀어야 할 새 Cholesky 문제다. 같은 절차를
반복하면 $\widehat{\mathbf{U}}$의 첫 행, 즉 원래 $\mathbf{U}$의 두 번째 행을 얻는다.

#### A.3.6 Residual이 Lemma 1과 같음을 보이기

위에서 구한 residual을 다시 적으면

$$
\mathbf{M}^{(1)}
=
\mathbf{M}_{1:,1:}
-
\mathbf{U}_{0,1:}^{T}\mathbf{U}_{0,1:}.
\tag{A.15}
$$

식 (A.10)에 의해

$$
\mathbf{U}_{0,1:}
=
\frac{\mathbf{M}_{0,1:}}{\sqrt{M_{00}}}
\tag{A.16}
$$

이므로 outer product는 다음과 같다.

$$
\begin{aligned}
\mathbf{U}_{0,1:}^{T}\mathbf{U}_{0,1:}
&=
\frac{\mathbf{M}_{1:,0}}{\sqrt{M_{00}}}
\frac{\mathbf{M}_{0,1:}}{\sqrt{M_{00}}} \\
&=
\frac{1}{M_{00}}
\mathbf{M}_{1:,0}\mathbf{M}_{0,1:}.
\end{aligned}
\tag{A.17}
$$

이를 식 (A.15)에 대입하면

$$
\boxed{
\mathbf{M}^{(1)}
=
\mathbf{M}_{1:,1:}
-
\frac{1}{M_{00}}
\mathbf{M}_{1:,0}\mathbf{M}_{0,1:}
}
\tag{A.18}
$$

을 얻는다. 이는 첫 행과 열을 제거한 뒤의 반복 갱신, 즉 GPTQ 논문 Lemma 1의
식 (3)이자 이 글의 식 (5)와 정확히 같다. $\mathbf{M}^{(1)}$에 같은 블록 전개를 반복하는
귀납법을 적용하면, 모든 단계 $k$에서 Cholesky의 residual 행렬과 Lemma 1의
$\mathbf{M}^{(k)}$가 같다는 결론을 얻는다.

#### A.3.7 유일한 차이: $\sqrt{M^{(k)}_{kk}}$ scaling

Residual 행렬 자체는 매 단계 같지만, Lemma 1이 사용하는 행과 Cholesky에 저장되는 행의
크기는 다르다.

$$
\begin{aligned}
\text{Lemma 1 row:}\quad
&\mathbf{M}^{(k)}_{k,k:}, \\
\text{Cholesky row:}\quad
&\mathbf{U}_{k,k:}
=
\frac{
\mathbf{M}^{(k)}_{k,k:}
}{
\sqrt{M^{(k)}_{kk}}
}.
\end{aligned}
\tag{A.19}
$$

3×3 예제의 모든 단계를 숫자로 비교하면 다음과 같다.

| $k$ | Lemma 1이 쓰는 행 | Cholesky가 저장한 행 | 다음 residual |
| ---: | --- | --- | --- |
| 0 | $[4,2,6]$ | $[2,1,3]$ ($\div\sqrt{4}$) | $[4,4;4,13]$ |
| 1 | $[4,4]$ | $[2,2]$ ($\div\sqrt{4}$) | $[9]$ |
| 2 | $[9]$ | $[3]$ ($\div\sqrt{9}$) | 종료 |

실제로 $k=1$에서 $\mathbf{M}^{(1)}$의 첫 행 $[4,4]$를 $\sqrt{4}=2$로 나누면
$\mathbf{U}_{1,1:}=[2,2]$가 된다. 그다음 residual은

$$
[13]-[2]^{T}[2]=[9]
\tag{A.20}
$$

이고, 마지막 Cholesky diagonal은 $\sqrt{9}=3$이다. 따라서 완성된 factor는

$$
\mathbf{U}
=
\begin{bmatrix}
2 & 1 & 3 \\
0 & 2 & 2 \\
0 & 0 & 3
\end{bmatrix},
\qquad
\mathbf{U}^{T}\mathbf{U}
=
\begin{bmatrix}
4 & 2 & 6 \\
2 & 5 & 7 \\
6 & 7 & 22
\end{bmatrix}.
\tag{A.21}
$$

즉 residual은 항상 정확히 같고, 꺼내 쓰는 행에만 대각 원소의 square-root scaling이 있다.

#### A.3.8 이 scaling이 GPTQ 결과에 영향을 주지 않는 이유

Lemma 1 형태에서 열 $j$를 quantize한 뒤의 원래 가중치 보정량은

$$
\boldsymbol{\delta}
=
-
\frac{
\mathbf{W}_{:,j}-\mathbf{Q}_{:,j}
}{
M^{(j)}_{jj}
}
\mathbf{M}^{(j)}_{j,j:}.
\tag{A.22}
$$

Cholesky 행과 residual 행 사이에는 식 (A.19)에 따라

$$
M^{(j)}_{jj}=U_{jj}^{2},
\qquad
\mathbf{M}^{(j)}_{j,j:}
=
U_{jj}\mathbf{U}_{j,j:}
\tag{A.23}
$$

의 관계가 있다. 이를 원래 보정식에 그대로 대입한다.

$$
\begin{aligned}
\boldsymbol{\delta}
&=
-
\frac{
\mathbf{W}_{:,j}-\mathbf{Q}_{:,j}
}{
U_{jj}^{2}
}
U_{jj}\mathbf{U}_{j,j:} \\
&=
-
\frac{
\mathbf{W}_{:,j}-\mathbf{Q}_{:,j}
}{
U_{jj}
}
\mathbf{U}_{j,j:} \\
&=
-\mathbf{E}_{:,j}\mathbf{U}_{j,j:},
\end{aligned}
\tag{A.24}
$$

여기서

$$
\mathbf{E}_{:,j}
=
\frac{
\mathbf{W}_{:,j}-\mathbf{Q}_{:,j}
}{
U_{jj}
}.
\tag{A.25}
$$

분모 $U_{jj}^{2}$ 중 하나가 residual 행에 들어 있는 $U_{jj}$와 정확히 약분된다. 따라서
Cholesky 행을 square root로 나눠 저장해도 보정 결과는 달라지지 않는다.
GPTQ Algorithm 1의

```text
E[:, j - i] = (W[:, j] - Q[:, j]) / U[j, j]
W[:, j:i2] -= E[:, j - i, None] @ U[j, j:i2][None, :]
```

가 바로 이 약분을 끝낸 형태다. 결론적으로 $\mathbf{U}$는 반복 갱신의 근사가 아니다.
고정된 열 순서에서 Lemma 1이 만드는 residual 정보 전체를 scaling된 형태로 미리 저장한다.

## 참고 자료

- Elias Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323), ICLR 2023.
- IST-DASLab, [GPTQ 공식 구현](https://github.com/IST-DASLab/gptq).
- Elias Frantar et al., [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/pdf/2208.11580), NeurIPS 2022.
