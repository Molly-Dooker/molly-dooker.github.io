---
title: "Optimal Brain Compression"
date: 2026-08-10 10:58:07 +0900
categories: [DeepLearning, Quantization]
tags: [quantization]
description: "Optimal Brain Compression이 행 단위 Hessian과 rank-1 갱신으로 ExactOBS pruning과 OBQ quantization을 하나의 틀에 담는 원리를 설명합니다."
math: true
render_with_liquid: false
---

## 문서 범위

- 핵심: 레이어 재구성 문제의 행 단위 분해, 공통 Hessian, ExactOBS, Optimal Brain Quantizer(OBQ), 역헤시안의 rank-1 갱신
- 선행 글: [Optimal Brain Surgeon](/posts/optimal-brain-surgeon/)
- 후속 글: [GPTQ](/posts/gptq-second-order-ptq/)
- 기준 자료: Frantar, Singh, Alistarh의 [Optimal Brain Compression 논문](https://arxiv.org/pdf/2208.11580)
- 확인일: 2026-08-11

## 동기와 목표

Optimal Brain Surgeon(OBS)은 가중치 하나를 제거할 때 늘어나는 손실을 역헤시안으로 계산하고, 남은 가중치를 함께 보정한다. 이 과정을 가중치마다 정확히 반복하려면 가중치를 하나 지울 때마다 활성 Hessian의 역행렬을 다시 구해야 한다.

레이어 하나의 가중치 행렬과 전체 가중치 수를 다음처럼 두자.

$$
W\in\mathbb{R}^{d_{\mathrm{row}}\times d_{\mathrm{col}}},
\qquad
d=d_{\mathrm{row}}d_{\mathrm{col}}.
$$

모든 가중치를 길이 $d$인 벡터로 펼치면 Hessian의 크기는 $d\times d$다. 가중치를 하나 제거할 때마다 역행렬을 처음부터 구하면 한 단계에 $\Theta(d^3)$, 최대 $O(d)$단계에 총 $O(d^4)$가 든다. 가중치가 수백만 개인 현대적인 DNN에는 적용하기 어려운 비용이다.

**Optimal Brain Compression(OBC)**은 두 가지 구조로 이 병목을 해결한다.

1. 레이어 재구성 손실을 $d_{\mathrm{row}}$개의 독립적인 행 문제로 정확히 나눈다.
2. 가중치 하나를 제거한 뒤 Lemma 1의 rank-1 연산으로 활성 역헤시안을 갱신한다.

그 결과 전체 시간 복잡도는 $O(d^4)$에서 $O(d_{\mathrm{row}}d_{\mathrm{col}}^3)$로 줄어든다. 이 과정은 선택한 레이어별 이차 목적 함수의 Hessian에 추가 근사를 도입하지 않는다.

| 항목 | 직접적인 레이어 전체 OBS | ExactOBS |
| --- | ---: | ---: |
| Hessian 크기 | $d\times d$ | $d_{\mathrm{col}}\times d_{\mathrm{col}}$ |
| 역헤시안 갱신 1회 | $\Theta(d^3)$ | $\Theta(d_{\mathrm{col}}^2)$ |
| 전체 시간 | $O(d^4)$ | $O(d_{\mathrm{row}}d_{\mathrm{col}}^3)$ |
| 핵심 행렬 메모리 | $\Theta(d^2)$ | $\Theta(d_{\mathrm{col}}^2)$ |

OBC는 두 방법을 하나의 프레임워크로 묶는다.

| 이름 | 역할 |
| --- | --- |
| ExactOBS | 가중치를 하나씩 0으로 만들고 매 단계 남은 가중치와 역헤시안을 갱신하는 레이어별 exact greedy pruning |
| OBQ | OBS의 목표값을 0에서 quantized value로 바꾼 quantization |
| OBC | ExactOBS와 OBQ를 아우르는 post-training compression 프레임워크 |

논문은 통합 프레임워크를 **Optimal Brain Compressor**라고 부르고, 제목에서는 **Optimal Brain Compression**이라는 표현을 사용한다. 둘 다 약어는 OBC다.

OBC는 one-shot 또는 post-training compression을 다룬다. 이미 학습한 모델과 소량의 calibration 입력을 사용하며, 압축한 모델을 gradient로 다시 학습하지 않는다. 그렇다고 post-training이 calibration 데이터나 통계 보정을 전혀 쓰지 않는다는 뜻은 아니다.

## 제안 방법과 핵심 기호

먼저 한 행의 ExactOBS를 읽는 데 필요한 기호를 정리하자. 레이어 문제를 왜 행 단위로 바꿀 수 있는지는 2절에서, 모든 행이 왜 같은 $H$를 쓰는지는 3절에서 설명한다.

현재 가중치 행을 $w\in\mathbb{R}^{1\times d_{\mathrm{col}}}$, 활성 인덱스 집합을 $M$이라고 하자. 후보 $p\in M$의 OBS score와 선택 결과는 다음과 같다.

$$
s_p
=
\frac{w_p^2}{[H^{-1}]_{pp}},
\qquad
p^*
=
\operatorname*{arg\,min}_{p\in M}s_p.
$$

$p$를 제거할 때 사용하는 보정 벡터는 다음과 같다.

$$
\delta_p
=
-\frac{w_p}{[H^{-1}]_{pp}}H^{-1}_{:,p}.
$$

여기서 $s_p$는 후보끼리 비교하는 **스칼라 손실 score**이고, $\delta_p$는 선택한 가중치의 오차를 다른 좌표에 분배하는 **벡터 보정량**이다. 둘은 역할과 shape가 다르다. Magnitude pruning과 달리 $w_p^2$만 비교하지 않으며, 분모의 $[H^{-1}]_{pp}$에는 곡률과 다른 가중치가 해당 변화를 보상할 수 있는 정도가 반영된다.

| 표기 | 의미 | Shape |
| --- | --- | ---: |
| $H$ | 한 행의 재구성 손실에 대한 Hessian, $2XX^T$ | $d_{\mathrm{col}}\times d_{\mathrm{col}}$ |
| $H^{-1}$ | 현재 활성 Hessian의 역행렬 | $d_{\mathrm{col}}\times d_{\mathrm{col}}$ 저장 공간 |
| $p$ | 현재 제거 후보인 가중치 인덱스 | 스칼라 |
| $w_p$ | 행 벡터 $w$의 $p$번째 가중치 | 스칼라 |
| $[H^{-1}]_{pp}$ | 역헤시안의 $p$번째 대각 성분 | 스칼라 |
| $H^{-1}_{:,p}$ | 모든 행과 $p$번째 열을 택한 열벡터 | $d_{\mathrm{col}}\times1$ |
| $H^{-1}_{p,:}$ | $p$번째 행과 모든 열을 택한 행벡터 | $1\times d_{\mathrm{col}}$ |
| $\delta_p$ | $p$를 제거할 때 전체 좌표에 더하는 보정 벡터 | $d_{\mathrm{col}}\times1$ |

### $H^{-1}_{:,p}$에서 `:`가 뜻하는 것

행렬 인덱스의 `:`는 해당 축의 모든 인덱스를 선택한다는 뜻이다. 따라서 $H^{-1}_{:,p}$는 “행은 전부, 열은 $p$ 하나”를 선택한 $p$번째 열이다. 반대로 $H^{-1}_{p,:}$는 $p$번째 행이다. 아래 예는 논문처럼 1부터 인덱스를 세며, 실제 code에서는 언어에 따라 0부터 셀 수 있다.

대칭인 $3\times3$ 역헤시안에서 $p=2$인 예를 보자.

$$
H^{-1}
=
\begin{bmatrix}
a & b & c \\
b & d & e \\
c & e & f
\end{bmatrix}.
$$

그러면 세 표기는 각각 다음 값을 가리킨다.

$$
H^{-1}_{:,2}
=
\begin{bmatrix}
b \\
d \\
e
\end{bmatrix},
\qquad
H^{-1}_{2,:}
=
\begin{bmatrix}
b & d & e
\end{bmatrix},
\qquad
[H^{-1}]_{22}=d.
$$

논문의 수식은 보정량을 열벡터로 두므로 $H^{-1}_{:,p}$를 사용한다. 구현에서 $w$를 행벡터로 유지한다면 같은 갱신을 다음처럼 읽는 편이 자연스럽다.

$$
w
\leftarrow
w-
\frac{w_p}{[H^{-1}]_{pp}}H^{-1}_{p,:}.
$$

$H$는 대칭이고 역행렬도 대칭이므로 두 벡터의 값은 transpose 관계다.

$$
H^{-1}_{:,p}
=
\left(H^{-1}_{p,:}\right)^T.
$$

보정 벡터의 $p$번째 성분을 확인하면 가중치가 정확히 0이 되는 것도 알 수 있다.

$$
[\delta_p]_p
=
-\frac{w_p}{[H^{-1}]_{pp}}[H^{-1}]_{pp}
=
-w_p.
$$

따라서 $w_p+[\delta_p]_p=0$이다. 아래첨자 $p$가 붙은 $\delta_p$는 “후보 $p$를 제거할 때의 보정 벡터”를 뜻하고, 그 벡터의 $p$번째 성분은 $[\delta_p]_p$처럼 구분해 쓴다.

목적 함수를 $\frac{1}{2}\delta^TH\delta$로 쓰면 실제 손실 증가량에는 $1/2$이 붙는다. 모든 후보에 공통인 상수이므로 선택 순서에는 영향을 주지 않는다.

## 1. ExactOBS 알고리즘

한 행 $w$에서 가중치 $k$개를 제거하는 ExactOBS의 실행 순서는 다음과 같다.

1. 레이어 입력 $X$로 $H=2XX^T$와 초기 $H^{-1}$을 한 번 계산한다.
2. 활성 집합 $M$에서 $w_p^2/[H^{-1}]_{pp}$가 가장 작은 인덱스 $p$를 고른다.
3. $H^{-1}_{:,p}$를 이용해 남은 가중치를 보정한다.
4. Lemma 1로 $H^{-1}$을 갱신한다.
5. $p$를 $M$에서 제거하고 목표 개수에 도달할 때까지 2단계부터 반복한다.

이 절에서는 전체 동작을 먼저 보여 준다. 행 단위 처리가 정확한 이유와 Lemma 1의 유도는 2~4절에서 차례로 설명한다.

논문의 Algorithm 1은 다음과 같다.

![ExactOBS Algorithm 1의 활성 집합 초기화, 최소 saliency 가중치 선택, 가중치 보정, 역헤시안 rank-1 갱신, 선택 인덱스 비활성화 순서](/assets/img/posts/optimal-brain-compression/exactobs-algorithm.png)

가중치 $k$개를 제거할 때까지 최소 saliency를 갖는 좌표를 선택하고, 가중치와 역헤시안을 갱신한 뒤 활성 집합 $M$에서 해당 좌표를 제외한다.

가중치와 역헤시안을 갱신할 때는 모두 **갱신 전** $w_p$, $H^{-1}_{:,p}$, $H^{-1}_{p,:}$, $[H^{-1}]_{pp}$를 사용해야 한다. 따라서 실제 구현에서는 이 네 값을 갱신 전에 저장한다.

한 단계의 가중치 보정은 $O(d_{\mathrm{col}})$이고, 역헤시안 downdate는 $O(d_{\mathrm{col}}^2)$다. 따라서 한 행에서 $k$개를 제거하는 비용은 $O(kd_{\mathrm{col}}^2)$다. 모든 가중치를 처리하면 한 행에 $O(d_{\mathrm{col}}^3)$, 레이어 전체에 $O(d_{\mathrm{row}}d_{\mathrm{col}}^3)$가 든다.

### 1.1 레이어 전체의 전역 pruning mask

ExactOBS는 각 행에서 가중치가 제거되는 순서 $P$와 단계별 스칼라 손실 변화량 $\Delta L$을 기록한다. 레이어 전체에서 $K$개를 제거할 때는 행별 제거 순서를 건너뛰지 않으면서, 현재 선택할 수 있는 후보 가운데 $\Delta L$이 가장 작은 가중치를 차례로 골라 전역 mask를 만든다. N:M sparsity처럼 행이나 block마다 제거할 개수가 정해진 pattern에서는 이 병합 단계가 필요 없다.

## 2. 핵심 통찰: 레이어 문제를 행 문제로 바꾸기

Calibration 입력을 열 방향으로 모은 행렬을 다음과 같이 두자.

$$
X\in\mathbb{R}^{d_{\mathrm{col}}\times N}.
$$

원래 가중치 $W$와 압축한 가중치 $W_c$의 출력 차이를 제곱 Frobenius norm으로 측정하면 레이어별 압축 문제는 다음과 같다.

$$
W_c^*
=
\operatorname*{arg\,min}_{W_c}
\left\lVert WX-W_cX\right\rVert_F^2
\quad
\text{subject to}
\quad
\mathcal{C}(W_c).
$$

$\mathcal{C}(W_c)$에는 원하는 sparsity pattern이나 quantization grid 같은 제약 조건이 들어간다.

제곱 Frobenius norm을 행별로 전개하면 다음 identity를 얻는다.

$$
\left\lVert WX-W_cX\right\rVert_F^2
=
\sum_{i=1}^{d_{\mathrm{row}}}
\left\lVert
W_{i,:}X-W_{c,i,:}X
\right\rVert_2^2.
$$

이 식이 정확히 분해되는 이유는 두 가지다.

- 행렬 곱에서 출력 행 $i$는 가중치 행 $W_{i,:}$에만 의존한다.
- 제곱 Frobenius norm은 각 출력 행의 제곱 $L_2$ 오차를 모두 더한 값이다.

따라서 한 행의 가중치를 바꿔도 다른 행의 재구성 오차는 변하지 않는다. 행마다 압축 목표가 정해져 있다면 각 문제를 따로 푸는 일은 레이어 목적 함수의 근사가 아니라 정확한 대수적 분해다. 이 구조 덕분에 여러 행을 병렬로 처리할 수도 있다.

다만 **행별 목적 함수가 독립적이라는 사실과 레이어 전체의 sparsity budget을 모든 행에 똑같이 나눠도 된다는 주장은 별개**다. 전역 unstructured pruning에서는 1.1절처럼 어느 행에서 몇 개를 지울지 따로 정해야 한다.

## 3. 모든 행이 같은 $H$를 공유하는 이유

원래 밀집 가중치 행을 $w^{(0)}$, 최적화 중인 행을 $w$, 고정된 목표 출력을 $t=w^{(0)}X$라고 하자. 이 행의 재구성 손실은 다음과 같다.

$$
L(w)
=
\left\lVert wX-t\right\rVert_2^2.
$$

Gradient와 Hessian을 행렬 형태로 쓰면

$$
\nabla_w L
=
2(wX-t)X^T,
$$

$$
\boxed{H=\nabla_w^2 L=2XX^T}
$$

를 얻는다. 최종 Hessian에는 가중치 $w$가 사라지고 calibration 입력 $X$만 남는다. 같은 레이어의 모든 출력 행은 동일한 $X$를 받으므로 $H$와 초기 $H^{-1}$을 한 번 계산해 함께 쓸 수 있다.

### $H$와 $H^{-1}$의 대칭성

$H=2XX^T$는 대칭 행렬이다.

$$
H^T
=
(2XX^T)^T
=
2XX^T
=
H.
$$

가역 대칭 행렬의 역행렬도 대칭이다.

$$
(H^{-1})^T
=
(H^T)^{-1}
=
H^{-1}.
$$

따라서 앞에서 사용한 $H^{-1}_{:,p}$와 $H^{-1}_{p,:}$는 방향만 다르고 같은 성분을 담는다.

밀집 시작점 $w=w^{(0)}$에서는 $wX-t=0$이므로 재구성 손실과 gradient가 모두 0이다. 더 중요한 점은 $L(w)$ 자체가 이차식이라는 사실이다. 고전적인 OBS의 second-order model은 일반 task loss를 Taylor 근사한 식이지만, 여기서는 행 재구성 목적 함수 그 자체다.

이 차이가 OBC가 사용하는 **exact**라는 표현의 첫 번째 근거다.

## 4. $H^{-1}$ 갱신: Lemma 1

가중치 $p$를 제거하면 현재 Hessian에서 $p$번째 행과 열을 뺀 주부분행렬이 남는다. 이를 $H_{-p}$라고 쓰자. 다음 단계에 필요한 값은 이 행렬의 역행렬 $(H_{-p})^{-1}$이다.

여기서 두 표기를 구분해야 한다.

- $(H_{-p})^{-1}$: $H$에서 $p$번째 행과 열을 먼저 제거한 뒤 구한 역행렬
- $(H^{-1})_{-p}$: 기존 $H^{-1}$에서 $p$번째 행과 열만 잘라낸 행렬

두 연산은 순서를 바꿀 수 없으므로 일반적으로 다음 식이 성립한다.

$$
(H_{-p})^{-1}
\ne
(H^{-1})_{-p}.
$$

Lemma 1은 원하는 $(H_{-p})^{-1}$을 기존 $H^{-1}$에서 직접 계산한다.

$$
\boxed{
(H_{-p})^{-1}
=
\left(
H^{-1}
-
\frac{
H^{-1}_{:,p}H^{-1}_{p,:}
}{
[H^{-1}]_{pp}
}
\right)_{-p}
}.
$$

$H^{-1}_{:,p}H^{-1}_{p,:}$는 $d_{\mathrm{col}}\times1$ 열벡터와 $1\times d_{\mathrm{col}}$ 행벡터의 외적(outer product)이다. 결과는 $d_{\mathrm{col}}\times d_{\mathrm{col}}$ rank-1 행렬이다. 이를 스칼라 $[H^{-1}]_{pp}$로 나눠 기존 역헤시안에서 뺀 뒤, $p$번째 행과 열을 제외한다.

실제 구현은 행렬 크기와 인덱스를 매번 바꾸지 않고 다음 rank-1 downdate를 전체 저장 공간에 적용한 뒤 $p$를 비활성화한다.

$$
H^{-1}
\leftarrow
H^{-1}
-
\frac{
H^{-1}_{:,p}H^{-1}_{p,:}
}{
[H^{-1}]_{pp}
}.
$$

Rank-1 항을 계산하는 비용은 $\Theta(d_{\mathrm{col}}^2)$다. 활성 Hessian을 만들고 역행렬을 처음부터 구하면 한 단계에 $O(d_{\mathrm{col}}^3)$, 한 행의 모든 가중치를 처리하는 데 $O(d_{\mathrm{col}}^4)$가 든다. Lemma 1은 이를 각각 $O(d_{\mathrm{col}}^2)$와 $O(d_{\mathrm{col}}^3)$로 줄인다.

### 왜 이 식이 성립하는가

성분 단위로 보면 downdate 이후의 $(i,j)$ 성분은 다음과 같다.

$$
[H^{-1}]_{ij}
-
\frac{[H^{-1}]_{ip}}{[H^{-1}]_{pp}}
[H^{-1}]_{pj}.
$$

이는 Gaussian elimination에서 행 $i$로부터 pivot 행 $p$의 $[H^{-1}]_{ip}/[H^{-1}]_{pp}$배를 빼는 연산과 같다. 모든 행 $i$에 이 연산을 적용하면 $p$번째 열이 0이 된다. 빼는 외적이 대칭 행렬이므로 갱신 결과도 대칭이고, 따라서 $p$번째 행도 함께 0이 된다.

이 기본 행 연산을 $H^{-1}H=I$의 양변에 똑같이 적용하자. 변환한 식에서 $p$번째 행과 열을 지우면 남은 블록은 다음 관계를 만족한다.

$$
\left(
H^{-1}
-
\frac{H^{-1}_{:,p}H^{-1}_{p,:}}
{[H^{-1}]_{pp}}
\right)_{-p}
H_{-p}
=
I.
$$

따라서 왼쪽 첫 번째 행렬이 정확히 $(H_{-p})^{-1}$이다. 여기서 $H=2XX^T$ 자체의 값은 바뀌지 않는다. 가중치를 제거할 때 달라지는 것은 이후 계산에 참여하는 활성 행과 열의 집합이다.

위 결론에서 생략한 행 연산 행렬 $C$와 블록 곱의 전개는 [Lemma 1 상세 증명](#lemma-1-proof)에 정리한다.

엄밀히 말해 첫 downdate 이후 전체 저장 배열은 처음의 밀집 $H$에 대한 역행렬이 아니다. 알고리즘은 표기를 단순하게 유지하려고 계속 $H^{-1}$이라고 쓰지만, 실제 의미는 “현재 활성 집합 $M$에 해당하는 $H_{M,M}$의 역행렬을 원래 크기 배열에 넣어 둔 상태”다. 비활성화한 행과 열은 이후 계산에서 읽지 않는다.

## 5. OBQ: 0 대신 quantized value에 고정하기

Pruning은 선택한 가중치를 0으로 만드는 제약이다. Optimal Brain Quantizer(OBQ)는 이 목표값을 quantization grid에서 표현할 수 있는 값으로 바꾼다.

$Q(w)$를 integer code가 아니라 **quantize한 뒤 dequantize한 floating-point 값**으로 정의하자. Quantization residual은 다음과 같다.

$$
r_p
=
w_p-Q(w_p).
$$

선택한 좌표가 만족해야 할 제약 조건은

$$
e_p^T\delta_p
=
Q(w_p)-w_p
=
-r_p
$$

다. OBS의 제약 이차 문제 해에 이를 대입하면 OBQ의 score와 보정량을 얻는다.

$$
\boxed{
p^*
=
\operatorname*{arg\,min}_{p\in M}
\frac{r_p^2}{[H^{-1}]_{pp}}
},
$$

$$
\boxed{
\delta_p
=
-\frac{r_p}{[H^{-1}]_{pp}}H^{-1}_{:,p}
}.
$$

선택한 좌표에 보정량을 적용하면

$$
w_p+[\delta_p]_p
=
w_p
-
\frac{w_p-Q(w_p)}{[H^{-1}]_{pp}}[H^{-1}]_{pp}
=
Q(w_p)
$$

가 된다. ExactOBS와 OBQ의 식에서 달라지는 것은 residual뿐이다.

| 항목 | ExactOBS pruning | OBQ quantization |
| --- | --- | --- |
| 고정 목표값 | $0$ | $Q(w_p)$ |
| Residual $r_p$ | $w_p$ | $w_p-Q(w_p)$ |
| Selection score | $w_p^2/[H^{-1}]_{pp}$ | $[w_p-Q(w_p)]^2/[H^{-1}]_{pp}$ |
| Compensation | $-w_pH^{-1}_{:,p}/[H^{-1}]_{pp}$ | $-[w_p-Q(w_p)]H^{-1}_{:,p}/[H^{-1}]_{pp}$ |
| 역헤시안 갱신 | Lemma 1 | 같은 Lemma 1 |

### Quantization order가 결과를 바꾼다

일반적인 rounding은 각 가중치의 처음 값을 서로 독립적으로 grid에 투영한다. OBQ의 순서는 다르다.

1. 현재 score가 가장 작아 쉽게 quantize할 수 있는 가중치 하나를 고정한다.
2. 그 rounding 오차를 아직 quantize하지 않은 가중치 전체에 나눠 준다.
3. 보정한 가중치에서 $Q(w)$와 residual을 다시 계산한다.
4. 모든 가중치가 grid에 고정될 때까지 반복한다.

보정 과정에서 다음 후보의 값 자체가 바뀌므로, 나중에 선택할 quantization level도 처음 값을 독립적으로 반올림했을 때와 달라질 수 있다. **순서가 처리 차례뿐 아니라 이후의 rounding 결과까지 바꾼다**는 점이 OBQ의 핵심이다.

행 전체를 quantize하면 결국 모든 가중치를 고정하므로 pruning의 전역 sparsity mask 단계는 필요 없다. 나머지 시간·메모리 특성은 ExactOBS와 같다.

### Quantization outlier를 먼저 처리하는 이유

$b$-bit uniform grid가 $x_{\min}$과 $x_{\max}$ 사이에 $2^b$개의 값을 둔다면 일반적인 step size는 다음과 같다.

$$
\Delta
=
\frac{x_{\max}-x_{\min}}{2^b-1}.
$$

Grid 범위 안에서 nearest rounding만 적용하면 error는 항상 다음 범위 안에 있다.

$$
\left\lvert Q(w)-w\right\rvert
\le
\frac{\Delta}{2}.
$$

그러나 일부 극단값을 clipping하도록 range를 정했거나 앞선 보정이 가중치를 grid 밖으로 밀어내면 다음 조건을 만족하는 outlier가 생길 수 있다.

$$
\left\lvert Q(w)-w\right\rvert
>
\frac{\Delta}{2}.
$$

Residual이 큰 가중치는 일반 score의 오름차순에서 뒤로 밀리기 쉽다. 마지막까지 미루면 큰 오차를 흡수할 미양자화 가중치가 거의 남지 않는다. 그래서 OBQ는 이런 outlier를 발견하는 즉시 먼저 quantize한다.

이 규칙은 이차 목적 함수에서 유도한 최적 조건이 아니다. 논문이 clipping과 연쇄 오차를 다루려고 추가한 **heuristic**이다. 일반적인 nearest rounding 범위 안에서는 위 조건이 생기지 않는다는 점도 기억해야 한다.

## 6. 실험 결과

아래 수치는 가중치 행렬에 OBC 식만 적용해서 얻은 결과가 아니다. 모든 calibration dataset은 무작위 학습 샘플 1,024개로 구성했다. ImageNet 기준으로는 학습 데이터의 약 0.1%이며, flip과 crop으로 calibration 샘플을 10배 늘렸다. ResNet에는 batch normalization statistics reset을, YOLO와 BERT에는 mean/variance correction을 적용했다.

따라서 결과를 재현하거나 다른 PTQ 방법과 비교할 때 이 post-processing을 포함해야 한다.

### Unstructured pruning

다음은 논문 Table 1의 결과다. Reduction factor는 parameter sparsity가 아니라 목표 FLOP reduction이며, metric은 ResNet50의 ImageNet top-1 accuracy, YOLOv5l의 COCO mAP@0.5, BERT의 SQuAD F1이다.

| Model / dense metric | Method | 2x | 3x | 4x |
| --- | --- | ---: | ---: | ---: |
| ResNet50 / 76.13 | GMP | 74.86 | 71.44 | 64.84 |
| ResNet50 / 76.13 | L-OBS | 75.48 | 73.73 | 71.24 |
| ResNet50 / 76.13 | AdaPrune | 75.53 | 74.47 | 72.39 |
| ResNet50 / 76.13 | ExactOBS | **75.64** | **75.01** | **74.05** |
| YOLOv5l / 66.97 | GMP | 65.83 | 62.30 | 55.09 |
| YOLOv5l / 66.97 | L-OBS | **66.21** | 64.47 | 61.15 |
| YOLOv5l / 66.97 | AdaPrune | 66.00 | 64.88 | 62.71 |
| YOLOv5l / 66.97 | ExactOBS | 66.14 | **65.35** | **64.05** |
| BERT / 88.53 | GMP | 65.64 | 12.52 | 9.23 |
| BERT / 88.53 | L-OBS | 77.67 | 3.62 | 6.63 |
| BERT / 88.53 | AdaPrune | 87.12 | 70.32 | 18.75 |
| BERT / 88.53 | ExactOBS | **87.81** | **85.87** | **82.10** |

압축 강도가 높아질수록 이 실험에서 ExactOBS의 우위가 커진다. 특히 BERT 4x에서 ExactOBS는 F1 82.10을 유지했지만 비교 방법은 모두 20 아래로 떨어졌다. 이를 “second-order 기법은 언제나 우수하다”는 일반 법칙으로 읽어서는 안 된다. 이 논문의 설정에서 가중치를 하나 고를 때마다 전체 보정과 역헤시안 갱신을 다시 적용한 결과다.

### 가중치 quantization

다음은 모든 레이어의 가중치를 asymmetric per-channel 방식으로 quantize한 논문 Table 4의 ImageNet top-1 accuracy다. `Layer-wise`는 레이어 재구성을 사용하는지, `Independent`는 앞서 압축한 레이어의 결과와 관계없이 각 레이어를 처리하는지를 나타낸다.

| Model | Method | Layer-wise | Independent | 4-bit | 3-bit | 2-bit |
| --- | --- | :---: | :---: | ---: | ---: | ---: |
| ResNet18 | AdaRound | yes | no | 69.34 | 68.37 | 63.37 |
| ResNet18 | AdaQuant | yes | no | 68.12 | 59.21 | 0.10 |
| ResNet18 | BRECQ | no | no | 69.37 | 68.47 | **64.70** |
| ResNet18 | OBQ | yes | yes | **69.56** | **68.69** | 64.04 |
| ResNet50 | AdaRound | yes | no | 75.84 | 75.14 | 71.58 |
| ResNet50 | AdaQuant | yes | no | 74.68 | 64.98 | 0.10 |
| ResNet50 | BRECQ | no | no | **75.88** | **75.32** | **72.41** |
| ResNet50 | OBQ | yes | yes | 75.72 | 75.24 | 70.71 |

OBQ는 레이어를 독립적으로 처리하면서도 4-bit와 3-bit에서 sequential method와 비슷한 결과를 낸다. 덕분에 레이어별 결과를 미리 만들어 두고 mixed-precision 설정을 빠르게 조합할 수 있다.

다만 2-bit에서는 두 모델 모두 BRECQ보다 낮다. 따라서 “OBQ가 모든 bit-width에서 가장 정확하다”고 요약하면 안 된다.

### Runtime

논문의 Appendix A.5에서 저자들의 PyTorch 구현을 NVIDIA RTX 3090 한 대로 측정한 대표 runtime은 다음과 같다.

| Model | OBQ quantization | ExactOBS unstructured pruning |
| --- | ---: | ---: |
| ResNet50 | 65 min | 64 min |
| YOLOv5s | 7 min | 6 min |
| BERT | 111 min | 103 min |

비용이 $d_{\mathrm{col}}$의 세제곱에 비례하므로, 입력 차원이 큰 몇몇 레이어가 전체 실행 시간을 좌우할 수 있다. ResNet50에서는 마지막 block의 $3\times3$ convolution 세 개가 전체 OBQ 실행 시간의 약 75%를 차지했다.

### 이 실험에서 ExactOBS가 기존 방법보다 정확한 이유

AdaPrune 같은 기존 방법은 여러 가중치를 한꺼번에 제거한 뒤 남은 가중치를 주기적으로 다시 최적화한다. ExactOBS는 이를 가중치 한 개 단위까지 세분화한다. 매번 현재 score가 가장 작은 가중치를 고르고, 즉시 전체 보정과 Lemma 1의 역헤시안 갱신을 적용한다.

이 방식은 단계 수가 많지만 Lemma 1이 각 역헤시안 갱신을 $O(d_{\mathrm{col}}^2)$로 줄이므로 계산 가능한 범위에 들어온다. 논문의 BERT 실험에서는 실행 시간이 비슷한 AdaPrune 16회 반복보다도 ExactOBS의 F1 하락 폭이 더 작았다. 이 결과는 앞에서 설명한 레이어별 이차 목적 함수의 정확한 greedy step에서 나온다.

## OBC에서 exact가 의미하는 범위

OBC의 exact를 모델 전체 task loss의 전역 최적해와 혼동하면 안 된다.

| 정확하게 처리하는 것 | 여전히 남는 근사와 제약 |
| --- | --- |
| 제곱 Frobenius 레이어 손실의 행 분해 | End-to-end task loss 대신 레이어 출력 오차 사용 |
| 선택한 이차 시스템의 OBS greedy 단계 | Greedy search는 전역 최적해를 보장하지 않음 |
| Lemma 1의 역헤시안 downdate | Calibration sample로 입력 분포 추정 |
| 가중치 고정 뒤의 closed-form 보정 | Singular Hessian에는 dampening이 필요할 수 있음 |

논문 역시 레이어별 sparsity나 quantization 제약을 전역으로 푸는 문제는 NP-hard이며, ExactOBS는 **정확한 greedy 해**를 계산한다고 선을 긋는다. 이를 풀어 쓰면 다음과 같다.

> ExactOBS는 선택한 레이어별 이차 목적 함수 안에서 OBS의 greedy 선택, 보정, 역헤시안 갱신을 Hessian에 추가 근사를 더하지 않고 계산한다.

## 수치 안정성과 구현상의 제약

이론식은 역행렬이 존재한다고 가정하지만 실제 calibration 입력에서는 $H=2XX^T$가 singular할 수 있다.

- 샘플 수가 입력 차원보다 적을 수 있다.
- 입력 feature끼리 선형 종속일 수 있다.
- Activation이 항상 0인 dead feature가 있을 수 있다.

실제 구현은 calibration augmentation으로 sample을 늘리거나 작은 diagonal dampening을 추가한다.

$$
H_{\lambda}
=
2XX^T+\lambda I.
$$

Dampening을 적용하면 역행렬은 안정되지만 계산 대상은 원래의 singular 행렬이 아니라 regularized 행렬이다. 따라서 **exact는 선택한 이차 시스템 안에서의 알고리즘 갱신을 가리킬 뿐, 수치 regularization이 필요 없다는 뜻은 아니다.**

또한 가중치를 하나씩 처리하는 loop를 그대로 GPU에 올리면 작은 kernel launch가 지나치게 많아진다. 공식 구현은 여러 행을 batch로 묶어 이 overhead를 줄인다. Convolution은 filter 하나의 가중치를 행으로 펼쳐 같은 선형 문제로 바꾼다.

## 핵심 정리

OBC가 고전적인 OBS를 현대적인 DNN 레이어에 적용할 수 있게 만든 핵심은 세 가지다.

1. $\lVert WX-W_cX\rVert_F^2$를 독립적인 행 목적 함수의 합으로 정확히 분해한다.
2. 모든 행이 공유하는 $H=2XX^T$와 초기 $H^{-1}$을 계산해 재사용한다.
3. 가중치 하나를 제거하거나 고정할 때 rank-1 downdate로 축소한 역헤시안을 $O(d_{\mathrm{col}}^2)$에 갱신한다.

ExactOBS는 이 구조로 pruning한다. OBQ는 pruning residual $w_p$를 quantization residual $w_p-Q(w_p)$로 바꿔 같은 선택·보정·역헤시안 갱신을 재사용한다.

결국 OBC의 가장 중요한 통찰은 pruning과 quantization이 완전히 별개의 heuristic이 아니라, **하나의 이차 재구성 문제에 서로 다른 가중치 제약을 적용한 두 형태**라는 점이다.

## 부록: Lemma 1 상세 증명 {#lemma-1-proof}

현재 활성 Hessian의 차원을 $n=\lvert M\rvert$이라고 하고, $e_p$를 $p$번째 성분만 1인 표준기저 열벡터라고 하자. 다음 두 벡터와 행렬을 정의한다.

$$
u
=
\frac{H^{-1}_{:,p}}{[H^{-1}]_{pp}},
\qquad
C
=
I-u e_p^T.
$$

$e_p^TH^{-1}=H^{-1}_{p,:}$이므로 $C$를 $H^{-1}$의 왼쪽에 곱하면 Lemma 1의 rank-1 downdate가 된다.

$$
\begin{aligned}
A
&=CH^{-1} \\
&=(I-u e_p^T)H^{-1} \\
&=H^{-1}
-
\frac{H^{-1}_{:,p}H^{-1}_{p,:}}
{[H^{-1}]_{pp}}.
\end{aligned}
$$

### $C$가 나타내는 행 연산

$R_i$를 $H^{-1}$의 $i$번째 행이라고 하자. 왼쪽에서 $C$를 곱하면 모든 행에 다음 연산을 적용한다.

$$
R_i'
=
R_i
-
\frac{[H^{-1}]_{ip}}
{[H^{-1}]_{pp}}
R_p,
\qquad
i=1,\ldots,n.
$$

각 행의 $p$번째 성분은 다음처럼 0이 된다.

$$
[H^{-1}]_{ip}
-
\frac{[H^{-1}]_{ip}}
{[H^{-1}]_{pp}}
[H^{-1}]_{pp}
=0.
$$

특히 $i=p$이면 $R_p'=R_p-R_p=0$이다. 따라서 $A=CH^{-1}$의 $p$번째 행과 열은 모두 0이다. $p$를 기준으로 $C$를 세 구간으로 나누면 다음 구조를 얻는다.

$$
C
=
\begin{bmatrix}
I_{p-1} & c_1 & 0 \\
0^T & 0 & 0^T \\
0 & c_2 & I_{n-p}
\end{bmatrix},
$$

$$
c_1
=
-\frac{H^{-1}_{1:p-1,p}}{[H^{-1}]_{pp}},
\qquad
c_2
=
-\frac{H^{-1}_{p+1:n,p}}{[H^{-1}]_{pp}}.
$$

$C$에서 $p$번째 행과 열을 제거하면 $C_{-p}=I_{n-1}$만 남는다. 여기서 $C$는 여러 소거 연산을 한꺼번에 나타내는 **소거 행렬(elimination matrix)**이다. $p$번째 행을 0으로 만들기 때문에 특이 행렬이며, 엄밀한 의미에서 가역인 기본 행렬(elementary matrix) 하나와는 다르다.

### $H^{-1}H=I$에 $C$ 적용하기

항등식 $H^{-1}H=I$의 양변에 왼쪽에서 $C$를 곱한다.

$$
(CH^{-1})H
=
CI
=
C.
$$

$A=CH^{-1}$로 두면 $AH=C$다. $p$번째 행과 열을 기준으로 $A$와 $H$를 블록 행렬로 쓰면 다음과 같다.

$$
A
=
\begin{bmatrix}
A_1 & 0 & A_2 \\
0^T & 0 & 0^T \\
A_3 & 0 & A_4
\end{bmatrix},
\qquad
H
=
\begin{bmatrix}
H_1 & h_1 & H_2 \\
h_3^T & h_{pp} & h_4^T \\
H_3 & h_2 & H_4
\end{bmatrix}.
$$

$A_1,A_2,A_3,A_4$와 $H_1,H_2,H_3,H_4$는 각각 $p$번째 행과 열을 제외하고 남는 네 블록이다. $h_1,h_2$는 $H$의 $p$번째 열을 위아래로 나눈 벡터이고, $h_3^T,h_4^T$는 $p$번째 행을 좌우로 나눈 벡터다.

$AH=C$에서 $p$번째 행과 열을 제거한 블록만 보자. 행렬곱에는 원래 가운데 블록을 거치는 항도 들어가지만 $A_{-p,p}=0$이므로 그 항은 사라진다. 따라서 다음 식을 얻는다.

$$
\underbrace{
\begin{bmatrix}
A_1 & A_2 \\
A_3 & A_4
\end{bmatrix}
}_{A_{-p}}
\underbrace{
\begin{bmatrix}
H_1 & H_2 \\
H_3 & H_4
\end{bmatrix}
}_{H_{-p}}
=
\underbrace{
\begin{bmatrix}
I_{p-1} & 0 \\
0 & I_{n-p}
\end{bmatrix}
}_{I_{n-1}}.
$$

즉 $A_{-p}H_{-p}=I_{n-1}$이므로 $A_{-p}=(H_{-p})^{-1}$이다. 앞에서 구한 $A=CH^{-1}$을 다시 대입하면 다음과 같이 Lemma 1을 얻는다.

$$
\boxed{
\left(
H^{-1}
-
\frac{H^{-1}_{:,p}H^{-1}_{p,:}}
{[H^{-1}]_{pp}}
\right)_{-p}
=
(H_{-p})^{-1}
}.
$$

## 참고 자료

- Elias Frantar, Sidak Pal Singh, Dan Alistarh, [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/pdf/2208.11580), NeurIPS 2022
- [IST-DASLab/OBC 공식 구현](https://github.com/IST-DASLab/OBC)
- [Optimal Brain Surgeon](/posts/optimal-brain-surgeon/)
