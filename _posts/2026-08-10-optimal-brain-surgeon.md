---
title: "Optimal Brain Surgeon"
date: 2026-08-10 09:43:58 +0900
categories: [DeepLearning, Quantization]
tags: [quantization]
description: "Optimal Brain Surgeon이 역헤시안으로 제거할 가중치와 보정량을 결정하는 과정을 수식과 MONK 실험으로 설명합니다."
math: true
render_with_liquid: false
---

## 문서 범위

- 핵심: OBS saliency, 역헤시안 기반 가중치 보정, 반복 pruning 절차
- 실험: 원 논문의 MONK 1·2·3 결과
- 후속 글: OBS를 현대적인 pruning과 quantization으로 확장한 [Optimal Brain Compression](/posts/optimal-brain-compression/)
- 확인일: 2026-08-10

Optimal Brain Surgeon(OBS)은 학습을 마친 신경망에서 중요도가 낮은 가중치를 제거하는 고전적인 pruning 기법이다. 이름은 독특하지만 묻는 바는 명확하다.

> 어느 가중치를 지워야 학습 오차가 가장 적게 늘어날까? 남은 가중치는 어떻게 조정해야 그 증가분을 최소화할 수 있을까?

OBS는 이 질문을 오차 곡면의 곡률, 즉 Hessian을 이용하는 제약 최적화 문제로 바꾼다. 단순히 작은 가중치를 고르는 데서 그치지 않는다. 하나를 제거한 뒤 **나머지 모든 가중치를 함께 보정**한다는 점이 핵심이다.

## 왜 가중치 크기만으로는 부족한가

가장 단순한 pruning은 $\lvert w_q\rvert$가 작은 가중치부터 제거한다. 값이 작으면 신경망 출력에 미치는 영향도 작으리라는 직관에 따른 방법이다.

그러나 가중치의 크기만 봐서는 그 지점의 오차 곡면이 어떤 모양인지 알 수 없다.

- 작은 가중치라도 오차가 급격히 변하는 방향에 놓여 있으면 제거 비용이 클 수 있다.
- 큰 가중치라도 다른 가중치가 그 역할을 쉽게 대신할 수 있으면 제거 비용이 작을 수 있다.
- 가중치 사이의 상관관계는 Hessian의 비대각 성분에 나타나지만, 크기만 비교해서는 이 정보를 볼 수 없다.

고전적인 세 방법의 차이를 요약하면 다음과 같다.

| 방법 | 선택 기준 | Hessian 정보 | 제거 후 보정 |
| --- | --- | --- | --- |
| Magnitude pruning | 작은 $\lvert w_q\rvert$ | 사용하지 않음 | 일반적으로 없음 |
| Optimal Brain Damage(OBD) | 2차 오차 증가량 | 대각 근사 | 전체 벡터를 직접 보정하지 않음 |
| Optimal Brain Surgeon(OBS) | 제약 조건 아래의 최소 오차 증가량 | 전체 역헤시안 | 모든 가중치를 함께 조정 |

OBS는 계산량을 더 쓰는 대신 “작은 가중치는 중요하지 않다”는 단순한 가정을 피한다.

## 오차의 2차 근사

신경망의 모든 가중치를 1차원 벡터 $\mathbf{w}$로 펼치고, 변화량을 $\delta\mathbf{w}$라고 하자. 학습 오차를 $E(\mathbf{w})$라 하면 Hessian은 다음과 같다.

$$
\mathbf{H}
\equiv
\frac{\partial^2 E}{\partial \mathbf{w}^2}
$$

$\mathbf{w}$ 근처의 오차 변화를 Taylor 전개하면

$$
\delta E
=
\left(\frac{\partial E}{\partial \mathbf{w}}\right)^T
\delta\mathbf{w}
+
\frac{1}{2}
\delta\mathbf{w}^T
\mathbf{H}
\delta\mathbf{w}
+
O\!\left(\lVert\delta\mathbf{w}\rVert^3\right)
\tag{1}
$$

이 된다.

OBS는 신경망이 국소 최솟값에 충분히 가깝게 학습됐다고 가정한다. 이때 gradient는 거의 0이다.

$$
\frac{\partial E}{\partial \mathbf{w}} \approx 0
$$

여기에 $\delta\mathbf{w}$가 충분히 작다는 가정까지 더해 3차 이상의 항을 버리면, 오차 변화는 이차 형식으로 근사할 수 있다.

$$
\delta E
\approx
\frac{1}{2}
\delta\mathbf{w}^T
\mathbf{H}
\delta\mathbf{w}
\tag{2}
$$

이 식이 OBS 유도의 출발점이다.

## 가중치 제거를 제약 조건으로 표현하기

$q$번째 가중치 $w_q$를 제거한다는 말은 갱신한 뒤 그 값이 0이 된다는 뜻이다.

$$
w_q + \delta w_q = 0
$$

$q$번째 좌표만 고르는 단위 벡터를 $\mathbf{e}_q$라고 하면 같은 제약 조건을 벡터 형태로 쓸 수 있다.

$$
\mathbf{e}_q^T\delta\mathbf{w} + w_q = 0
\tag{3}
$$

따라서 특정 $w_q$를 제거할 때 풀어야 하는 문제는 다음과 같다.

$$
L_q
=
\min_{\delta\mathbf{w}}
\frac{1}{2}
\delta\mathbf{w}^T
\mathbf{H}
\delta\mathbf{w}
\quad
\text{subject to}
\quad
\mathbf{e}_q^T\delta\mathbf{w} + w_q = 0
\tag{4}
$$

$L_q$는 $w_q$를 없애고 나머지 가중치를 최적으로 조정해도 피할 수 없는 최소 오차 증가량이다. OBS에서는 이를 가중치 $q$의 **saliency**라고 부른다.

## Lagrange multiplier로 최적 보정량 구하기

이차 목적 함수에 제약 조건이 붙은 문제이므로 Lagrangian을 세운다.

$$
\mathcal{L}
=
\frac{1}{2}
\delta\mathbf{w}^T
\mathbf{H}
\delta\mathbf{w}
+
\lambda
\left(
\mathbf{e}_q^T\delta\mathbf{w} + w_q
\right)
\tag{5}
$$

### 1. $\delta\mathbf{w}$에 대해 미분하기

Stationary condition을 적용하면

$$
\nabla_{\delta\mathbf{w}}\mathcal{L}
=
\mathbf{H}\delta\mathbf{w}
+
\lambda\mathbf{e}_q
= 0
$$

이므로

$$
\delta\mathbf{w}
=
-\lambda
\mathbf{H}^{-1}
\mathbf{e}_q
\tag{6}
$$

를 얻는다.

### 2. 제약 조건으로 $\lambda$ 구하기

식 (6)을 pruning 제약 조건에 대입한다.

$$
\mathbf{e}_q^T
\left(
-\lambda\mathbf{H}^{-1}\mathbf{e}_q
\right)
+ w_q
= 0
$$

정리하면

$$
\lambda
=
\frac{w_q}
{\mathbf{e}_q^T\mathbf{H}^{-1}\mathbf{e}_q}
$$

이다. $\mathbf{e}_q$는 $q$번째 좌표만 고르므로 분모는 역헤시안의 $q$번째 대각 원소와 같다.

$$
\mathbf{e}_q^T
\mathbf{H}^{-1}
\mathbf{e}_q
=
[\mathbf{H}^{-1}]_{qq}
$$

따라서

$$
\lambda
=
\frac{w_q}
{[\mathbf{H}^{-1}]_{qq}}
\tag{7}
$$

가 된다.

### 3. 최적 가중치 보정

식 (7)을 식 (6)에 대입하면 최적 보정량은 다음과 같다.

$$
\boxed{
\delta\mathbf{w}
=
-\frac{w_q}
{[\mathbf{H}^{-1}]_{qq}}
\mathbf{H}^{-1}
\mathbf{e}_q
}
\tag{8}
$$

이 벡터의 $q$번째 성분은 정확히 $-w_q$이므로 선택한 가중치가 0이 된다. 나머지 성분은 제거로 늘어나는 오차가 가장 작아지도록 다른 가중치를 함께 조정한다.

### 4. OBS saliency

식 (8)을 이차 오차식에 다시 대입하면

$$
\boxed{
L_q
=
\frac{1}{2}
\frac{w_q^2}
{[\mathbf{H}^{-1}]_{qq}}
}
\tag{9}
$$

를 얻는다. OBS는 모든 후보의 $L_q$를 계산해 값이 가장 작은 가중치를 고른다.

$$
q^*
=
\operatorname*{arg\,min}_q
L_q
\tag{10}
$$

## Saliency 식을 어떻게 읽어야 하는가

식 (9)에는 가중치 크기와 곡률 정보가 함께 들어 있다.

$$
L_q
=
\frac{w_q^2}
{2[\mathbf{H}^{-1}]_{qq}}
$$

- $w_q^2$가 작으면 대체로 제거 비용도 작아진다.
- $[\mathbf{H}^{-1}]_{qq}$가 크면 그 방향이 비교적 평평하거나 다른 가중치로 보상하기 쉬워 saliency가 작아진다.
- $\mathbf{H}^{-1}\mathbf{e}_q$는 $q$번째 가중치를 제거한 영향을 다른 좌표에 어떻게 나눠 보상할지 정한다.

가중치 크기는 여전히 식에 들어가지만, **크기만으로 결론을 내리지는 않는다**. 크기가 같은 두 가중치도 역헤시안 구조에 따라 saliency와 보정량이 달라질 수 있다.

## OBS pruning 절차

전체 알고리즘은 다음과 같이 반복된다.

1. 충분히 큰 신경망을 국소 최솟값에 가깝게 학습한다.
2. 역헤시안 $\mathbf{H}^{-1}$를 계산한다.
3. 모든 가중치에 대해 $L_q$를 계산한다.
4. $q^*=\operatorname*{arg\,min}_q L_q$를 선택한다.
5. $L_{q^*}$가 허용할 만한 오차 증가인지 확인한다.
6. 식 (8)의 $\delta\mathbf{w}$를 적용해 $w_{q^*}$를 0으로 만들고 나머지 가중치를 보정한다.
7. 역헤시안을 갱신한 뒤 다음 가중치를 고른다.
8. 더 이상 안전하게 제거할 가중치가 없으면 끝낸다.

원 논문은 $P$개의 학습 패턴에 대한 제곱 오차를 다음과 같이 정의한다.

$$
E
=
\frac{1}{2P}
\sum_{k=1}^{P}
\left(
\mathbf{t}^{[k]}-\mathbf{o}^{[k]}
\right)^T
\left(
\mathbf{t}^{[k]}-\mathbf{o}^{[k]}
\right)
\tag{11}
$$

여기서 $\mathbf{t}^{[k]}$는 목표값이고, $\mathbf{o}^{[k]}$는 신경망 출력이다. 후보의 saliency가 현재 오차보다 매우 작다면, 즉 $L_q \ll E$라면 그 가중치를 제거해도 영향이 작다고 볼 수 있다.

```text
train to a local minimum
compute H^-1

repeat:
    L_q = 0.5 * w_q^2 / [H^-1]_qq  for every q
    q_star = argmin_q L_q

    if L[q_star] is too large:
        stop

    delta_w = -w[q_star] / H_inv[q_star, q_star]
              * H_inv * e[q_star]
    w       = w + delta_w
    update H^-1
```

## MONK 실험 결과

OBS 논문은 BPWD(Backpropagation with Weight Decay)로 학습한 신경망을 세 가지 MONK 분류 문제에서 pruning했다. 남은 가중치 수와 정확도는 다음과 같다.

| Problem | Method | Training accuracy (%) | Testing accuracy (%) | Weights | Weight reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| MONK 1 | BPWD | 100 | 100 | 58 | - |
| MONK 1 | OBS | 100 | 100 | 14 | 75.9% |
| MONK 2 | BPWD | 100 | 100 | 39 | - |
| MONK 2 | OBS | 100 | 100 | 15 | 61.5% |
| MONK 3 | BPWD | 93.4 | 97.2 | 39 | - |
| MONK 3 | OBS | 93.4 | 97.2 | 4 | 89.7% |

세 문제 모두 OBS는 BPWD와 같은 학습·테스트 정확도를 유지하면서 가중치 수를 약 62~90% 줄였다.

특히 잡음이 섞인 MONK 3에서는 39개 중 가중치 4개만 남기고도 학습 정확도 `93.4%`, 테스트 정확도 `97.2%`를 유지했다. 작은 값을 기계적으로 없애지 않고, 제거에 따른 오차 증가를 예측한 뒤 나머지 가중치를 보정한 결과다.

## 계산 비용과 가정

OBS의 이론은 간결하지만 전체 역헤시안을 그대로 다루려면 적지 않은 비용이 든다.

### Local minimum 가정

Gradient 항을 버리려면 학습 결과가 국소 최솟값에 충분히 가까워야 한다. Gradient를 무시할 수 없을 만큼 크다면 식 (2)의 근사부터 성립하지 않는다.

### Local quadratic approximation

Taylor 전개의 3차 이상을 무시하므로 한 번에 제거하고 보정하는 양이 충분히 작아야 한다. Pruning이 진행되면서 곡률이 크게 달라지면 Hessian도 다시 갱신해야 한다.

### 전체 역헤시안 비용

가중치가 $N$개라면 밀집 $\mathbf{H}^{-1}$을 저장하는 데 $O(N^2)$ 메모리가 필요하고, 단순한 역행렬 계산에는 $O(N^3)$ 연산이 든다. 원래 OBS도 단계마다 역행렬을 처음부터 구하지 않도록 재귀 갱신을 사용하지만, 현대의 대규모 모델에 그대로 적용하기에는 여전히 부담스럽다.

이후 연구는 layer-wise approximation, block structure, low-rank approximation, Fisher information, Woodbury identity 등을 이용해 second-order 정보를 더 적은 비용으로 근사한다.

## 핵심 정리

1. 가중치 크기만으로는 제거 뒤의 오차 증가량을 정확히 판단하기 어렵다.
2. OBS는 국소 최솟값 부근의 오차를 Hessian 기반 이차 형식으로 근사한다.
3. $w_q \to 0$을 제약 조건으로 두고 오차를 최소화하는 보정량을 구한다.
4. Saliency는 $L_q = \frac{1}{2}\frac{w_q^2}{[\mathbf{H}^{-1}]_{qq}}$이다.
5. 제거할 가중치뿐 아니라 남은 모든 가중치의 보정량도 함께 계산한다.
6. 전체 역헤시안은 정확하지만 비싸므로, 현대 기법은 이 구조를 근사하거나 block 단위로 적용한다.
7. 같은 제약 오차 보정 관점은 pruning뿐 아니라 post-training quantization으로도 이어진다.

## 참고 자료

- Hassibi and Stork, [Second Order Derivatives for Network Pruning: Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html), NIPS 1992
- Hassibi, Stork, and Wolff, [Optimal Brain Surgeon: Extensions and Performance Comparisons](https://proceedings.neurips.cc/paper/1993/hash/b056eb1587586b71e2da9acfe4fbd19e-Abstract.html), NIPS 1993
- LeCun, Denker, and Solla, [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html), NIPS 1989
- Frantar and Alistarh, [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html), NeurIPS 2022
