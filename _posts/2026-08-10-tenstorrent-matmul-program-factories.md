---
title: "Tenstorrent Matmul Program Factory: 2D, Minimal, DRAM Sharded 비교"
date: 2026-08-10 15:52:54 +0900
categories: [Tenstorrent, matmul]
tags: [matmul]
description: "Tenstorrent matmul의 2D, Minimal, DRAM Sharded program factory를 코어 배치와 입력 이동, 적용 조건 중심으로 비교합니다."
math: true
render_with_liquid: false
---

## 문서 범위

- 핵심: 출력 타일의 코어 분배, Reader·Compute·Writer pipeline, factory별 입력 이동 방식, kernel 인자와 program cache 재사용
- 다루는 factory: 2D Multicast, Minimal Matmul, DRAM Sharded
- 기준: Tenstorrent `tt-metal`의 matmul factory와 data-movement·compute kernel
- 확인일: 2026-08-12

TTNN matmul에는 1D Multicast와 Batched DRAM Sharded 경로도 있다. 이 글은 세 경로의 차이를 선명하게 설명하는 데 집중하고, `transpose_mcast`, fused operation, quantization과 layout별 예외 같은 세부 분기는 다루지 않는다.

세 factory는 같은 행렬곱을 계산하지만 입력을 나누고 전달하는 방식이 다르다.

> 2D는 A와 B를 서로 직교하는 방향으로 multicast한다. Minimal은 같은 두 축을 이웃 코어 chain으로 연결한다. DRAM Sharded는 B의 N 방향 조각을 DRAM bank에 고정하고 bank별 worker가 직접 읽는다.

## Program factory가 결정하는 것

행렬곱 자체는 다음 식으로 나타낼 수 있다.

$$
C_{m,n}
=
\sum_{k=0}^{K-1} A_{m,k}B_{k,n},
\qquad
A\in\mathbb{R}^{M\times K},
B\in\mathbb{R}^{K\times N}.
$$

Device에서 이 식을 실행하려면 계산식 밖의 문제도 정해야 한다.

1. 출력 $C$의 어느 영역을 어느 Tensix 코어가 맡는가?
2. 각 코어가 A와 B 블록을 어디서 읽는가?
3. 여러 코어가 같은 입력을 재사용할 때 어떤 경로로 전달하는가?
4. 입력 이동, K 방향 누적, 출력 쓰기를 어떻게 겹치는가?

Program config는 그리드와 블록 크기 같은 실행 조건을 제공한다. Program factory는 이를 바탕으로 코어 역할, circular buffer(CB), semaphore, kernel과 인자를 묶어 device program을 만든다.

```text
Tensor shape, layout, and config
                 |
                 v
          Program factory
       +---------+----------+
       | core roles         |
       | CBs and semaphores |
       | kernels and args   |
       +---------+----------+
                 |
                 v
          Device program
```

Factory와 kernel의 역할은 다음처럼 구분할 수 있다.

| 구성 요소 | 주된 역할 |
| --- | --- |
| Program config | 그리드와 블록 크기, factory 종류 지정 |
| Program factory | 코어 배치, 통신 topology, CB와 kernel 구성 |
| Data-movement kernel | DRAM·L1 읽기, 코어 간 전달, 출력 쓰기 |
| Compute kernel | 입력 CB를 소비해 블록 행렬곱과 K 누적 수행 |

### Program 생성 시 고정하는 값과 실행 전에 바꾸는 값

Program factory는 연산 구조에 맞춰 kernel과 program을 만든다. Compile-time 인자는 kernel binary에 반영되고, 코어 매핑과 CB 구성은 program에 고정된다. Runtime 인자는 kernel을 실행하기 직전에 설정한다.

| 구분 | 대표적인 값 | 다음 호출에서 바뀌면 |
| --- | --- | --- |
| Compile-time 인자와 program 구조 | Dtype, shape, memory layout, tile·block 크기, 코어 매핑, CB 구성 | Cache miss 후 새 program과 kernel을 만든다. |
| Runtime 인자 | Buffer address, 코어별 시작 tile과 처리량, offset | 값을 갱신하고 cached program을 재사용할 수 있다. |

Program cache가 활성화되어 있으면 matmul을 호출할 때 먼저 dtype, shape, layout, program config 등으로 cache key를 계산한다. Cache hit이면 기존 kernel·코어 매핑·CB 구성을 재사용하고 `override_runtime_arguments()`로 주소 같은 runtime 인자만 갱신한다. Cache miss이면 program 구조를 다시 정하고 kernel을 컴파일한 뒤 runtime 인자를 설정한다.

```text
Matmul call
    |
    v
Build cache key
(dtype / shape / layout / program config)
    |
    +-- cache hit
    |      reuse kernel / core mapping / CB config
    |      update runtime args (address, ...)
    |      |
    |      v
    |    execute
    |
    +-- cache miss
           choose core mapping / CB config
           create and compile kernels
           set runtime args
           cache program
           |
           v
         execute
```

Tensor의 실제 원소값은 runtime 인자로 넘기지 않는다. Runtime 인자에는 buffer address가 들어가고, kernel은 그 주소에서 데이터를 읽는다.

## 세 factory에 공통인 실행 모델

### 행렬을 타일과 블록으로 나눈다

TTNN matmul은 일반적으로 TILE layout을 사용한다. 기본 full tile은 $32\times32$ 원소지만, factory는 원소보다 타일 수를 중심으로 작업을 나눈다. 타일 높이와 너비를 각각 $T_H$, $T_W$라고 하면 다음과 같이 쓸 수 있다.

$$
M_t=\left\lceil\frac{M}{T_H}\right\rceil,
\qquad
K_t=\left\lceil\frac{K}{T_W}\right\rceil,
\qquad
N_t=\left\lceil\frac{N}{T_W}\right\rceil.
$$

```text
A tiles [Mt x Kt]  x  B tiles [Kt x Nt]  =  C tiles [Mt x Nt]

output tile (m, n)
    = sum over k of A tile (m, k) x B tile (k, n)
```

M과 N은 출력 공간을 나누는 축이고 K는 부분곱을 더하는 reduction 축이다. 각 코어는 M·N의 일부를 맡지만, 자기 출력을 완성하려면 K 블록 전체를 순회해야 한다.

코어 내부에서도 출력을 한 번에 계산하지 않는다.

```text
per-core output
       |
       v
  output block
       |
       v
 output subblock  ---> held in Dst registers

K axis: block 0 -> block 1 -> ... -> last block
```

블록을 키우면 반복 횟수를 줄일 수 있지만 입력 CB가 더 많은 L1을 차지한다. 서브블록은 Dst register 용량에 맞아야 한다. 따라서 큰 블록이 항상 빠른 것은 아니다.

### Reader·Compute·Writer는 CB로 연결된다

Reader는 A와 B 블록을 CB에 넣고, Compute는 두 입력을 소비해 K 방향으로 누적한다. 완성된 결과는 출력 CB를 거쳐 Writer가 대상 텐서에 기록한다.

```text
input A --> Reader A --> CB A --+
                                |
                                +--> Compute --> output CB --> Writer
                                |
input B --> Reader B --> CB B --+
```

CB는 코어 L1에 있는 FIFO이자 생산자와 소비자의 동기화 경계다. 두 블록 분량의 입력 CB를 사용하면 다음 입력 이동과 현재 계산을 겹칠 수 있다.

```text
time ------------------------------------------------------>

Reader   fill block 0 | fill block 1 | fill block 2
Compute               | compute 0    | compute 1
Writer                               | write output 0
```

Reader·Compute·Writer는 논리적인 역할이다. 한 data-movement kernel이 입력을 읽은 뒤 출력까지 쓰기도 하므로, 세 역할이 항상 별도 소스 파일로 나뉘는 것은 아니다.

## 세 factory 비교

| 항목 | 2D Multicast | Minimal Matmul | DRAM Sharded |
| --- | --- | --- | --- |
| 출력 분할 | M·N 2D 그리드 | M·N 2D 그리드 | DRAM bank별 N 조각 |
| A 이동 | 행 방향 multicast | 행 방향 이웃 코어 chain | L1 shard sender의 multicast |
| B 이동 | 열 방향 multicast | 열 방향 이웃 코어 chain | 담당 DRAM bank에서 직접 읽기 |
| 코어 배치 | 직사각형 compute 그리드 | 직사각형 compute 그리드 | A 저장 코어와 bank 작업 코어의 조합 |
| 출력 | 각 코어가 담당 영역 기록 | 각 코어가 담당 영역 기록 | 작업 코어 결과를 출력 L1 그리드로 reshard |
| 두드러진 특성 | 빠른 multicast 전달 | padding과 부분 블록 허용 | DRAM bank 병렬성 |
| 주요 비용 | 블록·layout 제약 | chain의 fill/drain과 padding | 강한 layout 제약과 reshard |

## 2D Multicast Factory

### A는 행, B는 열 방향으로 multicast한다

2D factory의 config는 `MatmulMultiCoreReuseMultiCastProgramConfig`다. 기본 방향에서는 그리드의 x축이 출력 N을, y축이 출력 M을 나눈다.

- 각 행의 왼쪽 코어가 A sender다.
- 각 열의 위쪽 코어가 B sender다.
- 왼쪽 위 코어는 두 sender 역할을 모두 맡는다.

```text
                         N split
              x=0          x=1          x=2
          +------------+------------+------------+
y=0       | A+B sender | B sender   | B sender   |
          +------------+------------+------------+
y=1       | A sender   | receiver   | receiver   |
          +------------+------------+------------+
y=2       | A sender   | receiver   | receiver   |
          +------------+------------+------------+
                         M split
```

A의 같은 M 블록은 한 행의 여러 N 작업 코어가 재사용한다. B의 같은 N 블록은 한 열의 여러 M 작업 코어가 재사용한다. 2D factory는 이 재사용 방향에 맞춰 A를 행으로, B를 열로 보낸다.

### K 블록마다 두 입력을 multicast한다

K 블록 하나의 흐름은 다음과 같다.

1. A sender와 B sender가 필요한 입력 블록을 읽거나 로컬 shard에서 준비한다.
2. Receiver가 CB 공간을 준비하면 각 sender가 담당 행 또는 열로 multicast한다.
3. 모든 작업 코어가 로컬 A·B CB를 소비해 자기 C 블록에 부분곱을 누적한다.
4. K 반복이 끝나면 B data-movement 경로가 Writer 역할을 맡아 결과를 기록한다.

```text
A source -- row multicast ------+
                                |
                                +--> local Compute --> output
                                |
B source -- column multicast ---+
```

Multicast는 하나의 sender가 같은 블록을 여러 receiver에 전달하는 일대다 구조다. 각 receiver가 같은 입력을 DRAM에서 따로 읽는 일을 피하고, 한 행이나 열의 코어를 함께 깨울 수 있다.

다음 그림은 입력 하나의 sender와 receiver 사이의 동기화를 단순화한 개념도다. 실제 NoC 도착 시점은 물리적 거리와 혼잡에 따라 달라질 수 있다.

```text
R1 ready --+
R2 ready --+--> Sender waits for all
R3 ready --+             |
                          v
               multicast block + VALID
                   +-----+-----+
                   v     v     v
                  R1    R2    R3
```

각 receiver는 CB 공간을 확보한 뒤 sender에 준비 신호를 보낸다. Sender는 대상 receiver가 모두 준비되면 목적지 범위에 `async_write_multicast`를 한 번 발행한다. NoC가 블록을 각 receiver에 복제해 전달한 뒤 `VALID` 신호도 multicast한다. Receiver는 전체 블록과 `VALID`를 확인한 다음 로컬 CB에 블록을 공개한다. Compute는 A·B 블록이 모두 공개되면 해당 K 블록의 부분곱을 시작한다.

### 2D가 맞는 경우

2D는 M과 N을 모두 나눌 만큼 출력이 크고, shape와 블록 크기가 직사각형 그리드에 잘 맞을 때 자연스럽다. 코어당 계산이 짧아 chain 전달 시간을 숨기기 어려운 경우에도 multicast가 유리할 수 있다.

다만 `per_core_M`, `per_core_N`, K 블록, 출력 블록과 서브블록 사이에는 나눗셈과 layout 제약이 있다. `transpose_mcast`나 sharded 입력을 사용하면 sender 방향과 코어 집합도 달라질 수 있으므로, 실제 config의 검증 조건을 함께 확인해야 한다.

## Minimal Matmul Factory

### 출력은 2D로 나누고 입력은 이웃으로 전달한다

`MinimalMatmulConfig`는 출력의 M·N을 2D 그리드에 나누는 점은 2D factory와 같지만, 입력을 한 번에 multicast하지 않고 각 행과 열의 이웃 코어를 차례로 지난다.

```text
A chain, one row

DRAM A --> Injector --> Core 1 --> Core 2 --> Sink
               |           |          |         |
               v           v          v         v
            Compute     Compute    Compute   Compute

B chain, one column

             DRAM B
                |
                v
             Injector
                |
                v
              Core 1
                |
                v
               Sink
```

Injector가 아닌 코어는 이전 코어에 준비 신호를 보낸 뒤 블록을 받는다. 받은 블록은 로컬 CB에 먼저 공개하고, 다음 코어가 준비되면 같은 블록을 전달한다. 현재 코어의 계산과 다음 코어로의 전달이 겹치면서 시작 시점이 조금씩 밀린 wavefront가 형성된다.

다음 그림은 입력 블록 하나가 Core 0에서 Core 1로 전달되는 동안의 실행을 단순화한 개념도다. 막대 길이는 실제 측정값이 아니며, 실제 Compute는 A와 B가 모두 도착한 뒤 시작한다.

```text
Long compute

time              ---------------------------------------->

Core 0 compute    [----------]
Forward 0 -> 1    [---]
Core 1 compute         [----------]

Short compute

time              ---------------------------------------->

Core 0 compute    [-]
Forward 0 -> 1    [---]
Core 1 compute         [-]
```

두 경우 모두 Core 1은 전달이 끝난 뒤 계산을 시작한다. 코어당 계산이 길면 전달이 Core 0의 계산 시간과 겹치지만, 계산이 짧으면 Core 0의 계산이 먼저 끝나 전달 지연이 그대로 드러난다. 비동기 전송은 이런 겹침을 허용할 뿐 전송 시간과 chain의 fill/drain을 없애지는 않는다. 출력 쓰기도 같은 원리로 다음 계산과 겹칠 때만 비용을 숨길 수 있으며, 마지막 출력의 drain은 남는다.

입력 하나의 전달 경로만 놓고 두 방식을 요약하면 다음과 같다.

- **2D:** 한 sender → 여러 receiver. Receiver는 받은 입력을 다시 전달하지 않고 로컬 계산에 사용한다.
- **Minimal:** sender → 이웃 → 다음 이웃. 각 코어는 받은 입력을 로컬 계산에 공개한 뒤, 계산과 다음 코어로의 전달을 겹친다.

### Padding으로 그리드 나눗셈 제약을 줄인다

M과 N 병렬화에 사용하는 코어 수를 각각 $G_M$, $G_N$이라고 하자. Minimal factory는 M과 N 타일 수를 각 코어 수의 배수로, K 타일 수를 K 블록 크기의 배수로 올린다.

$$
\begin{aligned}
M_t' &= \operatorname{round\_up}(M_t, G_M),\\
N_t' &= \operatorname{round\_up}(N_t, G_N),\\
K_t' &= \operatorname{round\_up}(K_t, K_{\text{block}}).
\end{aligned}
$$

마지막 M·N 블록은 부분 블록이 될 수 있다. Data-movement kernel과 Writer는 유효한 범위만 읽고 기록하지만, 일부 handshake와 계산은 padding 영역에도 필요하다. 나눗셈 제약이 느슨한 대신 padding이 무료는 아니다.

Compute는 다른 factory와 마찬가지로 A·B 블록을 소비하고 K가 끝날 때까지 intermediate CB에 부분합을 유지한다. 출력 Writer는 읽을 입력이 작은 data-movement kernel에 붙으며, 여러 코어의 쓰기가 한 시점에 몰리지 않도록 쓰기 시점을 분산한다.

### Minimal Matmul이 맞는 경우

Minimal은 다음 조건에서 후보가 된다.

- M·N 타일 수가 그리드에 정확히 나뉘지 않아 부분 블록이 필요한 경우
- 더 큰 로컬 블록으로 반복 입력 읽기를 줄일 수 있는 경우
- 코어당 계산이 길어 이웃 코어 전달과 출력 쓰기를 계산 뒤에 숨길 여지가 있는 경우

반대로 chain에는 코어 간 handshake와 injector-to-sink fill/drain 시간이 든다. Padding 비율이 크거나 코어당 계산이 짧으면 이 비용이 그대로 드러날 수 있다. 따라서 Minimal이라는 이름을 계산량이 적다는 뜻으로 해석하면 안 된다. 유효 출력의 행렬곱은 같고, 차이는 주로 데이터 이동과 블록 schedule에서 생긴다.

## DRAM Sharded Factory

### 작은 M과 DRAM width-sharded B를 위한 전용 경로다

`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`는 세 factory 중 memory layout 계약이 가장 강하다.

| 텐서 | 대표 위치와 layout | 분할 의미 |
| --- | --- | --- |
| A, `in0` | L1 `WIDTH_SHARDED` | K 방향 shard를 L1 코어에 배치 |
| B, `in1` | DRAM `WIDTH_SHARDED` | N 방향 shard를 DRAM bank에 배치 |
| C, `output` | L1 `WIDTH_SHARDED` | N 방향 출력 shard를 L1 코어에 배치 |

이 경로는 M과 `per_core_M`이 한 타일 높이인 경우를 전제로 한다. 기본 full tile에서는 최대 32행이다. Decode처럼 M이 작고 N이 큰 matmul이 대표적인 대상이다.

### DRAM bank가 작업 코어를 정한다

2D와 Minimal은 먼저 직사각형 compute 그리드를 정한다. DRAM Sharded는 B shard가 놓인 DRAM bank마다 읽기를 담당할 작업 코어를 고른다. A shard를 보관한 L1 코어와 작업 코어가 겹치면 한 코어가 sender와 compute 역할을 함께 맡을 수 있다.

```text
A in L1, sharded over K

 Storage 0       Storage 1       Storage 2
  A[:, K0]        A[:, K1]        A[:, K2]
      +---------------+---------------+
                      |
                      v
              multicast to workers

B in DRAM, sharded over N

   Bank 0           Bank 1           Bank 2
   B[:, N0]         B[:, N1]         B[:, N2]
      |                |                |
      v                v                v
  Worker 0         Worker 1         Worker 2
   C[:, N0]         C[:, N1]         C[:, N2]
```

K 블록 하나에서는 다음 동작이 겹친다.

1. 현재 A shard를 가진 sender가 작업 코어에 A 블록을 multicast한다.
2. 각 작업 코어는 자신에게 지정된 DRAM bank에서 같은 K 구간의 B 조각을 읽는다.
3. Compute가 두 입력 CB를 소비해 작업 코어가 맡은 N 조각의 부분합을 갱신한다.
4. K 반복이 끝나면 작업 코어의 결과를 최종 출력 저장 코어의 L1로 보낸다.

DRAM bank와 가까운 작업 코어 집합은 출력 shard를 보관할 L1 코어 집합과 다를 수 있다. 이때 Writer는 작업 코어의 결과를 하나 이상의 출력 저장 코어로 NoC write한다. 이 마지막 이동이 출력 reshard다.

```text
worker output CB --> NoC write --> output shard in L1
```

### DRAM Sharded가 맞는 경우

DRAM Sharded는 작은 M에서 2D M 분할로 충분한 병렬성을 얻기 어렵고, 넓은 N의 B를 여러 DRAM bank에서 병렬로 읽고 싶을 때 유용하다.

대신 A·B·출력의 위치와 sharding 방향이 정해져 있고 M도 한 타일 높이로 제한된다. A multicast와 마지막 출력 reshard 비용도 남는다. 따라서 2D보다 발전된 일반 경로가 아니라, 특정 shape와 memory placement를 위한 별도 경로로 봐야 한다.

## Factory를 고를 때 확인할 것

Factory 선택은 sequence length나 행렬 크기 하나로 결정할 수 없다.

| 확인할 질문 | 우선 검토할 경로 |
| --- | --- |
| B가 DRAM `WIDTH_SHARDED`이고 M이 한 타일인가? | DRAM Sharded |
| M·N이 2D 그리드와 블록에 자연스럽게 맞는가? | 2D Multicast |
| 부분 블록이 필요하거나 더 유연한 블록 구성이 유리한가? | Minimal Matmul |
| 코어당 계산이 짧아 chain 지연 시간을 숨기기 어려운가? | 2D Multicast |
| 코어당 계산이 길고 padding 비율이 낮은가? | Minimal Matmul도 함께 측정 |

2D와 Minimal의 유효 출력과 MAC 연산 수는 같다. 성능 차이는 multicast와 chain의 지연 시간, 블록 크기, 반복 입력 읽기, padding, L1 CB 사용량, 출력 쓰기 정체에서 나온다. DRAM Sharded도 bank 병렬성만 보고 선택하지 말고 마지막 reshard를 포함한 전체 kernel time을 측정해야 한다.

```text
Input layout and shape
          |
          +-- B is DRAM width-sharded and M is one tile? --> DRAM Sharded
          |
          +-- regular 2D blocks and short per-core work? --> 2D Multicast
          |
          +-- partial blocks or long per-core work? -----> Compare Minimal and 2D
```

## 핵심 정리

1. Program factory는 행렬곱 식이 아니라 코어 역할, 입력 이동, CB와 kernel 배치를 결정한다.
2. 세 factory 모두 담당 출력 영역을 K 블록 방향으로 누적하지만, 입력을 공급하는 topology가 다르다.
3. 2D는 A와 B를 행·열 방향으로 multicast한다.
4. Minimal은 입력을 이웃 코어로 전달하고 padding과 부분 블록으로 그리드 나눗셈 제약을 줄인다.
5. DRAM Sharded는 작은 M에서 bank-local B 읽기를 병렬화하고 결과를 출력 L1 그리드로 reshard한다.
6. Compile-time 인자는 kernel binary를 특화하고, runtime 인자는 같은 binary에서 호출별·코어별 값을 바꾼다. Program cache hit에서는 기존 구성을 유지하고 달라진 주소만 갱신한다.
7. 실제 factory 선택에서는 shape뿐 아니라 블록, padding, L1, NoC와 DRAM traffic을 함께 측정해야 한다.

## 참고 자료

- Tenstorrent, [Reader·Compute·Writer pipeline](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html).
- Tenstorrent, [Circular Buffer API](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/host_apis/buffers/CircularBuffers.html).
- Tenstorrent, [compile-time과 runtime kernel 인자의 선택](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html), [`get_compile_time_arg_val()`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/kernel_args/get_compile_time_arg_val.html)과 [`get_arg_val()`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/kernel_args/get_arg_val.html) API.
- Tenstorrent, [`SetRuntimeArgs()`·`GetRuntimeArgs()`와 dynamic CB address API](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/api/tt-metalium/host_api.hpp#L435-L555).
- Tenstorrent, [`ttnn.matmul` API와 program config별 memory support](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.matmul.html).
- Tenstorrent, [matmul program config type 정의](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config_types.hpp).
- Tenstorrent, [2D Multicast program factory](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp), [A sender kernel](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp), [A receiver kernel](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp), [B sender·Writer kernel](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp), [B receiver·Writer kernel](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp).
- Tenstorrent, [Minimal Matmul program factory](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp), [A data movement](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp), [B data movement·Writer](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp).
- Tenstorrent, [DRAM Sharded program factory](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp), [A multicast kernel](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp), [B read·output reshard kernel](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp).
