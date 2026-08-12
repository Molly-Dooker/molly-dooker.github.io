---
title: "Tenstorrent SDPA Prefill"
date: 2026-08-10 12:12:07 +0900
categories: [Tenstorrent, sdpa]
tags: [sdpa]
description: "Tenstorrent SDPA Prefill이 Q chunk를 코어에 배분하고 필요한 K/V를 순회해 로컬 online softmax로 출력을 완성하는 과정을 설명합니다."
math: true
render_with_liquid: false
---

## 문서 범위

- 핵심: causal Q/K chunk 관계, GQA mapping, 출력 작업 소유권, 로컬 online softmax, chunked prefix cache
- 구현 기준: `sdpa_device_operation.cpp`, `sdpa_program_factory.cpp`, Reader·Compute·Writer kernel
- 확인일: 2026-08-12

이 글은 일반 SDPA와 paged cache를 읽는 chunked prefill의 대표 경로를 다룬다. MLA, ring-distributed SDPA, streaming compute의 세부 조건, positional ABI와 CB 번호는 설명에서 제외한다.

[Flash Decode](/posts/tenstorrent-sdpa-flash-decode/)와 가장 큰 차이는 출력 소유권이다. Prefill에서는 코어 하나가 `(batch, Q head, Q chunk)` 출력 영역 하나를 처음부터 끝까지 처리한다. 필요한 K/V chunk를 모두 순회한 뒤 같은 코어에서 정규화하고 결과를 쓴다.

```text
one Q output chunk
       |
       v
assigned to one owner core
       |
       v
stream required K/V chunks
       |
       v
update local [M,L,O]
       |
       v
normalize locally and write output
```

## Prefill SDPA가 계산하는 것

Scaled Dot-Product Attention(SDPA)은 다음을 계산한다.

$$
Y
=
\operatorname{softmax}\!\left(sQK^T+A\right)V,
\qquad
s=\frac{1}{\sqrt{D}}.
$$

$D$는 Q/K head dimension이고 $A$는 causal 또는 외부 attention mask다.

Prefill은 prompt의 여러 query token을 한 호출에서 처리한다.

```text
prompt positions 0..S-1

Q: positions 0..S-1
K: positions 0..S-1
V: positions 0..S-1
```

논리적 score tensor의 shape는 `[B,Hq,Sq,Sk]`다. 구현은 이 전체 tensor를 외부 메모리에 만들지 않는다. 현재 Q/K chunk의 score block만 L1에서 계산하고, 처리한 K 범위를 Q 행별 상태로 요약한다.

Q와 K/V chunk의 token 수를 각각 $C_q$, $C_k$라고 하고, V head dimension을 $D_v$라고 하자.

| 상태 | Q 행별 의미 | 논리적 shape |
| --- | --- | --- |
| $M$ | 지금까지 본 score의 최댓값 | `[Cq]` |
| $L$ | 현재 최댓값을 기준으로 한 exponential 합 | `[Cq]` |
| $O$ | 정규화 전 weighted-value 합 | `[Cq,D_v]` |

모든 K/V chunk를 처리한 뒤 $Y=O/L$을 계산한다.

## Regular causal prefill의 shape

예제로 다음 설정을 사용하자.

```text
B              = 1
Hq             = 8
Hkv            = 2
G = Hq / Hkv   = 4
Sq = Sk        = 256 tokens
D              = 128
q_chunk_size   = 64 tokens
k_chunk_size   = 64 tokens
is_causal      = true
```

입출력의 논리적 shape는 다음과 같다.

```text
Q:      [1,8,256,128]
K:      [1,2,256,128]
V:      [1,2,256,128]
Output: [1,8,256,128]
```

Regular prefill SDPA는 전달받은 K/V를 읽지만 cache를 직접 갱신하지 않는다. 미래 decode용 cache가 필요하다면 상위 attention layer가 별도로 K/V를 저장한다.

### Causal mask를 chunk 단위로 본다

Token 단위 causal attention에서 Q position $q$는 K position `0..q`만 볼 수 있다.

```text
             K position
          0   1   2   3   ...
        +---+---+---+---+-----+
Q 0     | A | X | X | X | ... |
Q 1     | A | A | X | X | ... |
Q 2     | A | A | A | X | ... |
Q 3     | A | A | A | A | ... |
        +---+---+---+---+-----+

A = allowed
X = masked
```

Q와 K/V를 64-token chunk로 나누면 네 개씩 생긴다.

```text
Q0/K0: positions   0..63
Q1/K1: positions  64..127
Q2/K2: positions 128..191
Q3/K3: positions 192..255
```

```text
             K0   K1   K2   K3
          +----+----+----+----+
Q0        | A* | X  | X  | X  |
Q1        | A  | A* | X  | X  |
Q2        | A  | A  | A* | X  |
Q3        | A  | A  | A  | A* |
          +----+----+----+----+

A  = fully allowed chunk
A* = diagonal causal mask required
X  = future chunk skipped
```

Q chunk이 뒤에 있을수록 더 많은 K/V chunk를 읽는다.

| Q chunk | 필요한 K/V chunk |
| --- | --- |
| `Q0` | `K0/V0` |
| `Q1` | `K0/V0, K1/V1` |
| `Q2` | `K0/V0..K2/V2` |
| `Q3` | `K0/V0..K3/V3` |

Reader는 완전히 미래에 있는 K/V chunk를 공급하지 않는다. 마지막 경계 chunk 안의 future token은 additive mask로 제거한다.

### GQA는 Q head를 K/V head에 연결한다

Grouped Query Attention(GQA)에서는 여러 Q head가 같은 K/V head를 공유한다.

$$
G=\frac{H_q}{H_{kv}},
\qquad
h_{kv}=\left\lfloor\frac{h_q}{G}\right\rfloor.
$$

예제에서는 $G=4$다.

```text
KV head 0 <- Q heads 0,1,2,3
KV head 1 <- Q heads 4,5,6,7
```

같은 K/V를 읽더라도 Q head와 Q chunk가 다르면 softmax 상태와 출력도 별개다. GQA grouping factor는 reduction group 크기가 아니다.

## `(batch, Q head, Q chunk)`가 출력 작업이다

Q chunk 수를 $N_{q,\mathrm{chunk}}$라고 하면 독립 작업 수는 다음과 같다.

$$
N_{\mathrm{work}}
=
B H_q N_{q,\mathrm{chunk}}.
$$

예제에서는 Q chunk가 네 개이므로 32개 작업이 생긴다.

$$
N_{\mathrm{work}}=1\times8\times4=32.
$$

작업 `(b,h,q)`는 서로 겹치지 않는 출력 조각 하나를 소유한다.

```text
work (b,h,q)
    owns Q[b,h,q*Cq:(q+1)*Cq,:]
    maps h to one K/V head
    loops over every allowed K/V chunk
    writes Y[b,h,q*Cq:(q+1)*Cq,:]
```

Factory는 가용 코어를 batch, Q head, Q chunk 축에 배분한다. 물리 코어 하나가 여러 작업을 순차 처리할 수 있지만, 작업 하나의 K/V 범위를 여러 코어가 나누지는 않는다.

구체적인 배분을 보기 위해 이 예제에서는 program grid가 8개 코어를 제공한다고 가정한다. 이는 작업 소유권을 설명하기 위한 가정이며, 실제 코어 수가 달라지면 코어별 Q head와 Q chunk 범위도 달라진다.

```text
num_cores      = 8
batch_parallel = 1
head_parallel  = 8
q_parallel     = 1
head_per_core  = 1
q_per_core     = 4
```

이때 logical core 0은 Q head 0의 Q chunk 네 개를 다음 순서로 처리한다.

```text
logical core 0
+-----------------------------------------------------------------------+
| Q head 0                                                              |
|   Q0 -> K0/V0                         -> Y[0,h0,0:64,:]               |
|   Q1 -> K0/V0,K1/V1                   -> Y[0,h0,64:128,:]             |
|   Q3 -> K0/V0,K1/V1,K2/V2,K3/V3       -> Y[0,h0,192:256,:]            |
|   Q2 -> K0/V0,K1/V1,K2/V2             -> Y[0,h0,128:192,:]            |
+-----------------------------------------------------------------------+

logical works per active core = 1 head * 4 Q chunks = 4
K/V-chunk pairs per active core = 1 head * 10       = 10
```

`Q0,Q1,Q3,Q2`는 causal 작업량을 균형 있게 만들기 위한 balanced Q mapping 순서다. Q chunk별 K/V pair 수는 각각 1, 2, 4, 3개이므로 한 코어가 모두 10개 pair를 처리한다. 실행 순서를 섞어도 출력은 각 Q chunk의 원래 sequence 위치에 기록된다. 이 재배치는 출력 소유권을 바꾸지 않으며, 하나의 Q chunk를 여러 코어로 쪼개지도 않는다.

## 한 코어에서 online softmax를 끝낸다

Q chunk 하나와 K/V chunk 하나의 논리 shape는 다음과 같다.

```text
Q chunk        [Cq,D]
K chunk        [Ck,D]
V chunk        [Ck,Dv]
score          [Cq,Ck]
M and L        [Cq]
O              [Cq,Dv]
```

Chunk $j$의 scaled masked score를 다음처럼 두자.

$$
S_j=sQK_j^T+A_j.
$$

Chunk별 상태는 다음과 같다.

$$
\begin{aligned}
M_j &= \operatorname{rowmax}(S_j),\\
P_j &= \exp(S_j-M_j),\\
L_j &= \operatorname{rowsum}(P_j),\\
O_j &= P_jV_j.
\end{aligned}
$$

이전 K/V 범위의 상태를 $(M_a,L_a,O_a)$, 새 chunk 상태를 $(M_b,L_b,O_b)$라고 하자. 공통 최댓값과 보정 계수는 다음과 같다.

$$
\begin{aligned}
M &= \max(M_a,M_b),\\
\alpha &= \exp(M_a-M),\\
\beta &= \exp(M_b-M).
\end{aligned}
$$

분모와 분자를 같은 기준으로 옮겨 합친다.

$$
\begin{aligned}
L &= \alpha L_a+\beta L_b,\\
O &= \alpha O_a+\beta O_b.
\end{aligned}
$$

담당 코어는 필요한 K/V chunk마다 이 갱신을 반복한다. 마지막에만 $O/L$을 계산한다.

```text
state = empty

for each required K/V chunk:
    chunk_state = summarize(Q, K_chunk, V_chunk, mask)
    state = merge(state, chunk_state)

Y = state.O / state.L
```

```text
Q chunk retained
       |
       +--> K/V chunk 0 -> initialize state
       +--> K/V chunk 1 -> local merge
       +--> K/V chunk 2 -> local merge
       |
       v
normalize O/L
       |
       v
write owned output slice
```

이 상태는 다른 코어로 전달되지 않는다. Prefill에 Flash Decode식 reduction root가 없는 이유다.

## Chunked prefill은 prefix cache를 읽는다

`chunked_scaled_dot_product_attention`에서 “chunked”는 현재 prompt 일부가 이미 채워진 paged prefix cache를 참조한다는 뜻이다. K sequence를 여러 코어에 나눠 reduction한다는 의미가 아니다.

다음 상황을 보자.

```text
cached prefix       = positions 0..255
current prompt part = positions 256..383
current Sq          = 128
chunk_start_idx     = 256
logical Sk          = 384
```

현재 Q의 로컬 행 `0..127`은 absolute position `256..383`에 대응한다.

$$
q_{\mathrm{absolute}}
=
\operatorname{chunk\_start\_idx}+q_{\mathrm{local}}.
$$

이 offset을 causal 비교에 반영해야 cached prefix를 올바르게 볼 수 있다.

```text
                   cached prefix       current prompt part
K positions        0 ........ 255 | 256 .............. 383
                  +--------------+-------------------------+
Q abs 256         | all allowed  | allowed then masked     |
Q abs 383         | all allowed  | all allowed             |
                  +--------------+-------------------------+
```

현재 prompt 부분의 K/V도 SDPA 호출 전에 cache에 기록한다.

```text
cached K/V 0..255 ------+
                         |
new K/V 256..383 --------+--> updated paged cache 0..383
                                  |
                                  v
                          chunked prefill SDPA
```

실제 K/V 입력은 physical block pool과 page table이다.

```text
K/V pools:  [Nphysical,Hkv,block_size,D]
page table: [B,max_logical_blocks]
```

Reader가 logical position을 physical block으로 변환해 K/V를 순서대로 공급한다. Compute의 QK·PV와 online softmax 수식은 regular prefill과 같다.

```text
logical K/V position
        |
        v
page-table lookup
        |
        v
physical block read
        |
        v
ordered K/V stream
```

출력에는 cached prefix가 포함되지 않는다. 이번 호출의 Q row만 반환한다.

```text
Q input:  positions 256..383
K/V read: positions   0..383
Output:   positions 256..383 only
```

## Reader·Compute·Writer pipeline

Reader, Compute, Writer는 CB를 통해 동시에 진행된다.

```text
external tensors
  Q, regular K/V or paged pools and page table
                         |
                         v
+-------------------- owner core --------------------+
| Reader                                             |
|   Q chunk, mapped K/V chunks, optional mask        |
|                                                    |
| Compute                                            |
|   QK, online-softmax update, PV, normalize         |
|                                                    |
| Writer                                             |
|   constants, internal mask, owned output write     |
+----------------------------------------------------+
```

| 역할 | 주된 책임 |
| --- | --- |
| Reader | 로컬 작업을 순회하고 Q head를 K/V head에 매핑해 필요한 chunk를 공급 |
| Compute | QK·softmax·PV를 실행하고 로컬 $(M,L,O)$를 갱신한 뒤 정규화 |
| Writer | Reduction 상수와 internal mask를 만들고 완성된 Q chunk를 제 위치에 기록 |

시간 축에서는 다음 입력 이동과 현재 계산이 겹친다.

```text
time ----------------------------------------------------->

Reader   Q | K/V 0 | K/V 1 | K/V 2
Compute      calc 0 | calc 1 | calc 2 | normalize
Writer   constants and masks             | output
```

Regular과 chunked 경로의 차이는 주로 Reader가 K/V 주소와 absolute Q 위치를 구하는 방식이다. 작업 소유권과 로컬 online softmax 흐름은 같다.

## 주요 전제

대표 standard SDPA 경로를 읽을 때는 다음 조건을 먼저 확인하면 된다.

- Q/K/V는 TILE layout의 device tensor다.
- GQA에서는 $H_q$가 $H_{kv}$로 나누어떨어져야 한다.
- Q/K chunk 크기는 tile 단위에 맞춰 정한다.
- Regular causal mode에서는 Q와 K/V의 논리 sequence 길이가 같다.
- Chunked mode에서는 page table과 시작 위치가 현재 Q의 absolute 위치를 올바르게 가리켜야 한다.

External mask, sliding window, attention sink와 non-causal K/V forwarding은 Reader와 Writer의 경로를 추가하지만, 출력 작업 하나를 한 코어가 소유한다는 원칙은 바꾸지 않는다.

## Flash Decode와 비교하기

| 항목 | Prefill SDPA | Flash Decode |
| --- | --- | --- |
| Q 길이 | 여러 token | 보통 1 token |
| 독립 작업 | `(batch, Q head, Q chunk)` | `(batch, KV head)` |
| 주요 병렬화 | 서로 다른 Q 출력 영역 | 한 출력의 K/V sequence 분할 |
| 한 출력의 K/V 순회 | 담당 코어 하나가 모두 처리 | 여러 코어가 나눠 처리 |
| 코어 간 $(M,L,O)$ | 없음 | Tree reduction |
| 최종 정규화 | 각 작업 담당 코어 | Reduction root |
| Paged K/V | Chunked prefix cache | Decode cache |

두 경로 모두 전체 score matrix를 보관하지 않고 $(M,L,O)$를 갱신한다. Prefill에서는 이 상태가 담당 코어 안에 머물고, Decode에서는 여러 코어의 상태를 root로 모은다.

```text
Prefill
    K/V chunks -> one owner's [M,L,O] -> local O/L -> output

Flash Decode
    K/V chunks -> many cores' [M,L,O]
               -> tree merge -> root O/L -> output
```

## 핵심 정리

1. Prefill은 prompt의 여러 Q token을 한 호출에서 처리한다.
2. Factory는 `(batch, Q head, Q chunk)` 단위로 서로 겹치지 않는 출력 작업을 만든다.
3. 작업 하나는 담당 코어가 끝까지 처리하며 K/V sequence를 다른 코어와 나누지 않는다.
4. Reader는 GQA mapping과 causal 범위에 맞는 K/V chunk만 공급한다.
5. Compute는 score 전체 대신 로컬 $(M,L,O)$를 유지하고 마지막에 한 번 $O/L$을 계산한다.
6. Chunked prefill은 paged prefix cache와 현재 prompt 부분을 함께 읽되 현재 Q row만 출력한다.
7. Chunked mode의 `chunk_start_idx`는 로컬 Q 행을 absolute prompt 위치로 변환한다.
8. Prefill에는 Flash Decode식 inter-core state tree와 root-only normalization이 없다.

```text
partition by Q output chunks
stream K/V through one owner
normalize locally and write disjoint output
```

## 참고 자료

- Tenstorrent, [TTNN `scaled_dot_product_attention` API](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.transformer.scaled_dot_product_attention.html).
- Tenstorrent, [`sdpa.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp)와 [`sdpa.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp).
- Tenstorrent, [`sdpa_device_operation.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp).
- Tenstorrent, [`sdpa_program_factory.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp).
- Tenstorrent, [`reader_interleaved.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp).
- Tenstorrent, compute [`sdpa.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp)와 [`compute_common.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp).
- Tenstorrent, [`writer_interleaved.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp).
- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
