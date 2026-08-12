---
title: "Tenstorrent SDPA Flash Decode"
date: 2026-08-10 11:51:10 +0900
categories: [Tenstorrent, sdpa]
tags: [sdpa]
description: "Tenstorrent SDPA Flash Decode가 긴 K/V cache를 여러 코어에 나누고 online softmax 상태를 tree reduction으로 합치는 과정을 설명합니다."
math: true
render_with_liquid: false
---

## 문서 범위

- 핵심: decode shape, `cur_pos`, GQA 작업 단위, paged K/V cache, online softmax, 코어 간 tree reduction
- 구현 기준: `sdpa_decode_device_operation.cpp`, `sdpa_decode_program_factory.cpp`, Reader·Compute·Writer kernel
- 확인일: 2026-08-12

이 글은 causal Flash Decode의 대표 경로를 설명한다. Sharded Q·출력, sliding window, attention sink, half-tile, positional ABI와 CB 번호처럼 특정 설정이나 소스 수정에 필요한 세부 사항은 다루지 않는다.

[Prefill SDPA](/posts/tenstorrent-sdpa-prefill/)와 가장 큰 차이는 출력 하나의 K/V 범위를 여러 코어가 나눈다는 점이다. 각 코어는 자신이 맡은 구간을 $(M,L,O)$ 상태로 요약하고, reduction root가 모든 상태를 합친 뒤 한 번만 정규화한다.

```text
one decode query
       |
       v
split cached K/V across cores
       |
       v
local [M,L,O] states
       |
       v
tree reduction at root
       |
       v
normalize O/L and write output
```

## Decode SDPA가 해결하는 문제

Scaled Dot-Product Attention(SDPA)은 다음을 계산한다.

$$
Y
=
\operatorname{softmax}\!\left(sQK^T+A\right)V,
\qquad
s=\frac{1}{\sqrt{D}}.
$$

$D$는 Q/K head dimension이고 $A$는 causal mask와 같은 additive mask다.

Prefill에서는 Q와 K/V sequence가 모두 길다. 반면 autoregressive decode의 한 step에서는 현재 token의 Q만 새로 생기고, K/V cache에는 이전 token과 현재 token이 누적돼 있다.

```text
decode step t

Q:       current token
K cache: positions 0..t
V cache: positions 0..t
```

따라서 decode에서는 긴 K/V cache를 읽는 비용과 그 범위를 여러 코어에 분배하는 방식이 중요하다. Flash Decode는 전체 score matrix를 외부 메모리에 만들지 않는다. 대신 K/V 구간마다 softmax를 병합할 수 있는 상태만 유지한다.

| 상태 | Q 행별 의미 | 논리적 shape |
| --- | --- | --- |
| $M$ | 지금까지 본 score의 최댓값 | `[R]` |
| $L$ | 현재 최댓값을 기준으로 한 exponential 합 | `[R]` |
| $O$ | 정규화 전 weighted-value 합 | `[R,D_v]` |

$R$은 tile padding을 포함한 Q 계산 행 수이고 $D_v$는 V와 출력의 head dimension이다.

## 입력 shape와 `cur_pos`

일반적인 unpaged causal decode의 논리적 shape는 다음과 같다.

```text
Q:       [1,B,Hq,D]
K cache: [B,Hkv,Lmax,D]
V cache: [B,Hkv,Lmax,D]
cur_pos: [B]
Output:  [1,B,Hq,D]
```

`cur_pos[b]`는 길이가 아니라 batch $b$의 마지막 유효 K/V 위치다. 0부터 시작하는 inclusive index이므로 실제 attention 길이는 다음과 같다.

$$
L_{\mathrm{valid}}[b]
=
\operatorname{cur\_pos}[b]+1.
$$

예를 들어 `cur_pos=95`이면 유효 범위는 position `0..95`, 길이는 96이다. `Lmax`가 128이더라도 뒤의 `96..127`은 이번 attention에 참여하지 않는다.

```text
positions  0                              95        127
           |-------------------------------|----------|
cache      [=========== valid =============][ unused ]
```

현재 token의 K/V도 이 유효 범위에 포함돼야 한다. 따라서 상위 attention layer는 먼저 position `cur_pos`에 현재 K/V를 기록하고, 같은 `cur_pos`로 decode SDPA를 호출한다.

```text
current K/V
     |
     v
cache update at position t
     |
     v
decode reads positions 0..t
```

## GQA에서는 `(batch, KV head)`가 작업 단위다

Grouped Query Attention(GQA)에서 Q head 수를 $H_q$, K/V head 수를 $H_{kv}$라고 하자. KV head 하나를 공유하는 Q head 수는 다음과 같다.

$$
G=\frac{H_q}{H_{kv}}.
$$

예를 들어 $H_q=8$, $H_{kv}=2$이면 다음처럼 묶인다.

```text
KV head 0 <- Q heads 0,1,2,3
KV head 1 <- Q heads 4,5,6,7
```

Flash Decode는 개념적으로 batch와 KV head의 조합마다 독립된 작업을 만든다.

$$
N_{\mathrm{work}}=B H_{kv}.
$$

작업 `(b,h)`는 다음 책임을 가진다.

1. `K/V[b,h,:,:]`의 유효 sequence를 처리한다.
2. 같은 batch의 Q를 사용해 attention 후보를 계산한다.
3. KV head $h$에 대응하는 Q head $hG$부터 $(h+1)G-1$까지의 출력 행을 기록한다.

```text
work (b,kv0) -> Q rows 0..G-1 ---+
                                 +--> output for batch b
work (b,kv1) -> Q rows G..2G-1 --+
```

대표 경로의 Compute는 tile로 패딩된 Q 행 영역을 사용한다. 마지막 Writer가 해당 KV head에 속한 유효 Q 행만 선택하므로 작업별 출력 영역은 겹치지 않는다.

## K/V sequence를 chunk와 코어로 나눈다

유효 K/V 길이는 `k_chunk_size` 단위로 나뉜다. 마지막 chunk가 경계를 넘어가면 초과 position은 mask로 제거한다.

`cur_pos=95`, `k_chunk_size=32`라면 필요한 chunk는 세 개다.

```text
chunk 0: positions  0..31
chunk 1: positions 32..63
chunk 2: positions 64..95
```

Factory는 작업마다 reduction group을 만들고 chunk를 group의 코어에 나눈다. Chunk 수가 코어 수보다 많으면 코어 하나가 여러 chunk를 처리한다. 반대로 chunk가 더 적으면 로컬 데이터가 없는 코어는 이번 reduction에서 빠진다.

설명을 위해 작업 하나에 코어 네 개를 배정했다고 하자.

```text
Core 0        Core 1        Core 2        Core 3
chunk 2       chunk 1       chunk 0       no data
   |             |             |             |
   v             v             v             v
local state   local state   local state     idle
   +-------------+-------------+
                 |
                 v
            root state
```

각 활성 코어는 같은 Q 행 영역과 자신이 맡은 K/V chunk로 다음 크기의 계산을 수행한다. 여기서 $C_k$는 K/V chunk의 token 수다.

```text
Q domain       [R,D]
K chunk        [Ck,D]
V chunk        [Ck,Dv]
score          [R,Ck]
M and L        [R]
O              [R,Dv]
```

코어별 결과가 단순한 출력 조각이 아니라 $(M,L,O)$인 이유는 softmax 부분 결과를 그대로 더할 수 없기 때문이다.

## Paged cache에서는 주소 계산만 달라진다

Unpaged cache는 batch별 sequence를 연속된 논리 shape로 표현한다.

```text
unpaged K/V: [B,Hkv,Lmax,D]
```

Paged cache는 payload를 고정 크기 physical block pool에 저장하고 page table로 논리 순서를 복원한다.

```text
paged K/V:  [Nphysical,Hkv,block_size,D]
page table: [B,max_logical_blocks]
```

Reader는 다음 순서로 K/V를 찾는다.

```text
cur_pos
   |
   v
valid logical positions
   |
   v
logical block and offset
   |
   v
page-table lookup
   |
   v
physical K/V read
   |
   v
logical-order K/V stream
```

예를 들어 logical block `0,1,2`가 physical block `5,1,7`에 놓여 있어도 Compute에는 position 순서대로 전달된다.

```text
logical block   0        1        2
physical block  5        1        7
                |        |        |
                +--------+--------+--> ordered K/V stream
```

`block_size`와 `k_chunk_size`는 역할이 다르다.

| 설정 | 의미 |
| --- | --- |
| `block_size` | Page-table entry 하나가 가리키는 cache 할당 단위 |
| `k_chunk_size` | Compute가 한 번에 처리하고 상태를 갱신하는 sequence 단위 |

따라서 compute chunk 하나가 여러 cache block을 포함할 수 있다. Paged와 unpaged 경로의 online softmax 수식은 같고, 차이는 주로 Reader 앞단의 주소 변환에 있다.

## Reader·Compute·Writer pipeline

세 kernel은 함수처럼 순서대로 호출되지 않는다. 같은 활성 코어에서 CB와 semaphore를 경계로 함께 진행된다.

```text
external tensors
  Q, K/V cache, cur_pos, optional page table
                    |
                    v
+---------------- one active core ----------------+
| Reader                                          |
|   Q once, ordered K/V chunks, masks             |
|                                                 |
| Compute                                         |
|   QK, softmax update, PV, local state merge     |
|                                                 |
| Writer                                          |
|   constants, tree relay, final output           |
+-------------------------+-----------------------+
                          |
                          v
                 reduction-group root
```

각 역할의 핵심은 다음과 같다.

| 역할 | 주된 책임 |
| --- | --- |
| Reader | `cur_pos`로 작업 범위를 정하고 Q와 논리 순서의 K/V chunk를 CB에 공급 |
| Compute | Chunk별 QK·softmax·PV를 실행하고 $(M,L,O)$를 갱신·병합 |
| Writer | Mask와 상수를 만들고, 하위 코어의 상태를 NoC로 전달하며, 최종 유효 Q 행을 기록 |

Q는 작업 동안 재사용하지만 K/V는 chunk마다 읽고 소비한다.

```text
Q    read once --------------------------> retained
K/V  chunk 0 -> consume -> chunk 1 -> consume -> ...
```

Reader가 다음 chunk를 준비하는 동안 Compute는 현재 chunk를 처리할 수 있다. Tree 단계에서는 root가 아닌 Compute가 상태를 로컬 CB에 공개한다. Writer는 이를 상위 코어의 L1으로 옮긴 뒤 semaphore로 도착을 알린다.

## Chunk별 online softmax

Chunk $j$의 scaled masked score를 다음처럼 두자.

$$
S_j=sQK_j^T+A_j.
$$

각 Q 행의 chunk별 상태는 다음과 같다.

$$
\begin{aligned}
M_j &= \operatorname{rowmax}(S_j),\\
P_j &= \exp(S_j-M_j),\\
L_j &= \operatorname{rowsum}(P_j),\\
O_j &= P_jV_j.
\end{aligned}
$$

Score block은 $P_j$를 계산한 뒤 버릴 수 있다. 이후 chunk와 병합할 때 필요한 정보는 $M_j$, $L_j$, $O_j$뿐이다.

이전 범위의 상태를 $(M_a,L_a,O_a)$, 새 chunk 상태를 $(M_b,L_b,O_b)$라고 하자. 두 범위의 공통 최댓값은 다음과 같다.

$$
M=\max(M_a,M_b).
$$

각 상태를 새 기준으로 옮기는 계수는 다음과 같다.

$$
\alpha=\exp(M_a-M),
\qquad
\beta=\exp(M_b-M).
$$

분모와 분자를 같은 비율로 보정해 더한다.

$$
\begin{aligned}
L &= \alpha L_a+\beta L_b,\\
O &= \alpha O_a+\beta O_b.
\end{aligned}
$$

이 갱신을 반복하면 score 전체를 보관하지 않고도 한 코어가 여러 chunk를 처리할 수 있다.

```text
state = empty

for each local K/V chunk:
    chunk_state = summarize(Q, K_chunk, V_chunk, mask)
    state = merge(state, chunk_state)
```

## 같은 상태를 코어 사이에서도 병합한다

로컬 chunk 병합과 코어 간 tree reduction은 같은 `merge()` 연산을 사용한다. 하위 코어가 이미 여러 상태를 합쳤더라도 상위 코어 입장에서는 또 하나의 $(M,L,O)$ 상태다.

```text
Core 3 local --> Core 2 subtree --+
                                  |
Core 1 local ---------------------+--> Core 0 root
                                  |         |
Core 0 local ---------------------+         v
                                           O/L
```

Root가 아닌 코어는 정규화하지 않은 상태를 상위 코어로 보낸다. 모든 활성 상태를 합친 root만 다음을 계산한다.

$$
Y=\frac{O}{L}.
$$

미리 $O/L$을 계산해 보내면 다른 범위와 병합할 때 필요한 최댓값과 분모 정보를 잃는다. 이것이 tree 전체에서 $(M,L,O)$를 유지하고 마지막에 한 번만 나누는 이유다.

Root의 Writer는 해당 KV head가 맡은 Q 행만 최종 텐서에 기록한다. Interleaved 출력에서는 KV-head root들이 서로 겹치지 않는 행을 직접 쓸 수 있다. Sharded 출력에서는 별도의 출력 코어가 여러 root의 부분 행을 모을 수 있다.

## Prefill과 비교하기

| 항목 | Flash Decode | Prefill SDPA |
| --- | --- | --- |
| Q 길이 | 보통 1 token | 여러 token |
| 독립 작업 | `(batch, KV head)` | `(batch, Q head, Q chunk)` |
| 주요 병렬화 | 한 출력의 K/V sequence 분할 | 서로 다른 Q 출력 영역 분할 |
| $(M,L,O)$ 위치 | 여러 코어에서 생성 | 한 담당 코어에 유지 |
| 코어 간 상태 병합 | Tree reduction | 없음 |
| 최종 정규화 | Reduction root | 각 작업 담당 코어 |
| Paged K/V | Decode cache | Chunked prefill의 prefix cache |

두 경로 모두 전체 score matrix 대신 $(M,L,O)$만 유지한다. 차이는 상태가 한 코어 안에서 끝나는지, 여러 코어 사이를 이동하는지에 있다.

## 핵심 정리

1. Decode Q는 현재 token 하나이고, K/V cache의 유효 범위는 inclusive index인 `cur_pos`로 정한다.
2. 현재 token의 K/V를 cache에 먼저 기록한 뒤 SDPA가 position `0..cur_pos`를 읽는다.
3. GQA에서는 `(batch, KV head)`가 독립 작업이며, 각 작업은 대응하는 Q head 행을 출력한다.
4. Factory는 긴 K/V sequence를 chunk로 나눠 reduction group의 코어에 분배한다.
5. Reader는 paged와 unpaged storage 차이를 흡수해 논리 순서의 K/V stream을 만든다.
6. Compute는 chunk마다 score 전체 대신 병합 가능한 $(M,L,O)$만 유지한다.
7. 로컬 chunk 갱신과 코어 간 tree reduction은 같은 최댓값 보정식을 사용한다.
8. Root가 아닌 코어는 정규화 전 상태를 보내고 root만 $O/L$을 계산한다.

```text
reuse Q, stream K/V
keep mergeable [M,L,O]
normalize once at the root
```

## 참고 자료

- Tenstorrent, [TTNN `scaled_dot_product_attention_decode` API](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.transformer.scaled_dot_product_attention_decode.html).
- Tenstorrent, [`sdpa_decode_device_operation.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp).
- Tenstorrent, [`sdpa_decode_program_factory.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp).
- Tenstorrent, [`rt_args_common.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp).
- Tenstorrent, [`reader_decode_all.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp).
- Tenstorrent, [`sdpa_flash_decode.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp).
- Tenstorrent, [`writer_decode_all.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp).
- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
