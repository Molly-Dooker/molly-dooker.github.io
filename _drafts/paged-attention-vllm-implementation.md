---
title: "PagedAttention과 vLLM 구현: KV cache를 block table로 관리하는 법"
date: 2026-08-21 09:57:24 +0900
categories: [DeepLearning, Inference]
tags: [vllm, paged-attention, kv-cache]
description: "PagedAttention이 가변 길이 KV cache를 고정 크기 block으로 나누는 원리와 vLLM V1의 할당, block table, slot mapping, FlashAttention 실행 경로를 분석합니다."
math: true
render_with_liquid: false
---

## 문서 범위

- 핵심: PagedAttention의 논리·물리 block mapping, KV cache 할당과 회수, `block_table`과 `slot_mapping`, prefix caching
- 논문 기준: [PagedAttention SOSP 2023 논문](https://arxiv.org/abs/2309.06180)
- 구현 기준: vLLM commit [`d29f7f5c9294`](https://github.com/vllm-project/vllm/tree/d29f7f5c9294be8e489dac34d45a939b95a06336)
- 대표 실행 경로: vLLM V1 scheduler, GPU Model Runner V2(MRV2), `FLASH_ATTN` backend, decoder-only full attention
- 확인일: 2026-08-21

이 글은 PagedAttention을 특정 CUDA kernel 하나가 아니라 **paged KV cache를 할당하고 주소를 변환하는 전체 실행 경로**로 설명한다. 논문의 메모리 모델부터 시작해 vLLM scheduler가 block을 배정하는 과정, GPU worker가 주소 metadata를 만드는 과정, attention backend가 새 K/V를 쓰고 기존 K/V를 읽는 과정까지 연결한다.

vLLM 저장소의 [`docs/design/paged_attention.md`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/docs/design/paged_attention.md#L1-L8)는 해당 문서가 논문 당시의 옛 kernel을 설명하며 최신 코드와 일치하지 않는다고 경고한다. 따라서 이 글도 그 문서의 thread·warp별 kernel 분석을 현재 CUDA 경로로 간주하지 않는다. 현재 vLLM은 플랫폼과 설정에 따라 FlashAttention, FlashInfer, Triton, ROCm AITER 등 서로 다른 backend를 선택할 수 있다. 여기서는 주소 변환을 구체적인 코드까지 추적하기 위해 MRV2와 FlashAttention 경로 하나로 범위를 고정한다.

Tensor parallelism(TP), decode context parallelism(DCP), pipeline parallelism, speculative decoding, KV connector, encoder-decoder attention, sliding-window attention, MLA(Multi-head Latent Attention), Mamba와 hybrid KV cache의 세부 정책은 제외한다. 이 기능들은 KV cache group 수, block 크기, slot 계산과 cache hit 조건을 바꿀 수 있다.

이 글에서 **KV block** 또는 **cache block**은 여러 token의 K/V를 담는 할당 단위를 가리킨다. CUDA thread block, FlashAttention의 연산 tile, scheduler의 token chunk와는 다른 개념이다.

## 연속 KV cache가 낭비를 만드는 이유

Autoregressive decode에서 현재 token의 query는 앞에서 계산한 모든 key와 value를 사용한다. 이전 K/V를 매번 다시 계산하지 않도록 layer마다 KV cache에 보관한다.

Layer 수를 $L$, KV head 수를 $H_{kv}$, key와 value의 head dimension을 각각 $D_k$, $D_v$, 원소 크기를 $e$ byte라고 하자. 시퀀스의 token 하나가 차지하는 KV cache 크기는 다음과 같다.

$$
M_{\mathrm{token}}
=
L H_{kv}(D_k+D_v)e.
$$

일반적으로 $D_k=D_v=D$이면 다음처럼 단순해진다.

$$
M_{\mathrm{token}}
=
2L H_{kv}De.
$$

예를 들어 layer 32개, KV head 8개, head dimension 128, BF16 2 byte라면 token 하나에 128 KiB가 필요하다. 길이 8192인 시퀀스 하나만으로 1 GiB다. TP로 KV head를 나누면 worker 하나가 저장하는 양은 줄지만, KV cache가 시퀀스 길이와 동시 요청 수에 비례해 증가한다는 성질은 같다.

초기 serving 시스템은 요청마다 최대 길이만큼 연속된 공간을 미리 확보했다. 실제 출력 길이를 알 수 없으므로 아직 생성하지 않은 token의 자리까지 예약해야 한다.

```text
Contiguous reservation

Request A: [======== used ========][........ reserved ........]
Request B: [==== used ====][............ reserved ............]
Request C: [========== used ==========][.. reserved ..]

GPU pool : [ Request A ][ Request B ][ hole ][ Request C ][ hole ]
```

이 방식에는 두 종류의 단편화가 생긴다.

- 내부 단편화는 요청에 배정했지만 아직 쓰지 않았거나 끝까지 쓰지 않는 예약 영역이다.
- 외부 단편화는 빈 공간의 총합은 충분해도 연속된 큰 영역이 없어 새 요청을 넣지 못하는 상황이다.

[PagedAttention 논문](https://arxiv.org/abs/2309.06180)은 실험한 기존 시스템에서 실제 token state가 KV cache 메모리의 20.4%~38.2%만 차지한 사례를 보고했다. 이는 모든 워크로드에 적용되는 고정 비율이 아니라 논문 §6.2의 모델, 요청 trace와 allocator 조건에서 측정한 결과다.

## 논리 block과 물리 block을 분리한다

PagedAttention은 시퀀스의 연속된 token 위치를 고정 크기 **논리 block**으로 나눈다. GPU KV cache pool도 같은 용량의 **물리 block**으로 나눈다. 요청은 논리 block 순서만 유지하고, 실제 payload는 pool의 어느 물리 block에나 둘 수 있다.

Block 하나가 담는 token 수를 $B$, 현재 시퀀스 길이를 $S$라고 하자. 필요한 논리 block 수는 다음과 같다.

$$
N_{\mathrm{block}}
=
\left\lceil\frac{S}{B}\right\rceil.
$$

기본 경로에서는 기존 block이 가득 찼을 때 새 물리 block을 추가한다. 따라서 lookahead와 KV cache group별 예외를 제외하면 full-attention 시퀀스 하나에서 마지막 block 때문에 생기는 내부 낭비는 최대 $B-1$ token slot이다. 모든 물리 block의 크기가 같으므로 서로 다른 크기의 연속 영역을 찾을 필요도 없다.

다음은 주소 계산을 단순하게 보이기 위해 $B=4$로 만든 예제다. 뒤에서 살펴볼 기준 revision의 FlashAttention backend는 실제로 16의 배수인 block 크기를 요구한다.

```text
Request R, block size B = 4

Logical blocks
+----------+----------+----------+
| L0: 0..3 | L1: 4..7 | L2: 8..9 |
+----+-----+----+-----+----+-----+
     |          |          |
     v          v          v
Block table T[R] = [7, 1, 5]
     |          |          |
     v          v          v
Physical pool
+----+----+----+----+----+----+----+----+
| P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|    | L1 |    |    |    | L2 |    | L0 |
+----+----+----+----+----+----+----+----+
```

요청 $r$의 block table을 $T_r$라고 하자. 논리 token 위치 $t$의 논리 block index $b$, block 내부 offset $o$, 물리 block ID $p$는 다음과 같다.

$$
\begin{aligned}
b &= \left\lfloor\frac{t}{B}\right\rfloor,\\
o &= t \bmod B,\\
p &= T_r[b].
\end{aligned}
$$

KV cache를 block과 token slot 축으로 보면 실제 주소는 `(p, o)`다. Token slot을 일차원 ID로 펴면 다음 `slot`을 얻는다.

$$
\operatorname{slot}(r,t)
=
T_r\!\left[\left\lfloor\frac{t}{B}\right\rfloor\right]B
+(t\bmod B).
$$

이 예제에서 position 6은 논리 block 1, offset 2다. `T[R][1]=1`이므로 물리 slot은 $1\times4+2=6$이다. Position 9는 `T[R][2]=5`를 거쳐 slot 21로 변환된다.

Block table이 만드는 간접 참조 때문에 논리적으로 이웃한 block이 물리적으로 떨어져 있어도 시퀀스 순서를 복원할 수 있다. 반대로 물리 주소가 가깝다는 사실만으로 같은 요청이나 인접 position이라고 판단하면 안 된다.

## 하나의 block ID가 layer별 cache 위치를 정한다

간단한 full-attention 모델에서 scheduler가 관리하는 `block_id`는 요청의 token 구간을 나타낸다. 실제 K/V payload는 attention layer마다 별도의 cache tensor에 있다. 같은 `block_id=p`는 각 layer의 cache tensor에서 첫 번째 축의 같은 index를 선택한다.

기준 revision의 [`FlashAttentionBackend.get_kv_cache_shape()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/attention/backends/flash_attn.py#L141-L175)는 layer 하나의 논리 shape를 다음처럼 정의한다.

```text
KV cache per layer: [num_blocks, num_kv_heads, block_size, 2 * head_size]
```

마지막 축에는 K와 V가 함께 들어간다. `NHD` 또는 `HND` 설정에 따라 실제 stride 순서는 달라질 수 있지만, `num_blocks` 축이 물리 block pool 역할을 한다.

```text
One scheduler block id p
          |
          +--> layer 0 KV cache[p, ...]
          +--> layer 1 KV cache[p, ...]
          +--> layer 2 KV cache[p, ...]
          |
          +--> layer L-1 KV cache[p, ...]
```

따라서 앞의 $M_{\mathrm{token}}$과 $B M_{\mathrm{token}}$은 모델 전체 관점의 token·block 비용이다. CUDA worker 하나와 layer tensor의 실제 크기를 계산할 때는 TP로 나뉜 local KV head 수, cache dtype, K/V dimension, padding과 KV cache group 구성을 반영해야 한다.

## 비연속 K/V로 attention을 계산한다

PagedAttention이 바꾸는 것은 K/V의 **주소를 찾는 방법**이다. Attention의 수학적 결과는 바꾸지 않는다. 요청 $r$의 논리 block $j$에 해당하는 K/V는 다음처럼 물리 block을 거쳐 읽는다.

$$
\begin{aligned}
p_j &= T_r[j],\\
K_j &= K_{\mathrm{cache}}[p_j],\\
V_j &= V_{\mathrm{cache}}[p_j].
\end{aligned}
$$

현재 query $q$에 대해 block $j$의 score를 다음처럼 계산할 수 있다.

$$
S_j
=
\frac{qK_j^T}{\sqrt{D}}+A_j,
$$

여기서 $A_j$는 causal, sliding-window 또는 다른 attention mask 중 해당 block 범위에 적용되는 부분이다. 각 block의 안정적인 softmax 상태를 다음과 같이 두자.

$$
\begin{aligned}
m_j &= \max(S_j),\\
P_j &= \exp(S_j-m_j),\\
l_j &= \sum P_j,\\
O_j &= P_jV_j.
\end{aligned}
$$

두 범위의 상태 $(m_a,l_a,O_a)$와 $(m_b,l_b,O_b)$는 공통 최댓값 $m=\max(m_a,m_b)$로 옮겨 합칠 수 있다.

$$
\begin{aligned}
\alpha &= \exp(m_a-m),\\
\beta &= \exp(m_b-m),\\
l &= \alpha l_a+\beta l_b,\\
O &= \alpha O_a+\beta O_b.
\end{aligned}
$$

마지막 출력은 $O/l$이다. 이 online softmax는 score 전체를 메모리에 만들지 않게 해 주지만, **KV cache block과 kernel의 softmax tile이 반드시 같은 크기라는 뜻은 아니다**. Backend는 한 cache block을 여러 tile로 나누거나 여러 block을 한 split에서 처리할 수 있다. PagedAttention의 핵심 조건은 각 논리 범위가 가리키는 물리 K/V를 올바른 순서와 길이로 읽는 것이다.

## `block_table`과 `slot_mapping`은 역할이 다르다

vLLM 구현을 읽을 때 가장 먼저 구분해야 할 metadata는 `block_table`과 `slot_mapping`이다.

| Metadata | 단위 | 역할 |
| --- | --- | --- |
| `block_table[r, b]` | 요청 × 논리 block | 이미 저장된 K/V를 읽기 위해 논리 block $b$를 물리 block ID로 변환한다. |
| `slot_mapping[i]` | 이번 step의 input token | 새로 계산한 token $i$의 K/V를 기록할 일차원 물리 slot을 지정한다. |
| `seq_lens[r]` | 요청 | 마지막 물리 block의 미사용 slot을 attention에서 제외한다. |
| `query_start_loc` | 요청 경계의 prefix sum | 여러 요청의 가변 길이 query를 하나의 packed token tensor에서 구분한다. |

`block_table`은 요청의 전체 context를 읽는 page table이고, `slot_mapping`은 이번 step에서 계산한 K/V의 write address다. 둘 다 같은 논리-물리 mapping에서 나오지만 소비하는 kernel과 shape가 다르다.

다음 예제는 $B=4$일 때 prefill과 decode가 같은 mapping을 어떻게 갱신하는지 보여 준다.

| Step | 처리 position | Block table | 새 `slot_mapping` | 새 block 할당 |
| --- | --- | --- | --- | --- |
| Prefill | `0..5` | `[7, 2]` | `[28, 29, 30, 31, 8, 9]` | `P7`, `P2` |
| Decode 1 | `6` | `[7, 2]` | `[10]` | 없음 |
| Decode 2 | `7` | `[7, 2]` | `[11]` | 없음 |
| Decode 3 | `8` | `[7, 2, 5]` | `[20]` | `P5` |

Position 6과 7은 이미 할당한 마지막 block의 빈 slot을 사용한다. Position 8에서만 논리 block 2가 생기므로 새 물리 block `P5`를 받고 table 끝에 ID 5를 붙인다.

```text
New token position t
          |
          v
logical block = floor(t / B)
          |
          v
physical block = block_table[request, logical block]
          |
          v
slot_mapping = physical block * B + (t mod B)
          |
          v
scatter K/V into paged cache
```

## vLLM V1에서 block을 누가 관리하는가

현재 기준 경로는 CPU scheduler의 할당 metadata와 GPU worker의 payload tensor를 분리한다.

```text
+--------------------------- CPU ----------------------------+
| Request                                                    |
|    |                                                       |
|    v                                                       |
| Scheduler -> KVCacheManager -> Coordinator -> BlockPool    |
|    |              |                         |              |
|    |              +-- request block ids     +-- free queue |
|    |                                        +-- hash map   |
|    v                                                       |
| SchedulerOutput: token counts and new block ids            |
+-----------------------------+------------------------------+
                              |
                              v
+--------------------------- GPU ----------------------------+
| Model Runner -> persistent block tables                    |
|                    |                                       |
|                    +--> per-step block_table               |
|                    +--> per-token slot_mapping              |
|                              |                              |
|                              v                              |
| Attention layer: write new K/V -> paged read -> output     |
+------------------------------------------------------------+
```

### `KVCacheBlock`과 `BlockPool`

[`KVCacheBlock`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/kv_cache_utils.py#L161-L195)은 K/V tensor 자체가 아니라 `block_id`, reference count, prefix-cache hash와 free-queue link를 가진 CPU metadata 객체다.

[`BlockPool`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/block_pool.py#L143-L193)은 시작할 때 `num_gpu_blocks`개의 metadata 객체를 만들고, doubly linked free queue와 prefix hash map을 관리한다. 새 block이 필요하면 [`get_new_blocks()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/block_pool.py#L647-L677)가 free queue 앞에서 block을 꺼내 `ref_cnt`를 올린다. 꺼낸 block에 오래된 prefix-cache hash가 남아 있으면 그 mapping을 먼저 evict한다.

단일 full-attention group에서는 `FullAttentionManager`가 요청별 block list를 보관한다. Model에 attention type이나 state 크기가 다른 layer가 섞이면 여러 `SingleTypeKVCacheManager`를 coordinator가 조합한다. 따라서 현재 코드의 `KVCacheBlocks.blocks[i][j]`에서 바깥 index는 KV cache group이고, 안쪽 index가 요청의 논리 block 순서다. 이 글의 toy example은 group이 하나인 경우다.

### Scheduler의 `allocate_slots()`

새 요청이 들어오면 scheduler는 먼저 [`get_computed_blocks()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/kv_cache_manager.py#L232-L298)로 재사용 가능한 prefix block을 찾는다. 그다음 [`allocate_slots()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/kv_cache_manager.py#L347-L568)이 다음 범위를 함께 고려한다.

1. 이미 계산한 token과 새 prefix-cache hit
2. 이번 step에서 계산할 token
3. Speculative decoding용 lookahead slot
4. Sliding window 밖이라 회수할 수 있는 block
5. KV connector가 외부에서 가져올 cache와 예약 block

필요한 block 수가 free block보다 많으면 `allocate_slots()`는 `None`을 반환한다. 실행 중 요청에서 이런 상황이 생기면 scheduler는 설정한 scheduling policy에 따라 preemption 대상을 고르고 다시 할당을 시도한다. 기준 revision의 [`_preempt_request()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/sched/scheduler.py#L1347-L1388)는 요청 block을 풀고 `num_computed_tokens`를 0으로 되돌린 뒤 waiting queue에 넣는다. 재개할 때 살아 있는 prefix-cache block을 다시 찾을 수 있지만, 논문의 초기 구현처럼 CPU swap이 항상 일어나는 것은 아니다.

### MRV2가 GPU용 table과 slot을 만든다

Scheduler output에는 새 요청의 전체 block ID 또는 실행 중 요청에 덧붙일 새 block ID가 들어간다. MRV2의 [`add_requests()`와 `update_requests()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/worker/gpu/model_runner.py#L963-L1055)는 이를 persistent `BlockTables`에 기록한다. 큰 table 전체를 step마다 복사하지 않고 변경분을 stage한 뒤 GPU에 적용한다.

실제 batch 순서는 persistent request row 순서와 다를 수 있다. [`prepare_attn()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/worker/gpu/model_runner.py#L1311-L1330)은 이번 batch 순서로 block-table row를 gather하고, 각 input position의 `slot_mapping`을 계산한다.

Context parallelism을 사용하지 않는 일반 경로에서 [`_compute_slot_mappings_kernel`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/worker/gpu/block_table.py#L269-L327)은 앞에서 유도한 식을 그대로 구현한다.

```python
block_index = position // block_size
block_offset = position % block_size
block_number = block_table[request, block_index]
slot_id = block_number * block_size + block_offset
```

CUDA Graph padding token에는 유효한 cache를 덮어쓰지 않도록 `PAD_SLOT_ID`를 넣는다. DCP나 PCP(Parallel Context Processing)를 켜면 position을 rank별 local offset으로 다시 변환하므로 위 네 줄만으로는 충분하지 않다.

## FlashAttention backend에서 쓰기와 읽기

MRV2는 `block_table`, `slot_mapping`, `seq_lens`, `query_start_loc`을 attention metadata로 묶어 모델의 forward context에 넣는다. 각 attention layer는 Q/K/V projection 뒤 다음 순서로 cache와 attention을 처리한다.

```text
project hidden states to Q, K, V
              |
              v
unified_kv_cache_update
  scatter K/V by slot_mapping
              |
              v
unified_attention_with_output
  read paged K/V by block_table and seq_lens
              |
              v
attention output
```

기준 FlashAttention backend는 `forward_includes_kv_cache_update=False`다. 따라서 공통 attention layer는 [`unified_kv_cache_update()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/model_executor/layers/attention/attention.py#L701-L724)를 먼저 실행하고, dummy dependency를 attention call에 넘겨 compiler가 write-read 순서를 보존하게 한다.

[`FlashAttentionImpl.do_kv_cache_update()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/attention/backends/flash_attn.py#L1180-L1214)는 packed KV cache를 key cache와 value cache view로 나눈다. 이어 `reshape_and_cache_flash()`가 이번 token의 K/V를 `slot_mapping` 위치에 scatter한다.

그 뒤 [`FlashAttentionImpl.forward()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/attention/backends/flash_attn.py#L917-L1015)는 cache를 `[num_blocks, block_size, num_kv_heads, head_size]`에 해당하는 K/V view로 해석한다. 일반 경로는 [`flash_attn_varlen_func()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/attention/backends/flash_attn.py#L1123-L1148)에 다음 정보를 함께 넘긴다.

- packed query의 요청별 경계인 `cu_seqlens_q`
- 요청별 실제 KV 길이인 `seqused_k`
- 논리 block을 물리 block으로 바꾸는 `block_table`
- causal mask, sliding window, scale과 backend별 scheduling metadata

즉 현재 대표 CUDA 경로에서 PagedAttention은 “`paged_attention_v1`이라는 이름의 kernel을 호출하는가”로 판별할 수 없다. FlashAttention kernel도 `block_table`을 받아 paged cache를 직접 읽는다. **PagedAttention은 저장과 주소 지정의 성질이고, FlashAttention은 주로 attention 계산의 메모리 I/O와 tiling을 최적화하는 기법**이다. 현재 vLLM은 두 성질을 한 backend 경로에서 함께 사용한다.

## Prefill과 decode는 같은 table을 다르게 사용한다

Paged KV cache 추상화는 prefill과 decode 모두에 적용된다. 차이는 한 step의 query 길이와 할당량이다.

| 구분 | Prefill | Decode |
| --- | --- | --- |
| 이번 step의 token 수 | prompt 전체 또는 chunk의 여러 token | 일반적으로 요청당 새 token 하나 |
| Block 할당 | 여러 block을 한 번에 받을 수 있음 | 마지막 block을 채우다가 경계에서 하나 추가 |
| `slot_mapping` | 여러 input position의 write slot | 보통 요청당 한 write slot |
| Attention 범위 | 각 query가 causal prefix를 봄 | 새 query가 지금까지의 전체 유효 KV를 봄 |
| 주된 연산 특성 | Q/K/V가 커서 행렬 연산 비중이 큼 | 긴 KV read 때문에 메모리 대역폭에 민감함 |

Continuous batching에서는 같은 모델 step에 짧은 decode query와 chunked prefill query가 함께 들어갈 수 있다. `query_start_loc`과 `seq_lens`가 요청마다 다른 Q/KV 길이를 표현하므로 물리 block을 연속 시퀀스 tensor로 다시 복사할 필요가 없다.

다만 paging은 decode의 계산 복잡도를 없애지 않는다. Full attention decode는 새 token마다 과거 $S$개의 K/V를 읽으므로 token당 KV read와 attention 계산이 여전히 $O(S)$다. Paging이 줄이는 것은 최대 길이 예약과 단편화, 그리고 공유 불가능한 연속 buffer 때문에 생기는 메모리 낭비다.

## Prefix caching은 block mapping을 공유한다

PagedAttention은 여러 요청이 같은 물리 block을 가리킬 수 있게 한다. vLLM V1의 Automatic Prefix Caching(APC)은 이 성질을 이용해 같은 prompt prefix의 K/V를 다시 계산하지 않는다.

일반적인 full-block hash는 이전 block hash, 현재 block token ID, 추가 구분값으로 chain을 만든다.

$$
h_j
=
H(h_{j-1},\ \mathrm{tokens}_j,\ \mathrm{extra}_j).
$$

추가 구분값에는 LoRA adapter ID, multimodal input hash, tenant 격리를 위한 cache salt처럼 같은 token ID라도 K/V가 달라질 수 있는 정보가 들어간다. 자세한 구성은 vLLM의 [Automatic Prefix Caching 설계 문서](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/docs/design/prefix_caching.md#L1-L35)에서 확인할 수 있다.

```text
Request A table: [P7, P2, P5]
Request B table: [P7, P2, P9]
                   ^   ^
                   +---+--- shared prefix blocks
```

Cache hit가 나면 `BlockPool.touch()`가 해당 block의 `ref_cnt`를 올리고 free queue에 있었다면 제거한다. 요청이 끝나면 reference count를 내린다. `ref_cnt=0`이 된 cached block은 즉시 hash를 버리는 대신 free queue에서 재사용·eviction 후보로 남을 수 있다. 이후 같은 prefix가 도착하면 다시 touch하고, 새 payload를 쓸 공간이 부족하면 오래된 cached block을 evict해 재할당한다.

공유 중인 block을 한 요청이 그대로 수정하면 다른 요청의 K/V가 오염된다. Full block 뒤에는 새 block을 붙이면 되지만, 공유한 partial tail 안에 token을 이어 써야 하면 전용 복사본이 필요하다. 기준 revision의 [`allocate_new_blocks()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/single_type_kv_cache_manager.py#L327-L366)은 partial hit를 새 CoW(Copy-on-Write) block으로 전환한다. MRV2는 [forward 전에 원본 block을 대상 block으로 복사](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/worker/gpu/model_runner.py#L1042-L1055)한 뒤 새 K/V를 기록한다.

현재 `FullAttentionManager`는 cache block보다 작은 hash alignment를 쓰는 설정에서 [prompt의 partial tail도 cache한 뒤 찾을 수 있다](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/single_type_kv_cache_manager.py#L680-L821). “full block만 cache한다”는 과거 설명을 모든 revision의 불변 조건으로 보면 안 된다. Partial entry는 새 payload block이 아니라 기존 물리 block을 더 세밀한 prefix boundary에서 찾게 하는 metadata이며, 이어 쓰기 전에는 위 CoW가 필요하다.

Prompt 전체가 cache hit여도 logits가 자동으로 생기지는 않는다. [`get_computed_blocks()`](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/kv_cache_manager.py#L251-L268)은 다음 token logits를 얻기 위해 마지막 prompt token을 다시 계산할 수 있도록 최대 hit 길이를 `num_tokens - 1`로 제한한다.

## Block 크기의 trade-off

Block 크기는 단편화만 보고 작게 정할 수 없다.

| 작은 block | 큰 block |
| --- | --- |
| 마지막 block의 평균 미사용 slot이 적다. | 같은 sequence를 더 짧은 block table로 표현한다. |
| 짧은 prefix도 세밀하게 공유·회수하기 쉽다. | Allocator와 hash metadata 처리 횟수가 줄어든다. |
| Table entry와 주소 lookup 수가 늘어난다. | 연속된 token을 한 block에서 읽어 kernel 효율을 높이기 쉽다. |
| Block 할당·회수 빈도가 커질 수 있다. | 짧은 sequence와 partial tail의 내부 낭비가 커진다. |

2023년 논문은 당시 구현에서 block size 16을 기본값으로 사용했고, workload에 따라 16~128 사이의 성능을 비교했다. 이는 현재 모든 backend에 그대로 적용할 tuning 결론은 아니다. 기준 revision의 FlashAttention backend는 [16의 배수인 kernel block size](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/attention/backends/flash_attn.py#L89-L99)를 지원한다. Hybrid KV cache에서는 allocator가 보는 group block과 kernel block의 크기가 달라 하나의 manager block ID를 여러 kernel block ID로 펼칠 수도 있다.

따라서 실제 값을 정할 때는 시퀀스 길이 분포, prefix hit 분포, backend 제약, KV dtype, attention type과 kernel benchmark를 함께 봐야 한다.

## 흔히 생기는 오해

### PagedAttention은 OS virtual memory와 동일하지 않다

비유의 대상은 고정 크기 page, logical-to-physical mapping, on-demand allocation, sharing과 CoW다. GPU page fault나 hardware TLB가 요청의 KV block을 자동으로 옮기는 구조는 아니다. vLLM scheduler와 worker가 block ID, table, slot을 명시적으로 만들고 compatible kernel이 이를 해석한다.

### `block_table`이 K/V payload를 담지는 않는다

Table에는 보통 `int32` physical block ID가 들어간다. 실제 K/V는 layer별 GPU tensor에 있고 table은 그 첫 번째 축을 간접 참조한다. `KVCacheBlock` Python 객체도 payload가 아니라 allocator metadata다.

### KV block과 compute chunk는 같지 않다

KV block은 저장과 할당 단위다. FlashAttention tile, split-KV partition, chunked prefill 크기와 scheduler token budget은 실행 효율을 위한 별도 단위다. 우연히 수치가 같아도 의미까지 같아지는 것은 아니다.

### Paging만으로 OOM이 사라지지는 않는다

Paged allocation은 주어진 KV pool을 촘촘하게 쓰게 하지만 pool의 총용량을 늘리지는 않는다. 요청의 실제 live KV, prefix cache, lookahead slot과 backend workspace가 물리 메모리를 넘으면 admission 제한, preemption, recomputation, offload 또는 더 많은 worker가 필요하다.

## 코드를 읽는 순서

현재 vLLM에서 paged attention 동작을 추적할 때는 다음 순서가 효율적이다.

1. Attention backend가 요구하는 `block_size`, KV cache shape와 layout을 확인한다.
2. `KVCacheManager.allocate_slots()`가 요청에 어떤 physical block ID를 배정하는지 본다.
3. Scheduler output에서 새 요청과 실행 중 요청의 block ID가 어떻게 전달되는지 본다.
4. Model runner가 persistent table을 갱신하고 이번 batch용 table row를 gather하는지 확인한다.
5. Position, block table과 block size로 `slot_mapping`을 계산하는 식을 확인한다.
6. Attention layer가 `slot_mapping`으로 K/V를 먼저 쓴 뒤 `block_table`로 cache를 읽는 순서를 확인한다.
7. 마지막으로 선택된 backend의 kernel tiling, online softmax와 architecture별 최적화를 읽는다.

```text
cache shape
    |
    v
block allocation
    |
    v
scheduler output
    |
    v
GPU block table and slot mapping
    |
    v
KV scatter write
    |
    v
paged attention read
    |
    v
backend kernel details
```

이 순서를 따르면 allocator의 `block`, CUDA의 thread block, attention tile을 섞지 않고 시스템 수준의 mapping부터 kernel 수준의 계산까지 내려갈 수 있다.

## 제약과 가정

- 수식과 예제는 decoder-only full attention, KV cache group 하나, context parallelism을 사용하지 않는 경우를 기준으로 한다.
- K/V dimension이 다른 모델은 $2D$ 대신 $D_k+D_v$로 계산해야 한다. MLA와 quantized KV cache는 tensor layout과 byte 계산이 크게 달라질 수 있다.
- 현재 source는 빠르게 바뀐다. 특히 prefix-cache partial hit, hybrid group, MRV2와 backend별 metadata는 이 글의 고정 commit 이후 달라질 수 있다.
- 논문의 throughput 2~4배와 메모리 사용률 수치는 2023년 비교 대상과 워크로드에서 나온 결과다. 현재 hardware와 최신 serving engine 사이의 성능 우위를 뜻하지 않는다.
- Backend가 paged KV cache를 지원하더라도 block size, head size, dtype, sliding window와 CUDA architecture 조합별 지원 범위는 다르다.

## 핵심 정리

1. PagedAttention은 요청의 연속된 KV 위치를 고정 크기 논리 block으로 나누고, block table로 비연속 물리 block에 연결한다.
2. On-demand allocation은 full-attention sequence의 미사용 예약을 마지막 block의 빈 slot로 제한하고, 고정 크기 pool은 외부 단편화를 없앤다.
3. `block_table`은 기존 context를 읽는 request-level page table이고, `slot_mapping`은 이번 step에서 계산한 K/V를 쓰는 token-level address다.
4. vLLM V1에서는 scheduler의 `KVCacheManager`와 `BlockPool`이 block 수명과 공유를 관리하고, GPU model runner가 kernel용 table과 slot을 만든다.
5. 기준 FlashAttention 경로는 새 K/V를 `slot_mapping`으로 scatter한 뒤 `block_table`과 실제 시퀀스 길이를 넘겨 paged cache에서 attention을 계산한다.
6. Prefix caching은 여러 요청의 block table이 같은 물리 block을 가리키게 하고, reference count와 CoW로 공유 block의 수명을 보호한다.
7. PagedAttention과 FlashAttention은 대체 관계가 아니다. 전자는 KV storage와 주소 변환, 후자는 attention 계산의 I/O와 tiling에 초점을 둔다.

## 참고 자료

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [SOSP 2023 PagedAttention 논문 페이지](https://doi.org/10.1145/3600006.3613165)
- [vLLM 기준 source revision `d29f7f5c9294`](https://github.com/vllm-project/vllm/tree/d29f7f5c9294be8e489dac34d45a939b95a06336)
- [vLLM V1 KV cache manager](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/kv_cache_manager.py)
- [vLLM V1 block pool](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/core/block_pool.py)
- [vLLM MRV2 block table과 slot-mapping kernel](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/worker/gpu/block_table.py)
- [vLLM V1 FlashAttention backend](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/vllm/v1/attention/backends/flash_attn.py)
- [vLLM Automatic Prefix Caching 설계](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/docs/design/prefix_caching.md)
- [vLLM Model Runner V2 설계](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/docs/design/model_runner_v2.md)
- [vLLM historical PagedAttention kernel 문서](https://github.com/vllm-project/vllm/blob/d29f7f5c9294be8e489dac34d45a939b95a06336/docs/design/paged_attention.md)
