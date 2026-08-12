---
title: "Tenstorrent Blackhole FPU와 Matmul: MVMUL 분할과 Math Fidelity"
date: 2026-08-13 09:50:19 +0900
categories: [Tenstorrent, matmul]
tags: [matmul]
description: "Blackhole matrix engine(FPU)의 32×32 tile matmul을 8×16 MVMUL로 분할하고 MOP·Replay와 Math Fidelity phase로 누적하는 과정을 설명합니다."
math: true
render_with_liquid: false
---

## 문서 범위

- hardware 기준: Blackhole A0의 matrix engine(FPU). ISA 문서에서는 `Matrix Unit`이라고 부른다.
- 핵심 경로: transpose와 throttle을 끈 일반적인 32×32 full-tile floating-point matmul
- 기준 구현: `tt-metal` commit [`69096826694c`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e)
- 기준 ISA(Instruction Set Architecture): `tt-isa-documentation` commit [`b7738d9ac14a`](https://github.com/tenstorrent/tt-isa-documentation/tree/b7738d9ac14a34a4033d60dde9463466b23082e1)
- architecture 개요: [TT-Metalium compute engine 공식 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html)
- 비교 자료: Peter Cawley, [Tenstorrent Wormhole Series Part 7: Bits of the MatMul](https://www.corsix.org/content/tt-wh-part7), 2024-10-15
- 상위 수준의 연결: [Matmul Program Factory 비교](/posts/tenstorrent-matmul-program-factories/)
- register와 instruction 기초: [Tenstorrent LLK 기초](/posts/tenstorrent-llk-basics/)
- 확인일: 2026-08-13

이 글은 [LLK 기초 글](/posts/tenstorrent-llk-basics/)의 공통 구조에서 matrix engine(FPU) 경로를 더 깊게 내려가는 문서다. 큰 행렬을 여러 코어에 배치하는 방법보다 한 Tensix core 안에서 tile 곱 하나가 실제 instruction으로 어떻게 풀리는지에 집중한다. SFPU 경로는 별도 문서인 “Tenstorrent SFPU 기초와 Typecast 예제”에서 다룬다.

```text
Tenstorrent LLK basics
        |
        +--> Matrix engine (FPU): Matmul, MVMUL, MOP, Math Fidelity
        |
        +--> Vector engine (SFPU): SFPI, SFPLOAD, Typecast, scheduling
```

이 글은 32×32 tile pair 하나가 왜 fidelity phase마다 `MVMUL` 16개로 분할되는지, MOP(Macro-Operation)와 Replay Expander가 이를 어떻게 반복하는지, LoFi·HiFi2·HiFi3·HiFi4가 어떤 mantissa(가수부) 조각을 추가로 곱하는지 설명한다.

먼저 결론을 요약하면 다음과 같다.

1. `MVMUL` 하나는 `Dst[8×16] += SrcB[8×16] @ SrcA[16×16]`을 수행한다.
2. 32×32 tile 곱은 16×16 face 곱 8개로 나뉘고, face 곱 하나는 출력의 위·아래 8행에 `MVMUL` 두 개가 필요하다. 따라서 fidelity phase 하나에 `MVMUL`이 16개다.
3. LoFi, HiFi2, HiFi3, HiFi4는 각각 1, 2, 3, 4개 phase를 실행한다. 32×32 tile pair 하나에는 각각 16, 32, 48, 64개의 `MVMUL`이 필요하다.
4. High fidelity용으로 서로 다른 MVMUL 목록을 새로 만드는 것은 아니다. 한 phase의 `MVMUL` 16개를 Replay buffer에 기록하고, MOP가 같은 `REPLAY`를 phase 수만큼 반복한다.
5. 각 phase는 입력 mantissa의 high·low 조각으로 만든 네 cross-product 중 하나를 `Dst`에 더한다.

## 32×32 tile 곱은 왜 MVMUL 16개인가

분석 단위는 왼쪽 32×32 tile $L$과 오른쪽 32×32 tile $R$의 곱 하나다. 큰 행렬에서는 이런 tile-pair 곱을 K 방향으로 반복해 누적하지만, 여기서는 한 pair 내부가 어떻게 분할되는지만 따라간다.

`Dst += in0 @ in1`에서 `in0`은 `SrcB`, `in1`은 `SrcA`에 할당된다([compute `matmul.h`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/compute/matmul.h#L143-L150), [Blackhole unpack LLK](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h#L22-L42)).

`in0`과 `in1`은 operand 순서를 나타낼 뿐 tensor의 역할을 고정하지 않는다. 다만 일반적인 linear layer $Y=XW$에서는 activation $X$가 `in0 -> SrcB`, 가중치 $W$가 `in1 -> SrcA`에 놓인다. [공식 `ttnn.linear` 예제](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.linear.html#ttnn.linear)도 `ttnn.linear(activations, weight, ...)` 순서를 사용한다.

### Tile 하나를 네 face로 나눈다

32×32 tile은 네 개의 16×16 face로 배치된다.

$$
L=
\begin{bmatrix}
L_0 & L_1 \\
L_2 & L_3
\end{bmatrix},
\qquad
R=
\begin{bmatrix}
R_0 & R_1 \\
R_2 & R_3
\end{bmatrix}.
$$

출력 face는 보통의 2×2 block matrix 곱으로 정해진다.

$$
\begin{aligned}
C_0 &= L_0R_0 + L_1R_2, &
C_1 &= L_0R_1 + L_1R_3, \\
C_2 &= L_2R_0 + L_3R_2, &
C_3 &= L_2R_1 + L_3R_3.
\end{aligned}
$$

출력 face는 네 개이고, 각 face에는 K 방향 16×16 곱이 두 개씩 들어간다. 여기까지는 16×16 face product가 $4\times2=8$개다.

![SrcB를 왼쪽, Dst를 오른쪽, SrcA를 Dst 아래에 놓고 32×32 tile 곱을 face product와 MVMUL로 분해한 구조](/assets/img/posts/tenstorrent-blackhole-matmul-mvmul-fidelity/tile-to-mvmul.svg)

### Face 곱 하나는 위·아래 두 MVMUL로 나눈다

공식 [`MVMUL` ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md#L1-L13)에서 `MVMUL` 하나는 `SrcB`의 8×16 block과 `SrcA`의 16×16 block을 곱해 8×16 결과를 만든다. 따라서 16×16 face product는 왼쪽 face를 위·아래 8행으로 나눠 `MVMUL` 두 번으로 완성한다.

따라서 full tile의 한 fidelity phase에 필요한 instruction 수는 다음과 같다.

$$
4\ \text{output faces}
\times
2\ \text{K-face products}
\times
2\ \text{row halves}
=16\ \text{MVMULs}.
$$

`MVMUL` instruction 하나의 연산량은 $128\times16\times2=4096=2^{12}$ FLOP이다.

### LLK의 실제 face 순서

16개 `MVMUL` instruction은 다음 순서로 여덟 개의 8×16 `Dst` 출력 block을 방문한다.

| `MVMUL` 번호 | `SrcB @ SrcA` | 논리 face product | 갱신하는 출력 |
| ---: | --- | --- | --- |
| 1–2 | `B0 @ A0` | `L0 @ R0` | `C0`의 위·아래 8행 |
| 3–4 | `B0 @ A1` | `L0 @ R1` | `C1`의 위·아래 8행 |
| 5–6 | `B2 @ A0` | `L2 @ R0` | `C2`의 위·아래 8행 |
| 7–8 | `B2 @ A1` | `L2 @ R1` | `C3`의 위·아래 8행 |
| 9–10 | `B1 @ A2` | `L1 @ R2` | `C0`의 위·아래 8행에 추가 |
| 11–12 | `B1 @ A3` | `L1 @ R3` | `C1`의 위·아래 8행에 추가 |
| 13–14 | `B3 @ A2` | `L3 @ R2` | `C2`의 위·아래 8행에 추가 |
| 15–16 | `B3 @ A3` | `L3 @ R3` | `C3`의 위·아래 8행에 추가 |

![한 fidelity phase에서 MVMUL 1–8번이 Dst의 여덟 8×16 block을 방문하고 9–16번이 같은 순서로 다시 방문하는 과정](/assets/img/posts/tenstorrent-blackhole-matmul-mvmul-fidelity/mvmul-face-order.svg)

## MOP와 Replay가 MVMUL을 반복하는 방법

### Replay 목록을 기록하고 MOP로 펼친다

`Replay buffer`는 반복할 instruction 목록을 저장하는 공간이다. 표준 full-tile matmul에서는 fidelity phase 하나를 계산하는 `MVMUL` 16개를 저장한다.

[`REPLAY`](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/REPLAY.md#L1-L19)는 Replay buffer의 어느 위치에서 instruction을 몇 개 내보낼지 지정한다. `REPLAY(offset, count=16)`을 실행하면 Replay Expander가 저장된 `MVMUL` 16개를 내보낸다.

`MOP`(macro-op)은 Tensix frontend에 [미리 설정한 MOP template을 실행](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MOPExpander.md#L1-L13)하는 instruction이다. MOP template에는 반복 횟수 $F$와 반복 본문 `REPLAY(offset, count=16)`이 들어 있다. 이 template은 `MVMUL` 목록을 저장하는 Replay buffer와 별도로 구성된다.

```text
ReplayBuffer
    [MVMUL 0, MVMUL 1, ..., MVMUL 15]
                    ^
                    |
            REPLAY(offset, count=16)
                    ^
                    |
            MOP repeats REPLAY F times
```

아래에서 위로 읽으면 MOP가 `REPLAY`를 $F$번 반복하고, 각 `REPLAY`가 Replay buffer에 저장된 `MVMUL` 16개를 내보낸다. 즉 MOP는 반복 횟수를, `REPLAY`는 저장 목록의 범위를, Replay buffer는 실제 `MVMUL` 목록을 담당한다.

- **설정 단계([`matmul_init()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/compute/matmul.h#L95-L107))**: AddrMod slot을 설정하고 `MVMUL` 16개를 Replay buffer에 기록한다. 이어서 `REPLAY`를 $F$번 반복하도록 MOP template을 구성한다.
- **실행 단계([`matmul_tiles()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/compute/matmul.h#L126-L135))**: Tile pair를 `SrcB`와 `SrcA`에 unpack하고 `MOP` instruction 하나를 실행한다. 이후 frontend가 $F\times16$개의 `MVMUL`을 내보내 `Dst`에 누적한다.

```text
MOP configuration
  |
  +--> program AddrMod slots
  |      move or reset SrcA / SrcB / Dst positions after MVMUL
  |
  +--> record 16 MVMULs in ReplayBuffer
  |      each MVMUL selects an AddrMod slot
  |
  +--> program MOP template
         repeat REPLAY for F fidelity phases

One tile-pair execution
  |
  +--> Unpack: L -> SrcB, R -> SrcA
  +--> Math RISC-V issues one MOP
          |
          v
      MOP Expander: repeat REPLAY F times
          |
          v
      Replay Expander: emit MVMUL[0..15]
          |
          v
      Matrix Unit: accumulate into Dst
```

표준 full-tile template을 수도 코드로 옮기면 다음과 같다.

```text
F = 1 for LoFi, or 2 / 3 / 4 for HiFi2 / HiFi3 / HiFi4

MOP template 1:
  outer_count = 1
  repeat phase F times:
    REPLAY(offset, count=16)
      MVMUL 0
      MVMUL 1
      ...
      MVMUL 15 with ADDR_MOD_5
  end_op:
    reset the reused source state for high fidelity
```

MOP template의 반복 횟수는 LoFi, HiFi2, HiFi3, HiFi4에서 각각 1, 2, 3, 4가 된다. [`ckernel_template::run()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_template.h#L317-L320)은 runtime에 `MOP` instruction 하나만 내보내며, MOP가 내부에서 `REPLAY`를 해당 횟수만큼 반복한다.

### Fidelity별 instruction 수

| 설정 | 실행 phase | 32×32 tile pair당 `MVMUL` |
| --- | --- | ---: |
| `LoFi` | 0 | 16 |
| `HiFi2` | 0, 1 | 32 |
| `HiFi3` | 0, 1, 2 | 48 |
| `HiFi4` | 0, 1, 2, 3 | 64 |

## Fidelity phase가 추가 mantissa bit를 계산하는 원리

### 5-bit×7-bit는 유효숫자부 multiplier의 폭이다

Floating-point 곱셈은 개념적으로 sign, exponent, significand(유효숫자부)를 나눠 처리한다. 정규화된 `in0` 값 $x$와 `in1` 값 $y$는 다음처럼 나타낼 수 있다. 여기서 $m_x$와 $m_y$는 hidden bit `1`과 explicit mantissa를 합친 유효숫자부다.

$$
x=(-1)^{s_x}2^{e_x}m_x,
\qquad
y=(-1)^{s_y}2^{e_y}m_y.
$$

[공식 matrix-engine report](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tech_reports/matrix_engine/matrix_engine.md#L54-L62)가 말하는 5-bit×7-bit는 **전체 floating-point 형식의 폭이 아니다.** 한 phase에서 multiplier가 최대로 소비하는 유효숫자부 조각의 폭이다.

| 입력 | Phase 0에서 사용하는 유효숫자부 | 폭 |
| --- | --- | ---: |
| `in0`의 $x_H$ | hidden bit `1` + 상위 explicit mantissa 6 bit | 7 bit |
| `in1`의 $y_H$ | hidden bit `1` + 상위 explicit mantissa 4 bit | 5 bit |

Sign은 결과의 부호를 정하고 exponent는 partial product의 크기를 정하며, 5×7 multiplier는 그와 별도로 유효숫자부 조각을 곱한다.

[Matrix Unit functional model](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md#L100-L120)은 원래 값의 하위 mantissa bit를 버려 high 근삿값 $x_H$, $y_H$를 만든다. Low residual $x_L=x-x_H$, $y_L=y-y_H$는 원래 값에서 high 부분을 뺀 값이므로, 잘려 나간 하위 mantissa의 값만 남는다. 이 residual을 정규화하면 남은 mantissa가 유효숫자부 앞쪽으로 이동하고, 원래 자리값을 보존하도록 exponent는 그만큼 낮아진다.

`in0`의 값 $x$와 `in1`의 값 $y$를 각각 $x=x_H+x_L$, $y=y_H+y_L$로 나누면 두 값의 곱은 다음 네 항으로 분해된다.

$$
xy
=
x_Hy_H
+x_Hy_L
+x_Ly_H
+x_Ly_L.
$$

공유 [`MVMUL` functional model](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md#L77-L119)에서 fidelity RWC의 bit 0은 `in1` high·low를, bit 1은 `in0` high·low를 고른다. 네 fidelity phase는 위의 네 항에 그대로 대응한다.

| Phase | Fidelity RWC | `in0` 조각 | `in1` 조각 | 이 phase부터 필요한 설정 |
| ---: | --- | --- | --- | --- |
| 0 | `00` | `x_H` | `y_H` | LoFi |
| 1 | `01` | `x_H` | `y_L` | HiFi2 |
| 2 | `10` | `x_L` | `y_H` | HiFi3 |
| 3 | `11` | `x_L` | `y_L` | HiFi4 |

### BF16 예제로 본 네 phase

다음 그림은 BF16의 sign 1 bit, exponent 8 bit, explicit mantissa 7 bit를 모두 펼쳐 보여 준다. 유효숫자부에는 hidden bit `1`을 추가했다. `in0`은 7-bit 쪽, `in1`은 5-bit 쪽이며, 각 phase는 하나의 7×5 multiplier block으로 표시한다. BF16 bit가 부족한 multiplier lane은 `0`으로 채웠다. 회색 `0`은 해당 lane에 기여하는 BF16 bit가 없다는 표시일 뿐, low 조각이 원래 exponent를 유지한 채 zero-padding된다는 뜻은 아니다.

![BF16의 in0과 in1을 sign, exponent, mantissa bit로 펼치고 네 fidelity phase를 7×5 격자로 나타낸 그림](/assets/img/posts/tenstorrent-blackhole-matmul-mvmul-fidelity/fidelity-mop.svg)

_그림. BF16의 sign과 exponent를 bit 단위로 표시하고, 유효숫자부의 네 fidelity phase를 각각 7×5 block으로 나타낸 자체 제작 개념도다. Sign은 XOR, exponent는 덧셈으로 결합하며 7·5 bit에는 포함되지 않는다. 회색 cell은 BF16에 없는 multiplier 입력을 채운 `0`이다. 근거: Tenstorrent [Matrix Engine report](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tech_reports/matrix_engine/matrix_engine.md#L54-L62), [`SrcA`·`SrcB` fidelity 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md#L87-L99), [Matrix Unit functional model](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md#L100-L120). 표현 방식은 [Corsix의 bit-grid 설명](https://www.corsix.org/content/tt-wh-part7)을 참고했다._

### Math fidelity와 누산 정밀도는 별개다

High fidelity는 곱셈에 사용하는 입력 정밀도를 높이는 설정이며, `Dst`의 누산 정밀도와는 별개다. `Dst`는 기본적으로 16-bit 누산을 사용한다. 별도 옵션(`fp32_dest_acc_en=true`)을 사용하면 32-bit 누산을 사용할 수 있다. 자세한 내용은 [FP32 accuracy 공식 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.html#host-side-configuration)를 참고한다.

## 핵심 정리

- 논리 왼쪽 tile $L$은 `SrcB`, 오른쪽 tile $R$은 `SrcA`에 놓이며 Matrix Unit은 `Dst += SrcB @ SrcA`를 수행한다.
- `MVMUL` primitive는 8×16 · 16×16 → 8×16이다. 32×32 tile의 phase 하나는 네 출력 face, 두 K-face product, 두 row-half 때문에 `MVMUL` 16개가 필요하다.
- MOP는 `REPLAY(16 MVMUL)`를 fidelity phase 수만큼 반복한다. LoFi부터 HiFi4까지 `MVMUL` 수는 tile pair당 16, 32, 48, 64개다.
- 네 phase는 high 근삿값과 low residual의 $HH$, $HL$, $LH$, $LL$ cross-product를 순서대로 `Dst`에 더한다.
- Math fidelity와 `Dst` 누산 정밀도는 별도 설정이다. `Dst`는 기본적으로 16-bit 누산을 사용하며, 옵션에 따라 32-bit 누산을 사용할 수 있다.

## 참고 자료

- Tenstorrent, [TT-Metalium compute engine과 register dataflow](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html)
- Tenstorrent, [FP32 accuracy와 FPU·SFPU 설정](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.html)
- Tenstorrent, [`Matrix Engine`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tech_reports/matrix_engine/matrix_engine.md#L9-L81)
- Tenstorrent, [Compute kernel `matmul.h`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/compute/matmul.h#L100-L141)
- Tenstorrent, [Blackhole `llk_math_matmul.h`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h#L298-L448)
- Tenstorrent, [Blackhole `llk_unpack_AB_matmul.h`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h#L22-L46)
- Tenstorrent, [Blackhole `ckernel_template.h`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_template.h#L343-L368)
- Tenstorrent, [`MVMUL` primitive](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md#L1-L17)과 [fidelity functional model](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md#L87-L132)
- Tenstorrent, [`SrcA`·`SrcB` data type과 fidelity phase](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md#fidelity-phases-floating-point)
- Tenstorrent, [Matrix Unit fidelity helper functional model](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md#L100-L120)
- Tenstorrent, [MOP Expander template 1](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/MOPExpander.md#L58-L105)
- Tenstorrent, [`REPLAY`](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/REPLAY.md#L1-L19)
- Peter Cawley, [Tenstorrent Wormhole Series Part 7: Bits of the MatMul](https://www.corsix.org/content/tt-wh-part7)
