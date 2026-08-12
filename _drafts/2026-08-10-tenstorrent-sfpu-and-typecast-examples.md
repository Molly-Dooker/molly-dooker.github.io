---
title: "Tenstorrent SFPU 기초와 Typecast 예제"
date: 2026-08-10 12:49:38 +0900
categories: [Tenstorrent, llk]
tags: [llk]
description: "Tenstorrent Blackhole의 SFPI, SFPLOAD, 32-lane tile mapping과 scheduling을 살펴보고 i32·FP32·BF16 Typecast 구현을 예제로 분석합니다."
render_with_liquid: false
---

## 문서 범위

- hardware 설명 기준: Blackhole의 일반적인 full-tile SFPU 경로
- 기준 구현: `tt-metal` commit [`69096826694c`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e), Blackhole
- 기준 ISA(Instruction Set Architecture): `tt-isa-documentation` commit [`b7738d9ac14a`](https://github.com/tenstorrent/tt-isa-documentation/tree/b7738d9ac14a34a4033d60dde9463466b23082e1), Blackhole A0
- architecture 개요: [TT-Metalium compute engine 공식 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html)
- 보조 참고: Jason Davies, [Typecast on Tenstorrent](https://www.jasondavies.com/2025/tenstorrent-typecast/)
- 확인일: 2026-08-13
- 선수 지식: [숫자와 데이터 표현 기초](/posts/number-and-data-representation-basics/), [Tenstorrent LLK 기초](/posts/tenstorrent-llk-basics/)

이 글은 SFPU(Special Function Processing Unit)의 register와 instruction 실행 방식을 먼저 설명한 뒤, 현재 `tt-metal`의 Blackhole LLK(Low-Level Kernel)에 있는 다음 두 Typecast를 예제로 분석한다.

1. `i32 -> BF16`: two's-complement 정수를 floating-point로 바꾸고 BF16 정밀도로 줄이는 경로
2. `fp32/BF16 -> i32`: exponent와 mantissa를 조합해 소수부를 0 방향으로 버리는 경로

Jason Davies의 글은 변환을 설계한 아이디어와 `SFPLOADMACRO` schedule을 이해하는 데 참고한다. 실제 instruction, register와 분기 구조는 위에 고정한 `tt-metal` revision을 기준으로 설명한다.

RISC-V core와 Tensix thread, unpacker·packer, FPU·SFPU의 공통 dataflow와 32-bit instruction word는 [LLK 기초 글](/posts/tenstorrent-llk-basics/)에서 설명한다. Bit pattern과 shift, ties-to-even, two's-complement는 [숫자와 데이터 표현 기초](/posts/number-and-data-representation-basics/)에서 다룬다. 이 글은 그 위에서 SFPI, `SFPLOAD`, tile·face·32-lane mapping, `VectorMode`와 SFPU scheduling을 구체적으로 내려가는 SFPU 심화편이다. 같은 공통 구조에서 matrix engine(FPU)을 따라가는 “Tenstorrent Blackhole FPU와 Matmul: MVMUL 분할과 Math Fidelity”는 `MVMUL`, MOP·Replay와 fidelity phase를 다룬다.

## SFPU가 compute dataflow에서 맡는 역할

[공식 compute engine 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#vector-engine-sfpu)에 따르면 SFPU는 `sin`, `cos`, `exp`, `relu`, `tanh`와 Typecast 같은 elementwise 연산에 적합한 vector engine이다. Matrix engine인 FPU가 주로 `SrcA`와 `SrcB`를 읽어 `Dst`에 기록하는 것과 달리, SFPU는 `Dst`를 source이자 destination으로 사용한다.

[`math_fidelity`는 matrix engine에만 적용](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.html#host-side-configuration)되며 SFPU instruction을 fidelity phase로 반복시키지 않는다. 일부 SFPU 함수의 근사 implementation을 선택하는 `math_approx_mode`는 이와 별도의 설정이다.

```text
L1 input tile
      |
      | Unpacker / copy_tile
      v
+-------------------+
| Dst register set  |
+---------+---------+
          |
          | SFPLOAD
          v
+-------------------+
| SFPU LRegs        |
| 32 lanes x 32 bit |
+---------+---------+
          |
          | vector instructions
          v
+-------------------+
| updated LRegs     |
+---------+---------+
          |
          | SFPSTORE
          v
+-------------------+
| Dst register set  |
+---------+---------+
          |
          | Packer / pack_tile
          v
L1 output tile
```

이 글에서 사용하는 register와 address 표기는 다음과 같다.

| 표기 | 종류 | 의미 |
| --- | --- | --- |
| `Dst` | shared destination register set | FPU의 결과를 보관하며, SFPU에는 입력과 출력 공간이다. L1 SRAM과는 별도다. |
| `Dst` tile slot, face, row | `Dst` 내부 영역 | 각각 tile 하나가 놓이는 영역과 그 안의 face·hardware row를 가리킨다. |
| `LReg[VD]` | SFPU local vector register | `VD`가 선택한 LReg다. 일반적인 LReg 하나는 32 lanes × 32 bits다. |
| `RWCs` | thread별 address counter 묶음 | `Dst`, `Dst_Cr`, `SrcA`, `SrcB` 등의 counter를 묶는다. ISA functional model은 현재 thread의 항목을 `RWCs[CurrentThread]`로 표기한다. |
| `RWC` | 이 글의 축약 표기 | `RWCs[CurrentThread]` 하나를 짧게 나타낸다. |
| `RWC.Dst` | 10-bit address counter | `RWCs[CurrentThread].Dst`이며, `Dst`의 현재 접근 주소를 추적한다. |
| `RWC.Dst_Cr` | 별도의 10-bit address counter | `RWC.Dst`가 face 안의 group을 순회하는 동안 현재 face의 시작 주소를 유지한다. |

`VD`는 vector destination의 약자다. 보통 SFPU instruction이 결과를 기록할 LReg index를 나타내지만, `SFPSTORE`에서는 같은 이름의 field가 `Dst`에 기록할 source LReg를 고른다. 따라서 이름만 보고 읽기·쓰기 방향을 단정하지 말고 해당 instruction의 ISA를 확인해야 한다.

### SFPI란?

SFPI(SFPU Interface)는 SFPU program을 C++ 형태로 작성하기 위한 library와 compiler다. [공식 Low Level Kernels 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.html#overview)는 SFPI를 vector data type과 builtin intrinsic을 추가한 RISC-V GCC 기반 compiler의 C++ wrapper로 설명한다. `sfpi::`는 SFPI가 제공하는 C++ namespace다.

```text
SFPI C++ code
     |
     | SFPI compiler
     v
SFPLOAD, SFPCAST, SFPADD, SFPSTORE, ...
     |
     v
SFPU hardware
```

SFPI의 C++ 표현 하나가 hardware instruction 하나와 일대일로 대응하는 것은 아니다. Compiler는 필요한 load·연산·store instruction을 만들고 LReg 할당과 scheduling도 처리한다. 예를 들어 [`dst_reg[]`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.html#user-visible-objects)는 `Dst`를 LReg 크기의 vector 단위로 읽고 쓰는 interface다. `v_if` 계열은 RISC-V branch가 아니라 lane predication으로 양쪽 vector 경로를 실행한 뒤 활성 lane의 write만 허용한다.

## `SFPLOAD`와 instruction encoding

`SFPLOAD`는 `Dst`의 값을 SFPU의 LReg로 읽는 instruction mnemonic이다. `VD`, `Mod0`, `AddrMod`, `Imm10`은 operand field다. [`TT_OP_SFPLOAD(...)`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h#L670-L676)가 이 field들을 하나의 32-bit instruction word로 조합하고, `TTI_SFPLOAD(...)`가 instruction으로 내보낸다.

```text
LLK / SFPI code
      |
      v
SFPLOAD + settings        VD, Mod0, AddrMod, Imm10
      |
      | encode into one word
      v
32-bit Tensix instruction 0x........
      |
      | Math RISC-V sends
      v
SFPU executes SFPLOAD
```

[`SFPLOAD` ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md#syntax)는 다음 operand를 정의한다.

```text
SFPLOAD(VD: u4, Mod0: u4, AddrMod: u3, Imm10: u10)
```

이를 bit 위치에 배치하면 다음과 같다. 이름이 없는 `[12:10]` bit는 아래 예에서 0으로 둔다.

```text
bit     31..24  23..20  19..16  15..13   12..10   9..0
       +-------+-------+-------+--------+--------+-------+
field  | 0x70  | VD    | Mod0  |AddrMod |   -    | Imm10 |
       +-------+-------+-------+--------+--------+-------+
```

각 field의 역할은 다음과 같다.

| Field | 의미 |
| --- | --- |
| `0x70` | 이 word가 `SFPLOAD`임을 나타내는 opcode |
| `VD` | `Dst`에서 읽은 값을 기록할 `LReg[VD]` index |
| `Mod0` | `Dst` 값을 BF16·FP16·FP32 등 어떤 type으로 읽고 LReg의 32-bit lane 값으로 변환할지 선택 |
| `AddrMod` | `instruction` 실행 후 적용할 RWC 갱신 규칙을 선택하는 3-bit index |
| `Imm10` | 이번 load의 effective `Dst` address에만 더할 10-bit immediate offset |

### `Imm10`과 `AddrMod`가 적용되는 순서

`Imm10`과 `AddrMod`는 같은 instruction에 들어 있지만 서로 다른 시점에 작동한다. 아래 그림은 특수한 address remap을 제외한 일반 Typecast 경로를 단순화한 것이다. `T`는 현재 `Dst` tile의 시작점, `C`는 instruction 실행 전의 10-bit counter `RWC.Dst` 값, `I`와 `n`은 각각 instruction에 담긴 `Imm10`과 `AddrMod`다.

```text
One SFPLOAD execution
  |
  +--> before issue
  |      tile base T, RWC.Dst = C
  |      Imm10 = I, AddrMod = n
  |
  +--> build this load address with Imm10
  |      Addr = T + C + I
  |
  +--> execute SFPLOAD
  |      Dst[Addr] -> LReg[VD]
  |
  +--> apply AddrMod after the load
         Rule = AddrModSlot[n]
         Next RWC.Dst = keep, increment, or reset by Rule
              |
              v
       Next instruction starts from the updated RWC.Dst
```

예를 들어 `T=0`, 실행 전 `RWC.Dst=4`라고 하자. 현재 Typecast 설정에서 slot 7은 `RWC.Dst`를 유지하고, slot 6은 2 증가시킨다. 

아래 표의 각 행은 연속 실행이 아니라 모두 이 초기 상태에서 시작하는 독립적인 예다.

| `Imm10` | `AddrMod` | 이번에 읽는 주소 | 실행 후 `RWC.Dst` |
| ---: | ---: | ---: | ---: |
| 8 | 7 | `0 + 4 + 8 = 12` | 4 |
| 0 | 6 | `0 + 4 + 0 = 4` | 6 |
| 8 | 6 | `0 + 4 + 8 = 12` | 6 |

전체 instruction word는 예를 들어 `VD=0`, `Mod0=2`(`MOD0_FMT_BF16`), `AddrMod=7`, `Imm10=0`이면 다음 word를 얻는다.

```text
instruction word = (0x70 << 24)
                 + (0    << 20)   // VD
                 + (2    << 16)   // Mod0: BF16
                 + (7    << 13)   // AddrMod
                 + 0              // Imm10
                 = 0x7002e000
```

여기서 `Mod0=2`는 `Dst`의 BF16 값을 LReg의 FP32 값으로 변환한다. `Imm10=0`은 이번 주소에 offset을 더하지 않는다. `AddrMod=7`은 현재 Typecast 설정에서 [`RWC.Dst`를 바꾸지 않는 rule 7](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h#L20-L34)을 선택한다. `0x7002e000` 전체가 instruction word다.

## 32×32 tile에서 32-lane SFPU vector까지

이 절은 transpose, `Dst` address remap, `DEST_RD_COL_EXCHANGE` 같은 특수 설정을 사용하지 않는 Blackhole의 일반적인 full-tile elementwise 경로를 기준으로 한다. 일반적인 tile이 32×32 element와 16×16 face 네 개로 구성된다는 큰 구조는 [공식 tile 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/tiles.html#internal-structure-of-a-tile)를 따른다. 정확한 lane과 address mapping은 위에 고정한 Blackhole [`SFPLOAD` functional model](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md#functional-model)을 기준으로 한다.

Unpacker는 L1 SRAM의 32×32 data tile을 16×16 face 네 개로 나눠 `Dst` tile slot 하나에 배치한다. Logical element 1,024개는 모두 `Dst`에 있지만, SFPU local register 하나에 한꺼번에 들어가지는 않는다. `SFPLOAD`는 `Dst`에서 특정 위치 32개만 골라 local register의 lane 32개로 읽는다.

```text
32x32 data tile in L1 SRAM
          |
          | Unpacker
          v
One tile slot in Dst
+-------------------+-------------------+
| Face 0            | Face 1            |
| tile rows  0..15  | tile rows  0..15  |
| tile cols  0..15  | tile cols 16..31  |
+-------------------+-------------------+
| Face 2            | Face 3            |
| tile rows 16..31  | tile rows 16..31  |
| tile cols  0..15  | tile cols 16..31  |
+-------------------+-------------------+
          |
          | SFPLOAD: select 32 elements
          v
LReg[k][0..31]
          |
          | lanewise SFPU instructions
          v
32 lanes execute the same instruction
```

Lane과 element는 서로 다른 개념이지만 일반적인 lanewise instruction에서는 1:1로 대응한다.

- **lane**은 같은 vector instruction을 독립적으로 수행하는 hardware 실행 위치다.
- **element**는 해당 lane에 들어가 처리되는 data 하나다.

모든 lane이 활성화돼 있다면 SFPU vector instruction 하나가 element 32개를 처리한다.

```text
32 lanes x 1 element per lane = up to 32 elements

LReg[k][ 0] -> lane  0 processes element  0
LReg[k][ 1] -> lane  1 processes element  1
...
LReg[k][31] -> lane 31 processes element 31
```

Predication으로 일부 lane을 끄면 실제 연산하는 element는 32개보다 적어진다. BF16이나 `u16`처럼 원래 폭이 32 bit보다 작은 element도 LReg에서는 lane마다 32-bit 공간 하나를 차지한다. 필요한 변환이나 확장은 `SFPLOAD` 과정에서 적용된다.

일반적인 `SFPLOAD` 한 번은 16×16 face에서 **연속된 `Dst` row 네 개 × 짝수 또는 홀수 column 여덟 개**를 읽는다. 4×16 영역에서 column을 하나씩 건너뛴 4×8 영역을 고르는 셈이며, 이렇게 모은 `4×8=32`개 element가 SFPU vector 하나를 이룬다.

Face의 row 0..3을 예로 들면 다음과 같다.

```text
              face column
           0  1  2  3  4  5  ... 14 15
row 0      E  O  E  O  E  O  ...  E  O    // E: Even; O: Odd
row 1      E  O  E  O  E  O  ...  E  O
row 2      E  O  E  O  E  O  ...  E  O
row 3      E  O  E  O  E  O  ...  E  O

Vector 0, even columns:
  lane  0..7  <- row 0, col 0,2,4,...,14
  lane  8..15 <- row 1, col 0,2,4,...,14
  lane 16..23 <- row 2, col 0,2,4,...,14
  lane 24..31 <- row 3, col 0,2,4,...,14

Vector 1, odd columns:
  lane  0..7  <- row 0, col 1,3,5,...,15
  lane  8..15 <- row 1, col 1,3,5,...,15
  lane 16..23 <- row 2, col 1,3,5,...,15
  lane 24..31 <- row 3, col 1,3,5,...,15
```

따라서 한 LReg의 lane layout은 logical tile row와 같은 `1×32`보다 다음 `4×8` 형태로 보는 편이 정확하다.

```text
lane  0  1  2  3  4  5  6  7
lane  8  9 10 11 12 13 14 15
lane 16 17 18 19 20 21 22 23
lane 24 25 26 27 28 29 30 31
```

### `Dst`에서 4×8 group을 선택하는 address

`SFPLOAD` functional model의 `Addr`는 element 하나가 아니라 LReg 하나에 담을 32개 element의 배치를 고르는 10-bit effective `Dst` address다. 이 값은 다음 두 가지를 정한다.

1. 높이 4인 column 조각이 시작할 `Dst` row `R`은 어디인가?
2. 짝수 column 여덟 개와 홀수 column 여덟 개 중 어느 쪽을 고를 것인가?

따라서 `SFPLOAD` 한 번은 **높이 4인 column 조각을 한 칸씩 건너뛰며 8개 읽는다.** 선택한 column 자체는 서로 붙어 있지 않지만, 각 column 안에서는 row `R`부터 `R+3`까지 네 값이 연속된다.

같은 32개 좌표를 row 방향에서 표현하면 “`R..R+3`의 연속된 row 네 개에서 짝수 또는 홀수 column 여덟 개를 고른다”가 된다. 이는 row 네 줄의 element를 모두 읽는다는 뜻이 아니다. Column 기준 설명과 row 기준 설명은 같은 `4×8` 영역을 서로 다른 방향에서 본 것이다.

`R=0`, 짝수 column을 고르는 경우는 다음과 같다. `X`가 이번 load에서 선택하는 element고, `.`은 건너뛰는 element다.

```text
             Dst column
          0  1  2  3  4  5  ... 14 15
        +--+--+--+--+--+--+-----+--+--+
row 0   | X| .| X| .| X| .| ... | X| .|
row 1   | X| .| X| .| X| .| ... | X| .|
row 2   | X| .| X| .| X| .| ... | X| .|
row 3   | X| .| X| .| X| .| ... | X| .|
        +--+--+--+--+--+--+-----+--+--+

selected: 8 columns x 4 rows = 32 elements
```

따라서 tile의 element 개수, SFPI group index, hardware address unit을 서로 다른 숫자로 봐야 한다.

| 숫자 | 의미 | 한 tile에서의 범위 |
| --- | --- | ---: |
| logical element index | 32×32 tile 안의 값 하나 | 0..1023 |
| `sfpi::dst_reg[i]`의 `i` | `Dst`를 32-element vector 단위로 보는 SFPI index | 0..31 |
| tile-relative `Addr` | 4-row block과 column parity를 고르는 effective address | 0, 2, ..., 62 |

1,024개 element를 주소 1,024개로 직접 순회하는 구조가 아니다. `SFPLOAD` 32번이 매번 element 32개를 읽어 tile 전체를 덮는다.

```text
32x32 tile: 1024 elements
 |
 +-- Face 0: 8 groups
 +-- Face 1: 8 groups
 +-- Face 2: 8 groups
 +-- Face 3: 8 groups
 |
 +-- 32 groups x 32 lanes = 1024 elements
```

16×16 face 하나에는 row 네 개로 이루어진 구간이 네 개 있다. 각 구간에서 짝수 column group과 홀수 column group을 한 번씩 고르므로 face 하나는 group 여덟 개로 나뉜다. `B`는 그 face의 시작 주소다.

```text
One 16x16 face

face rows       even columns              odd columns
 0.. 3          group 0, Addr B+0         group 1, Addr B+2
 4.. 7          group 2, Addr B+4         group 3, Addr B+6
 8..11          group 4, Addr B+8         group 5, Addr B+10
12..15          group 6, Addr B+12        group 7, Addr B+14
```

`B+0`은 row 0..3을 세로로 잇는 짝수 column 조각 여덟 개를 읽는다. 다음 주소 `B+2`는 같은 높이의 홀수 column 조각 여덟 개를 고른다. Load 두 번을 합치면 face row 0..3의 `4×16=64`개 값을 모두 처리한다.

LReg에는 첫 row의 선택값 여덟 개부터 차례대로 넣는 row-major 순서로 모은다. 같은 선택을 lane 관점에서 쓰면 lane 0..7이 첫 row, lane 8..15가 다음 row를 담당한다.

```text
Addr B+0                                  Addr B+2

row 0: c0 c2 c4 ... c14 -> lane  0.. 7   c1 c3 c5 ... c15 -> lane  0.. 7
row 1: c0 c2 c4 ... c14 -> lane  8..15   c1 c3 c5 ... c15 -> lane  8..15
row 2: c0 c2 c4 ... c14 -> lane 16..23   c1 c3 c5 ... c15 -> lane 16..23
row 3: c0 c2 c4 ... c14 -> lane 24..31   c1 c3 c5 ... c15 -> lane 24..31
```

`SFPLOAD`가 최종적으로 사용하는 `Addr`는 10 bit이며, `Dst` register set에서는 0부터 1,023까지 표현할 수 있다.

```text
Addr bit      | 9 8 7 6 5 4 3 2 | 1 | 0 |
              +-----------------+---+---+
field         | 4-row block q   | p | x |
              +-----------------+---+---+

row base      = q * 4
column parity = p        (0: even, 1: odd)
x             = unused
```

`Addr[9:2]`는 bit 9부터 bit 2까지를 묶은 bit-slice다. 이 값 `q`가 연속된 row 네 개의 시작점을 정한다. `Addr[1]`인 `p`는 짝수·홀수 column을 고르고, `Addr[0]`은 사용하지 않는다. 따라서 `B+0`과 `B+1`은 같은 위치를 선택하며, 다음으로 서로 다른 group을 고르는 최소 증가량이 2다.

Lane별 실제 위치는 다음처럼 계산한다. `/`는 여기서 나머지를 버리는 정수 나눗셈이다.

```text
row    = (q * 4) + (lane / 8)
column = ((lane % 8) * 2) + p
```

실제 instruction의 유효 주소에는 현재 tile의 시작점 `T`도 들어간다. [`set_dst_write_addr()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h#L224-L245)는 tile index에 따라 `T`를 설정한다. Typecast가 사용하는 일반 경로의 주소 계산은 다음처럼 단순화할 수 있다.

```text
effective Dst address = tile base T + RWC.Dst + Imm10
```

세 항의 역할은 다르다.

- `T`: 이번에 처리할 `Dst` tile slot의 시작점
- `RWC.Dst`: tile 안의 현재 group을 기억하는 10-bit counter
- `Imm10`: 이번 instruction에만 적용하는 10-bit immediate offset. 이 Typecast leaf에서는 0

`AddrMod`가 이 식에 없는 이유는 현재 load address를 만드는 항이 아니기 때문이다. 명령이 끝난 뒤 선택한 rule에 따라 `RWC.Dst` 같은 counter를 갱신한다. 일반 Typecast loop에서는 `SFPLOAD`의 `ADDR_MOD_7`이 현재 위치를 유지하고, 같은 group을 `Dst`에 기록하는 `SFPSTORE`의 `ADDR_MOD_6`이 `RWC.Dst`를 2 증가시킨다.

`SFPLOADMACRO` 경로에서는 load 단계가 `ADDR_MOD_6`으로 `RWC.Dst`를 2 증가시킨다. 나중에 실행되는 예약 `SFPSTORE`는 현재 RWC를 다시 읽지 않고 macro가 load할 때 계산한 주소를 사용한다. 이 store에는 address modifier도 다시 적용하지 않으므로, RWC가 이미 다음 group을 가리켜도 load와 같은 group에 결과를 기록한다.

Tile-relative 주소만 놓고 보면 네 face는 다음처럼 이어진다. 한 face의 유효 group 주소는 마지막이 `B+14`지만, 다음 face base는 `B+16`이다.

```text
Face 0: RWC.Dst  0,  2,  4,  6,  8, 10, 12, 14
Face 1: RWC.Dst 16, 18, 20, 22, 24, 26, 28, 30
Face 2: RWC.Dst 32, 34, 36, 38, 40, 42, 44, 46
Face 3: RWC.Dst 48, 50, 52, 54, 56, 58, 60, 62
```

### `VectorMode`로 tile의 face를 선택하는 방법

앞 절에서 `RWC.Dst`가 face 안의 group 여덟 개를 순회하는 방법을 봤다. 이제 남은 질문은 **누가 다음 face를 선택하는가**다. Typecast 경로에서는 역할이 다음처럼 나뉜다.

| 구성 요소 | 맡은 일 |
| --- | --- |
| Typecast leaf의 `ITERATIONS=8` | 현재 face의 group 여덟 개 처리 |
| LLK wrapper의 `VectorMode` | leaf를 호출할 face 선택 |
| `RWC.Dst` | 현재 group을 가리키는 counter |
| `RWC.Dst_Cr` | `RWC.Dst`와 별도인 companion counter. 이 wrapper에서는 현재 face의 시작 주소를 유지 |

`VectorMode`는 hardware vector instruction의 mode가 아니라 LLK wrapper option이다. Wrapper는 `RWC.Dst`를 선택한 face의 시작점에 맞춘 뒤 leaf를 호출한다.

```text
Tile face layout

               left half        right half
             +---------------+---------------+
top half     | Face 0        | Face 1        |
             +---------------+---------------+
bottom half  | Face 2        | Face 3        |
             +---------------+---------------+

R  calls:      Face 0 -> Face 1
C  calls:      Face 0 ------------> Face 2
RC calls:      Face 0 -> Face 1 -> Face 2 -> Face 3
```

이름에서 `R`은 row, `C`는 column을 가리킨다. [`_llk_math_eltwise_sfpu_apply_vector_mode_()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h#L45-L84)는 `R`에서 Face 0·1, `C`에서 Face 0·2, `RC`에서 네 face를 모두 고른다.

Typecast wrapper는 face 이동의 기준을 leaf가 바꾸는 `RWC.Dst`와 분리하기 위해 `RWC.Dst_Cr`를 사용한다. Face 시작 주소를 `B`라고 하자. Leaf가 group 여덟 개를 처리하는 동안 `RWC.Dst`만 움직이고 `RWC.Dst_Cr`는 `B`를 유지한다.

```text
Before leaf
  RWC.Dst_Cr = B
  RWC.Dst    = B
        |
        v
Leaf accesses
  B+0 -> B+2 -> B+4 -> B+6 -> B+8 -> B+10 -> B+12 -> B+14
        |
        | final iteration advances RWC.Dst by 2
        v
After leaf
  RWC.Dst_Cr = B
  RWC.Dst    = B+16
```

Leaf가 끝난 시점의 `RWC.Dst`는 다음 face base인 `B+16`과 같다. 그래도 wrapper는 이 값에 의존하지 않고 `RWC.Dst_Cr`를 기준으로 두 counter를 다시 맞춘다. [`_llk_math_eltwise_sfpu_inc_dst_face_addr_()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h#L28-L32)가 `inc_dst_addr<8>()`을 두 번 호출하는 과정은 다음과 같다.

```text
CR_D step 1: RWC.Dst_Cr B   + 8 -> RWC.Dst_Cr = B+8,  RWC.Dst = B+8
CR_D step 2: RWC.Dst_Cr B+8 + 8 -> RWC.Dst_Cr = B+16, RWC.Dst = B+16
                                                    |
                                                    v
                                              next face leaf
```

첫 단계에서 `RWC.Dst`가 `B+16`에서 `B+8`로 잠시 돌아가지만, 두 단계 사이에는 `Dst` 접근이 없으므로 문제되지 않는다. [`inc_dst_addr()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h#L255-L260)가 한 instruction에 넣을 수 있는 증가값은 최대 15라서 `+16` 대신 `+8`을 두 번 사용한다. `CR_D` mode는 증가값을 `RWC.Dst_Cr`에 더한 뒤 그 결과를 `RWC.Dst`에도 복사한다.

이 방식이면 wrapper가 leaf 내부의 `RWC.Dst` 이동량을 추측하지 않고도 일정한 face 간격을 만들 수 있다. `VectorMode`별 leaf 시작 위치와 wrapper 종료 위치는 다음과 같다.

| `VectorMode` | Leaf를 시작하는 tile-relative face base | 종료 후 기준값 |
| --- | --- | ---: |
| `R` | 0, 16 | 64 |
| `C` | 0, 32 | 64 |
| `RC` | 0, 16, 32, 48 | 64 |

`C`에서는 Face 0 뒤에 face step을 두 번 실행해 Face 1을 건너뛰고 Face 2로 간다. `R`에서는 Face 0·1을 처리한 뒤 남은 두 face만큼 `RWC.Dst`를 더 옮겨 종료 기준값 64에 맞춘다. 건너뛰는 face에서는 leaf를 호출하지 않는다.

[Full-tile Typecast](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h#L110-L167)는 `VectorMode::RC`, `ITERATIONS=8`을 사용한다. 따라서 wrapper 한 번이 leaf를 네 번 호출하고, 각 leaf가 현재 face의 group 여덟 개를 처리한다.

```text
1 wrapper call
  x 4 face calls
  x 8 groups per face
  x 32 lanes per group
  = 1024 tile elements
```

전체 주소 선택 과정을 한 줄로 줄이면 다음과 같다.

```text
tile_index --selects--> tile base T
VectorMode --selects--> face base B
ITERATIONS --walks----> group offsets 0, 2, ..., 14
Addr bits  --select---> four rows and column parity
lane index --selects--> one element inside the 4x8 group
```

## SFPU 실행 자원과 `SFPLOADMACRO`

`SFPLOADMACRO`의 schedule을 이해하려면 SFPU 실행 자원을 다음 다섯 종류로 나눠 보면 된다.

| 자원 | 대표 역할 |
| --- | --- |
| Load | `SFPLOAD`로 `Dst`의 32-element vector group을 LReg로 읽는다. |
| Simple | bitwise, add, sign 설정, min/max 같은 단순 연산을 수행한다. |
| MAD | multiply-add와 indirect register 선택을 수행한다. |
| Round | shift나 `SFPSTOCHRND` 계열 변환을 수행한다. |
| Store | 결과를 `Dst`에 기록한다. |

[`SFPLOADMACRO` ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md)는 `Dst`의 vector group을 LReg로 load하면서 `SFPCONFIG`로 미리 구성한 짧은 후속 schedule을 시작한다. `MacroIndex`가 schedule을 선택하면 Simple, MAD, Round, Store sub-unit에 각각 최대 한 instruction, 모두 합해 최대 네 instruction을 예약할 수 있다. 예약된 instruction은 software가 다시 issue하지 않아도 설정된 delay에 따라 hardware가 실행한다.

```text
SFPCONFIG
  - instruction templates
  - sub-unit sequence and delays
  - operand overrides
          |
          v
SFPLOADMACRO(MacroIndex, VD, effective Dst address)
  - current cycle : Dst -> LReg
  - future cycles: Simple / MAD / Round / Store
                   (up to one instruction each)
```

여기서 **issue**는 instruction을 실행 pipeline에 투입한다는 뜻이지, 실행을 모두 끝냈다는 뜻이 아니다. `SFPLOADMACRO`를 issue한 cycle에 load가 시작하고, 후속 instruction은 뒤 cycle에 실행될 수 있다. 마지막 macro 뒤의 `SFPNOP`은 새로운 연산을 추가하지 않고 예약된 instruction이 끝날 tail 구간을 채운다.

## 실제 Typecast 호출 흐름

실제 compute kernel은 입력 tile을 `Dst`로 복사한 뒤 Typecast를 수행하고, 결과를 output data format에 맞춰 pack한다. 실행 순서는 [`eltwise_typecast.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp#L22-L40)에서 확인할 수 있다.

```cpp
copy_tile(dfb::in, 0, 0);

TYPECAST_LLK_INIT();
TYPECAST_LLK(0);

pack_tile(0, dfb::out);
```

`TYPECAST_LLK_INIT`과 `TYPECAST_LLK`는 input/output format에 따라 각각 `typecast_tile_init<IN, OUT>`과 `typecast_tile<IN, OUT>`으로 치환된다.

```text
+---------------+       +---------------+
| L1 input tile |------>| copy to Dst   |
+---------------+       +-------+-------+
                                |
                                v
                        +---------------+
                        | SFPU typecast |
                        | in Dst/LRegs  |
                        +-------+-------+
                                |
                                v
                        +---------------+
                        | pack to L1    |
                        +---------------+
```

공개 compute API의 [`typecast.h`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h#L69-L401)는 format 조합에 따라 다음 leaf function을 선택한다.

| 요청한 변환 | Blackhole leaf function |
| --- | --- |
| `Int32 -> Float16_b` | `calculate_typecast_int32_to_fp16b` |
| `Float32 -> Int32` | `calculate_typecast_fp32_to_int32` |
| `Float16_b -> Int32` | `calculate_typecast_fp32_to_int32` |

여기서 `Float16_b`가 BF16이다. `fp32`와 BF16에서 `i32`로 가는 두 branch가 같은 leaf를 호출한다는 점도 중요하다. Leaf 관점에서는 둘 다 `SFPLOAD(DEFAULT)`로 읽은 뒤 같은 `SFPEXEXP`와 `SFPEXMAN` 경로를 거친다.

## 1. `i32 -> BF16`

### 왜 바로 `SFPCAST`할 수 없는가

`SFPLOAD(INT32)`로 읽은 `i32`는 LReg에 two's-complement bit pattern으로 들어온다. LReg 자체에 고정된 type이 있는 것은 아니며, 각 instruction이 register bit를 자신의 규칙에 따라 해석한다. Blackhole의 [`SFPCAST`](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPCAST_IntFloat.md)는 bit 31을 sign, 나머지 31 bit를 magnitude로 읽어 FP32로 변환한다.

예를 들어 `-13`의 두 표현은 다음처럼 다르다.

```text
two's-complement : 0xfffffff3
sign-magnitude   : 0x8000000d
```

따라서 `0xfffffff3`을 그대로 `SFPCAST`하면 `13`이 아니라 `0x7ffffff3`을 magnitude로 읽는다. 현재 leaf는 먼저 `SFPABS`로 two's-complement 절댓값 `0x0000000d`를 만든다. 양수는 two's-complement와 sign-magnitude의 bit pattern이 같으므로, 이 값은 `SFPCAST`에서 올바른 FP32 `+13.0`으로 변환된다.

```text
+-------------------------+
| i32 two's-complement v  |
+------------+------------+
             |
             | SFPABS (integer)
             v
+-------------------------+
| magnitude bits m        |
| INT_MIN: 0x80000000     |
+------------+------------+
             |
             | SFPCAST (sign-magnitude -> FP32)
             v
+-------------------------+
| FP32 magnitude          |
| normal: +abs(v)         |
| INT_MIN: -0.0           |
+------------+------------+
             |
             | SFPSETSGN
             | sign = v[31]
             v
+-------------------------+
| signed FP32             |
+------------+------------+
             |
             | INT_MIN fix (FP32)
             v
+-------------------------+
| corrected FP32          |
+------------+------------+
             |
             | SFPSTOCHRND
             v
+-------------------------+
| BF16 result             |
+-------------------------+
```

`SFPCAST`가 정수에서 FP32로 바뀌는 경계다. 그 다음의 `SFPSETSGN`은 sign-magnitude 정수를 만드는 instruction이 아니다. `SFPCAST` 결과의 FP32 exponent·mantissa와 원래 `i32` 입력의 bit 31을 합쳐 signed FP32를 만든다. `INT_MIN` 보정도 이 FP32 결과의 `-0.0`을 FP32 `-2**31`로 고치는 단계다.

이를 scalar pseudocode로 옮기면 다음과 같다.

```text
magnitude_bits = abs_twos_complement(value)
original_sign  = bit(value, 31)
is_int_min     = bit(magnitude_bits, 31)
magnitude_fp32 = cast_sign_magnitude_int_to_fp32(magnitude_bits)
signed_fp32    = copy_fp32_sign(magnitude_fp32, original_sign)

if is_int_min == 1:
    signed_fp32 = -2**31

result = round_fp32_to_bf16(signed_fp32)
```

### `INT_MIN`만 별도 보정하는 이유

32-bit two's-complement에서 `INT_MIN`은 `-2**31`, bit pattern은 `0x80000000`이다. 양수 `2**31`은 signed i32 범위에 없으므로 `abs(INT_MIN)`도 같은 bit pattern으로 남는다.

```text
abs(0x80000000) = 0x80000000
```

이 값을 sign-magnitude로 해석하면 sign bit만 1이고 magnitude가 0인 negative zero가 된다. 그래서 `SFPCAST` 결과는 FP32 `-0.0`이다. `SFPSETSGN`이 원래 음수 sign을 복사해도 값은 그대로 `-0.0`이므로, 보정 없이 저장하면 원래 `INT_MIN` 값 대신 BF16 `-0.0`이 남는다.

현재 구현은 절댓값의 bit 31을 `LReg[7]`에 추출한다. 정상적인 절댓값은 bit 31이 0이고, `INT_MIN`만 1이다. 이후 indirect `SFPMAD`가 `LReg[7]`을 index로 사용해 다음 두 상수 중 하나를 고른다.

```text
LReg[0] = 0.0
LReg[1] = -2**31

corrected_fp32 = LReg[LReg[7]] * 1.0 + signed_fp32
```

- 정상 입력: `LReg[7] = 0`이므로 `0.0`을 더한다.
- `INT_MIN`: `LReg[7] = 1`이므로 `-2**31`을 `-0.0`에 더한다.

별도의 lane branch 없이 `INT_MIN`만 고칠 수 있다.

### 실제 LLK와 `SFPLOADMACRO` schedule

기본 build에서 `DISABLE_SFPLOADMACRO`가 정의되지 않으면 [`calculate_typecast_int32_to_fp16b`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h#L124-L181)는 macro 경로를 사용한다.

반복 전에 다음 후속 instruction을 [`init_typecast_int32_to_fp16b`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h#L721-L755)에 등록한다.

| Sub-unit | 예약하는 동작 |
| --- | --- |
| Simple | cast 결과에 원래 `i32` sign 복사 |
| MAD | `LReg[7]`으로 `0.0` 또는 `-2**31`을 골라 보정 |
| Round | FP32 결과를 BF16 정밀도로 반올림 |
| Store | 결과를 `Dst`에 저장 |

Loop에서 software가 직접 issue하는 instruction은 다음 네 개다.

```cpp
TT_SFPLOADMACRO(...);                         // load v and schedule the tail
TT_SFPABS(0, v, t, 0);                        // t = abs_twos_complement(v)
TTI_SFPSHFT2(t, LREG12, LREG7, SHFT_LREG);   // LREG7 = t >> 31
TTI_SFPCAST(t, t, 0);                         // sign-magnitude i32 -> fp32
```

첫 `SFPLOADMACRO`가 load와 함께 sign 복사, `INT_MIN` 보정, BF16 rounding, store를 미래 cycle에 예약한다. 그 사이 software는 `SFPABS`, `SFPSHFT2`, `SFPCAST`를 issue한다. `v`는 후속 예약 동작이 끝날 때까지 살아 있어야 하므로 `LReg[2]`와 `LReg[3]`을 번갈아 쓴다.

```text
issue slot       0       1       2       3       0       1

group n        LOAD     ABS    FLAG    CAST    SIGN     FIX ... RND STORE
group n+1                                      LOAD     ABS ...

start interval: 4 issue slots per vector group
```

![i32에서 BF16으로 변환할 때 cycle별 SFPU sub-unit 동작과 LReg live range를 표시한 schedule](/assets/img/posts/tenstorrent-sfpu-and-typecast-examples/i32-to-bf16-schedule.png)

_그림. `i32 -> BF16` macro 경로의 schedule. [Typecast on Tenstorrent](https://www.jasondavies.com/2025/tenstorrent-typecast/#int32-bf16)의 schedule 표현을 기준으로 하며, 위에 고정한 Blackhole 구현의 instruction과 LReg 흐름을 나타낸다._

LLK 주석이 말하는 `4 cycles/input row`는 pipeline이 찬 뒤 다음 32-element vector group의 load를 시작하는 간격이다. 한 group의 load부터 store까지 걸리는 전체 latency가 4 cycle이라는 뜻은 아니다.

반복이 끝난 뒤에는 `TTI_SFPNOP` 다섯 개가 이어진다. 마지막 `SFPLOADMACRO`가 예약한 sign 복사, 보정, rounding과 store가 아직 남아 있기 때문이다. 이 `SFPNOP`은 load hardware의 오동작을 막는 dummy가 아니라, 더 이상 다음 group의 유효 instruction이 없는 tail에서 예약 schedule을 끝까지 진행시키는 issue 자리다.

`SFPLOADMACRO`를 끈 build에서는 같은 leaf의 plain loop가 load부터 store까지 모든 instruction을 직접 issue한다. 이 경로에는 미래에 예약한 동작이 없으므로 같은 tail `SFPNOP`도 없다.

### i32 값의 변환 예

`-13`은 다음 순서로 변환된다.

```text
input i32       : -13          (two's-complement 0xfffffff3)
SFPABS          : magnitude 13 (0x0000000d)
SFPCAST         : FP32 +13.0   (0x41500000)
SFPSETSGN       : FP32 -13.0   (0xc1500000)
SFPMAD fix      : add FP32 0.0
BF16 result     : -13.0        (exact)
```

`INT_MIN`은 다음처럼 보정된다.

```text
input i32       : -2**31       (two's-complement 0x80000000)
SFPABS          : magnitude bits 0x80000000
SFPCAST         : FP32 -0.0    (0x80000000)
SFPSETSGN       : FP32 -0.0
INT_MIN flag    : 1
SFPMAD fix      : FP32 -2**31  (0xcf000000)
BF16 result     : -2**31       (0xcf00, exact power of two)
```

`SFPCAST`는 `|x| <= 2**24`에서 정확하고, 그보다 큰 정수는 FP32로 바꾸는 과정에서 먼저 반올림한다. 이후 `SFPSTOCHRND`가 mantissa를 BF16의 7 bit로 줄이며 round-to-nearest, ties-away-from-zero를 적용한다. 따라서 이 경로는 단순한 bit truncation이 아니라 `i32 -> fp32 -> BF16 precision`의 두 단계 numeric conversion이다.

## 2. `fp32/BF16 -> i32`

### 두 input format이 같은 leaf를 쓰는 이유

[`typecast_tile`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h#L118-L158)는 `Float32 -> Int32`와 `Float16_b -> Int32`를 모두 `calculate_typecast_fp32_to_int32`로 보낸다.

BF16은 fp32와 sign/exponent 폭이 같고 mantissa만 짧다. 두 입력은 leaf에서 `SFPLOAD(DEFAULT)`로 읽힌 뒤 FP32 exponent와 mantissa 추출 instruction으로 처리된다. BF16 입력에서는 존재하지 않는 낮은 mantissa bit가 이미 0이므로 같은 정수 변환 알고리즘을 적용할 수 있다.

### 변환 규칙

이 경로는 소수부를 0 방향으로 버린다. `Exp`는 unbiased exponent이고, `Man`은 implicit leading bit를 포함한 24-bit mantissa integer다.

| 조건 | 중간 결과 |
| --- | --- |
| `Exp < 0` | `0` |
| `0 <= Exp < 31` | `Man << (Exp - 23)` |
| `Exp >= 31` | overflow sentinel `INT_MIN` |

음수는 magnitude를 만든 뒤 마지막에 two's-complement negation을 적용한다.

```text
result = 0
exp = extract_unbiased_exponent(value)

if exp >= 0:
    result = INT_MIN

    if exp < 31:
        mantissa = extract_mantissa_with_hidden_bit(value)
        result = shift(mantissa, exp - 23)

if value < 0:
    result = twos_complement_negate(result)
```

`Exp - 23`이 음수이면 right shift가 되어 fractional mantissa bit를 버린다. 먼저 절댓값의 정수부를 만든 다음 음수 lane만 negate하므로 결과는 floor가 아니라 0 방향 truncation이다.

### Predication으로 조건을 겹쳐 쓰기

실제 [`calculate_typecast_fp32_to_int32`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h#L183-L219)는 lane별 branch를 만들지 않는다. 대신 exponent를 계산하는 instruction이 다음 연산에 사용할 lane flag도 함께 설정한다.

1. `LREG1 = 0`으로 시작한다.
2. `SFPEXEXP`가 unbiased exponent를 `LREG2`에 쓰고 `Exp >= 0`인 lane만 활성화한다.
3. 활성 lane에 `INT_MIN`을 먼저 넣는다. 이 값이 overflow의 기본 결과다.
4. `SFPIADD(-31, CC_LT0)`가 `Exp < 31`인 lane만 다시 활성화한다.
5. 활성 lane에서 `SFPEXMAN`으로 implicit bit를 포함한 mantissa를 추출하고 `SFPSHFT`로 `Exp - 23`만큼 shift한다.
6. lane flag를 복원한 뒤 원래 입력이 음수인 lane만 결과를 negate한다.
7. `SFPSTORE(INT32)`로 `Dst`에 기록한다.

조건을 순서대로 좁히기 때문에 별도의 compare와 branch 없이 세 구간을 표현한다.

```text
all lanes
   |
   +-- Exp < 0 --------------------------> 0
   |
   +-- Exp >= 0
          |
          +-- Exp >= 31 -----------------> INT_MIN
          |
          +-- Exp < 31 ------------------> shifted mantissa
                                                |
                                                +-- negative: negate
```

`INT_MIN`의 two's-complement negation은 다시 `INT_MIN`이다. 따라서 음수 overflow lane에도 별도 예외 처리가 필요 없다. `-2**31`도 같은 bit pattern을 사용하므로 올바른 경계값과 overflow sentinel이 일치한다.

### 13개 instruction은 무엇을 뜻하는가

이 leaf의 loop 본체에는 load와 store를 포함해 SFPU instruction 13개가 명시돼 있다.

```text
load input
load zero
extract exponent and set flags
load INT_MIN sentinel
subtract 31 and narrow flags
add 8 to form Exp - 23
extract mantissa
shift mantissa
restore flags
select negative lanes
two's-complement negate
restore flags
store result
```

이 경로는 `SFPLOADMACRO`를 사용하지 않는다. 모든 연산을 software가 순서대로 issue하므로 미래에 남겨 둔 macro schedule도 없고, loop 끝을 `SFPNOP`으로 drain할 필요도 없다.

`13-instruction sequence`는 한 vector group의 명시적인 instruction 수다. `i32 -> BF16`의 `4 cycles/input row`처럼 pipeline이 찬 뒤 다음 group을 시작하는 간격과 같은 종류의 수치로 비교하면 안 된다.

### Floating-point 값의 변환 예

`3.75`의 unbiased exponent는 1이다. Implicit bit를 포함한 mantissa를 `1 - 23`만큼 right shift하면 정수부 3만 남는다.

```text
+3.75f -> magnitude 3 -> +3
-3.75f -> magnitude 3 -> -3
+0.75f -> Exp < 0     ->  0
```

BF16의 `-3.75`도 정확히 표현할 수 있으므로 같은 leaf에서 `-3`이 된다. `Exp >= 31`인 값은 계산 가능한 범위를 벗어난 것으로 보고 `INT_MIN`을 남긴다. Jason Davies의 글은 이 sentinel 선택을 PyTorch 동작에 맞추기 위한 것으로 설명한다.

## 두 경로 비교

| 구분 | `i32 -> BF16` | `fp32/BF16 -> i32` |
| --- | --- | --- |
| 핵심 문제 | two's-complement와 sign-magnitude 차이 | exponent 범위와 소수부 제거 |
| 주요 instruction | `SFPABS`, `SFPCAST`, `SFPSETSGN`, `SFPMAD`, `SFPSTOCHRND` | `SFPEXEXP`, `SFPEXMAN`, `SFPIADD`, `SFPSHFT` |
| 경계 처리 | `INT_MIN`을 `-2**31`로 보정 | overflow에 `INT_MIN` sentinel 사용 |
| 반올림 | FP32 intermediate 후 BF16 ties-away rounding | 0 방향 truncation |
| 기본 구현 | `SFPLOADMACRO` pipeline | plain 13-instruction loop |
| 끝의 `SFPNOP` | 마지막 예약 schedule drain | 없음 |

## Schedule과 throughput 해석

앞에서 본 `i32 -> BF16` macro 경로는 `4 cycles/input row`이고, 아래에서 예로 드는 `fp32 -> bf16` schedule은 `3 cycles/input row`다. 후자는 원문의 schedule 표기를 설명하기 위한 별도 예시이며, 두 수치 모두 같은 방법으로 읽는다.

### `cycles/input row`와 latency

SFPU input row 하나를 처리하는 데 여러 instruction이 필요할 수 있다. Pipeline이 채워진 뒤에는 앞 row의 모든 instruction이 끝나기 전에 다음 row의 load를 시작한다. 따라서 “row 하나를 끝내는 데 걸리는 latency”와 “연속된 row를 몇 cycle 간격으로 시작할 수 있는가”는 서로 다른 수치다.

원문의 `N cycles per input row`에서 `N`은 보통 **steady-state initiation interval**, 즉 pipeline이 찬 뒤 다음 32-element vector group을 시작하기까지의 간격이다. 한 group의 load부터 마지막 store까지 걸리는 latency가 `N` cycle이라는 뜻은 아니다.

| 구분 | 묻는 것 |
| --- | --- |
| latency | vector group 하나가 첫 load를 시작한 뒤 마지막 store를 마칠 때까지 얼마나 걸리는가? |
| throughput 또는 initiation interval | 연속된 두 vector group을 몇 cycle 간격으로 시작하고, steady state에서 몇 cycle마다 하나씩 끝낼 수 있는가? |

예를 들어 `fp32 -> bf16` schedule에서는 첫 group이 cycle 0에 시작해 마지막 store를 cycle 5에 실행하지만, 다음 group은 그 완료를 기다리지 않고 cycle 3에 시작한다.

```text
absolute cycle       0 1 2 3 4 5 6 7 8 ...
group 0              S-----work----F
group 1                    S-----work----F
group 2                          S-----work----F

S : first SFPLOADMACRO issue
F : final store

group start interval = 3 cycles
=> throughput = 3 cycles/input row
```

따라서 일반적인 full tile의 SFPU vector group 32개를 `3 cycles/input row`로 처리한다면 반복 본체의 steady-state 비용은 개념적으로 `32 × 3 = 96 cycles`다. 실제 tile 처리 시간에는 상수·macro 설정 같은 prologue, 마지막 예약 instruction을 끝내는 drain, loop 주변 instruction, hardware stall이 더해질 수 있다. 그러므로 “tile latency가 정확히 96 cycles”라고 단정할 수는 없다.

### 원문의 schedule 그림 읽는 법

다음 그림은 [Typecast on Tenstorrent](https://www.jasondavies.com/2025/tenstorrent-typecast/#fp32-bf16)의 schedule renderer가 출력한 `fp32 -> bf16` schedule이다. `a_n`·`v_n` 계산 경로뿐 아니라 각 instruction template과 LReg live range를 같은 cycle 행에 표시한다.

![fp32에서 BF16으로 변환할 때 cycle별 SFPU sub-unit 동작과 LReg live range를 함께 표시한 schedule](/assets/img/posts/tenstorrent-sfpu-and-typecast-examples/fp32-to-bf16-schedule.png)

이 그림의 시간축은 위에서 아래로 흐른다. 가로 한 줄이 한 cycle이며, 왼쪽 다섯 열은 그 cycle에 Load, Simple, MAD, Round, Store sub-unit이 실행하는 instruction을 보여 준다. 같은 줄에서 여러 칸이 채워져 있으면 여러 sub-unit이 같은 cycle에 병렬로 동작한다.

오른쪽의 `0..7`과 `16`은 cycle 번호가 아니라 LReg 번호다. 각 색 막대는 뒤의 instruction이 사용할 값을 해당 LReg에 보존해야 하는 기간, 즉 live range를 보여 준다. LReg를 읽어도 값은 사라지지 않으므로 마지막 사용 전에는 다른 값으로 덮어쓰면 안 된다. 마지막 사용이 끝난 뒤에는 기존 비트가 남아 있더라도 더 이상 보존할 필요가 없어 해당 LReg를 재사용할 수 있다. 따라서 막대의 길이는 instruction의 실행 시간이 아니다.

또한 `a0` 막대가 처음부터 끝까지 같은 비트가 유지된다는 뜻은 아니다. 같은 LReg의 값이 연산 결과로 갱신되더라도 `a0` 계산 경로가 계속 해당 LReg를 사용하면 막대가 이어진다. 즉, 막대는 바뀌지 않는 비트 패턴이 아니라 하나의 논리적인 값의 흐름을 추적한다.

| 표기 | 의미 |
| --- | --- |
| `a0`, `v0`, `a1`, `v1` | `a`와 `v`는 서로 다른 값의 흐름을, suffix는 input vector group 번호를 나타낸다. 따라서 `a0`와 `v0`는 같은 group 0을 처리한다. |
| 같은 색 | 같은 값의 흐름에 속한 instruction과 LReg live range를 연결한다. |
| instruction 칸 왼쪽 위의 작은 `0`, `1` | `SFPLOADMACRO`가 예약할 때 선택한 instruction template 번호다. 이 그림에서 `0`은 `SFPSHFT2`, `1`은 `SFPIADD` template이다. |

schedule의 길이와 input group의 시작 간격은 왼쪽 실행 영역의 cycle 행을 기준으로 판단한다. 오른쪽 영역은 각 cycle에 어떤 LReg가 어떤 값을 보관하는지 확인하는 용도다.

## 해석 범위와 검증 포인트

이 글의 schedule과 register 번호는 `tt-metal` commit `69096826694c`의 Blackhole 구현에 한정한다. Wormhole은 일부 load 방식과 address modifier가 다르고, Quasar는 통합 `calculate_typecast` 경로를 사용한다. `DISABLE_SFPLOADMACRO`를 정의한 build에서는 `i32 -> BF16`도 plain loop로 바뀐다.

구현이나 최적화를 검증할 때는 다음 값을 우선 확인하는 편이 좋다.

1. `i32 -> BF16`: `INT_MIN`, `INT_MAX`, `-1`, `0`, `1`, `2**24` 전후
2. `fp32/BF16 -> i32`: `-3.75`, `3.75`, 절댓값이 1보다 작은 값, `2**31` 경계
3. Macro 경로: 마지막 vector group의 store가 끝났는지 확인하는 tail case
4. 성능: `cycles/input row`, 한 group의 latency와 전체 tile latency를 구분

## 핵심 정리

1. SFPU는 `Dst`의 값을 32-lane LReg로 읽고, vector 연산 결과를 다시 `Dst`에 저장한다. SFPI는 이 과정을 C++ 형태로 표현하는 interface와 compiler다.
2. 일반적인 `SFPLOAD`는 16×16 face의 연속된 row 네 개에서 짝수 또는 홀수 column 여덟 개를 골라 `4×8=32`개 element를 LReg에 담는다.
3. `ITERATIONS=8`이 face 안의 group을 순회하고, `VectorMode::RC` wrapper가 네 face를 선택해 full tile의 1,024개 element를 처리한다.
4. `i32 -> BF16`은 two's-complement 절댓값을 만든 뒤 sign-magnitude `SFPCAST`를 사용하고, `INT_MIN`만 `SFPMAD`로 보정한다. Macro 경로의 `SFPNOP`은 마지막 예약 동작을 drain한다.
5. `fp32`와 BF16에서 `i32`로 가는 변환은 같은 leaf를 사용한다. Exponent로 lane을 좁히고 mantissa를 shift해 소수부를 0 방향으로 버리며, overflow에는 `INT_MIN`을 남긴다.

## 참고 자료

- [TT-Metalium compute engine과 SFPU dataflow](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html)
- [TT-Metalium FP32 accuracy와 FPU·SFPU 설정](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.html)
- [TT-Metalium tile 내부 구조](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/tiles.html)
- [TT-Metalium custom SFPU와 SFPI 소개](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/custom_sfpi.html)
- [TT-Metalium Low Level Kernels 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.html)
- [Blackhole `SFPLOAD` ISA와 functional model](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md)
- [Blackhole SFPU elementwise `VectorMode` wrapper](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h)
- [Blackhole `typecast_tile` dispatch](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h)
- [Blackhole SFPU typecast leaf 구현](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h)
- [Production typecast compute kernel](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp)
- [LLK SFPU typecast test kernel](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tests/sources/eltwise_unary_typecast_test.cpp)
- [Blackhole `SFPLOADMACRO` ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md)
- [Blackhole `SFPCAST` Int-to-Float ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPCAST_IntFloat.md)
- [Blackhole `SFPSTOCHRND` Float-to-Float ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md)
- [Blackhole `SFPEXEXP` ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPEXEXP.md)
- [Blackhole `SFPEXMAN` ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPEXMAN.md)
- [Blackhole `SFPABS` ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/SFPABS.md)
- [Typecast on Tenstorrent](https://www.jasondavies.com/2025/tenstorrent-typecast/)
