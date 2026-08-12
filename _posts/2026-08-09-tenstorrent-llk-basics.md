---
title: "Tenstorrent LLK 기초"
date: 2026-08-09 22:50:08 +0900
categories: [Tenstorrent, llk]
tags: [llk]
description: "Tenstorrent Blackhole LLK 코드를 읽는 데 필요한 Tensix thread, tile·register dataflow, FPU·SFPU와 instruction encoding의 공통 구조를 설명합니다."
render_with_liquid: false
---

## 문서 범위

- hardware 설명 기준: Blackhole의 일반적인 32×32 data tile 경로
- 기준 구현: `tt-metal` commit [`69096826694c`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e)
- 기준 ISA(Instruction Set Architecture): `tt-isa-documentation` commit [`b7738d9ac14a`](https://github.com/tenstorrent/tt-isa-documentation/tree/b7738d9ac14a34a4033d60dde9463466b23082e1)
- architecture 개요: [TT-Metalium compute engine 공식 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html)
- 확인일: 2026-08-13

이 글은 LLK(Low-Level Kernel) 코드를 읽을 때 Matrix engine인 FPU와 vector engine인 SFPU(Special Function Processing Unit)에 공통으로 필요한 hardware·programming 배경을 설명한다. Tensix instruction과 thread의 관계, tile과 register, dataflow, FPU·SFPU의 역할, 32-bit instruction encoding을 차례로 살펴본다.

`MVMUL`의 tile 분할·fidelity나 `SFPLOAD`의 lane mapping·pipeline schedule처럼 한 engine에만 해당하는 세부 구현은 범위에서 제외한다. 숫자와 bit 표현, 반올림, two's-complement가 먼저 필요하다면 [숫자와 데이터 표현 기초](/posts/number-and-data-representation-basics/)를 참고한다.

## Instruction과 thread

Tensix core 안에는 제어를 맡는 RISC-V core 다섯 개와 tile 연산을 실제로 처리하는 Tensix Engine이 있다. 다음 공식 그림에서 가운데 RISC-V 1~3은 Tensix Engine을 제어하고, 양쪽 RISC-V 0과 4는 router를 제어한다.

![Tensix chip의 core grid와 RISC-V core 다섯 개, router, Tensix Engine, internal memory의 관계](/assets/img/posts/tenstorrent-llk-basics/tensix-core.png)

_그림. Tensix core의 상위 구조. 출처: Tenstorrent의 [TT-Metalium Lab 1 공식 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html#kernel-types-and-data-flow)와 [commit `e7dfaaee4e1b`의 원본 PNG](https://github.com/tenstorrent/tutorial-assets/blob/e7dfaaee4e1b9188daa2da832ed1db644eb92f65/media/tt_metal/labs/lab1/tensix_core.png), © 2025 Tenstorrent AI ULC, [CC BY 4.0](https://github.com/tenstorrent/tutorial-assets/blob/e7dfaaee4e1b9188daa2da832ed1db644eb92f65/LICENSE). 원본을 변경하지 않았다._

[TT-Metalium compute engine 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#component-introduction)에 따르면 compute kernel 소스 하나는 unpack, math, pack용 binary 세 개로 컴파일된다. 각 binary는 서로 다른 RISC-V core에서 실행된다. 그림의 RISC-V 1~3은 ISA의 이름 체계로 RISC-V T0~T2에 해당하며, 일반적인 역할은 다음과 같다.

```text
TT-Metalium compute kernel
          |
          +--> unpack binary --runs on--> RISC-V T0 --pushes--> Tensix thread T0 --> Unpacker
          |
          +--> math binary   --runs on--> RISC-V T1 --pushes--> Tensix thread T1 --> FPU / SFPU
          |
          +--> pack binary   --runs on--> RISC-V T2 --pushes--> Tensix thread T2 --> Packer
```

RISC-V core와 Tensix thread는 같은 대상이 아니다. [Blackhole ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/BabyRISCV/PushTensixInstruction.md#L3-L7)에 따르면 RISC-V T0~T2는 32-bit Tensix instruction을 Tensix coprocessor 안의 각 T0~T2 thread에 넣는다. 각 coprocessor thread에는 독립된 instruction stream이 하나씩 있다.

이 글에서 별도로 `RISC-V instruction`이라고 한정하지 않은 `instruction`은 Tensix coprocessor가 실행하는 Tensix instruction을 뜻한다.

- **Instruction**은 `MVMUL`, `ELWADD`, `SFPLOAD`, `SFPCAST`처럼 coprocessor가 한 번 처리할 32-bit 명령이다.
- **Tensix thread**는 여러 instruction을 자기 stream의 순서대로 실행하는 hardware execution context다.
- **RWC**(Read/Write Counter)는 `SrcA`, `SrcB`, `Dst` 등의 현재 접근 위치를 추적하는 thread별 address state다. `RWCs[CurrentThread]`의 값은 instruction 하나가 끝난 뒤에도 남아 다음 instruction이 이어서 사용한다.

따라서 이 글의 `thread`는 OS thread가 아니라 Tensix coprocessor 내부의 T0~T2 thread를 가리킨다. Math binary를 실행하는 RISC-V T1은 FPU 또는 SFPU용 instruction을 Tensix thread T1에 보낸다. FPU와 SFPU는 독립적인 processing core가 아니며, 스스로 control flow를 결정하지 않는다.

## Tile과 register dataflow

[공식 tile 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/tiles.html#internal-structure-of-a-tile)가 설명하는 일반적인 data tile은 32×32 element로 이루어진다. 내부에서는 16×16 영역인 face 네 개로 나뉘며, matrix engine과 vector engine 모두 이 tile 구조를 사용한다. 다른 tile·face shape도 있지만 지원 범위가 제한적이므로 여기서는 표준 32×32 tile만 다룬다.

```text
32x32 data tile

             left half        right half
           +---------------+---------------+
top half   | Face 0        | Face 1        |
           | 16x16         | 16x16         |
           +---------------+---------------+
bottom     | Face 2        | Face 3        |
half       | 16x16         | 16x16         |
           +---------------+---------------+
```

FPU와 SFPU는 L1 SRAM을 직접 읽거나 쓰지 않는다. Unpacker가 L1 SRAM의 circular buffer(CB)에 있는 tile을 compute register에 놓고, FPU 또는 SFPU가 연산한 결과를 `Dst`에 남긴다. Packer는 `Dst` 값을 output format으로 변환해 L1으로 돌려보낸다.

```text
FPU path : L1 input --Unpacker--> SrcA / SrcB --> FPU -------> Dst --Packer--> L1 output
SFPU path: L1 input --Unpacker--> Dst ----------> SFPU/LRegs -> Dst --Packer--> L1 output
```

| 표기 | 종류 | 역할 |
| --- | --- | --- |
| `SrcA`, `SrcB` | FPU source register | Unpacker가 FPU operand를 배치한다. 이름과 달리 memory 주소가 아니라 physical register다. |
| `Dst` | shared destination register set | FPU의 결과를 보관하며, SFPU에는 source이자 destination이다. Compute API가 직접 노출하는 작업 공간이기도 하다. |
| `LReg` | SFPU local vector register | SFPU가 `Dst`에서 읽은 vector와 중간 결과를 보관한다. Blackhole에서는 일반적으로 32 lanes × 32 bits다. |
| `RWCs` | thread별 address counter 묶음 | `SrcA`, `SrcB`, `Dst` 등의 현재 접근 위치를 추적한다. |

![L1 SRAM, unpacker, packer, FPU, SFPU와 SrcA·SrcB·Dst·LReg 사이의 dataflow](/assets/img/posts/tenstorrent-llk-basics/compute-engines-and-register-dataflow.webp)

_그림. Tensix compute engine과 register dataflow. `RWCs` 같은 address state는 포함하지 않는다. 출처: Tenstorrent의 [공식 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#component-introduction)와 [commit `69096826694c`의 원본 WebP](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/docs/source/common/images/tenstorrent-sfpu-fpu-dst-register-diagram-and-dataflow.webp), © Tenstorrent, [Apache License 2.0](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/LICENSE). 원본을 변경하지 않았다._

Unpacker와 packer는 L1의 storage format과 compute register format 사이의 변환도 맡는다. 따라서 L1에 BF16이나 block floating-point 형식으로 저장한 tile이 compute register에서 같은 bit layout을 유지한다고 가정하면 안 된다.

### `Dst`의 소유권과 동기화

Unpacker, math, pack binary는 서로 다른 RISC-V core에서 동시에 실행될 수 있으므로 `Dst` 접근 순서를 맞춰야 한다. [공식 `Dst` 설명](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#dst-register)의 일반적인 compute loop는 다음 ownership 전환을 사용한다.

```text
tile_regs_acquire
        |
        v
unpack and compute on Dst
        |
        v
tile_regs_commit
        |
        v
tile_regs_wait
        |
        v
pack Dst to output CB
        |
        v
tile_regs_release
```

`tile_regs_acquire()`는 unpacker와 math 쪽이 사용할 `Dst`를 확보한다. 연산 뒤 `tile_regs_commit()`이 packer로 소유권을 넘기고, `tile_regs_wait()`가 packer의 접근 가능 시점을 기다린다. `tile_regs_release()`까지 끝나야 다음 반복에서 해당 공간을 다시 쓸 수 있다.

## FPU와 SFPU

FPU와 SFPU는 모두 math RISC-V가 내린 명령을 실행하고 `Dst`를 중심으로 결과를 주고받는다. 차이는 어떤 register를 operand로 사용하고 어떤 형태의 연산에 특화됐는지에 있다.

| 구분 | Matrix engine(FPU) | Vector engine(SFPU) |
| --- | --- | --- |
| 주된 역할 | matrix·block 연산과 일부 elementwise·pooling 연산 | 복잡한 elementwise 함수와 32-lane vector 연산 |
| 주된 입력 | `SrcA`, `SrcB` | `Dst`에서 `LReg`로 읽은 vector |
| 결과 | `Dst`에 기록하거나 누적 | `LReg`에서 연산한 뒤 `Dst`에 저장 |
| 대표 compute API | `matmul_tiles`, `add_tiles` | `exp_tile`, `relu_tile`, `typecast_tile` |
| 대표 backend instruction | `MVMUL`, `ELWADD` | `SFPLOAD`, `SFPCAST`, `SFPIADD`, `SFPSTORE` |
| 정밀도 관련 설정 | `math_fidelity`가 partial-product phase 수를 정한다. | `math_fidelity`는 적용되지 않으며, 일부 함수는 `math_approx_mode`의 영향을 받는다. |

[공식 FP32 accuracy 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.html#host-side-configuration)에 따르면 `math_fidelity`는 matrix engine에만 적용된다. Vector engine은 이 설정에 따라 같은 연산을 여러 fidelity phase로 나누지 않는다.

### Matrix engine(FPU) 경로

[공식 FPU 설명](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#matrix-engine-fpu)에 따르면 FPU는 대부분의 AI workload에서 큰 비중을 차지하는 matrix 연산을 처리한다. Unpacker가 input tile을 `SrcA`와 `SrcB`에 배치하면 `MVMUL` 같은 instruction이 block 단위로 곱하고 `Dst`에 누적한다. FPU는 matrix multiplication 외에도 elementwise 곱셈·덧셈·뺄셈과 pooling을 지원한다.

```text
input CB A/B --Unpacker--> SrcA / SrcB --FPU instructions--> Dst
```

연산 전에 init 함수가 input/output format과 operation mode에 맞춰 unpacker, FPU, packer를 설정한다. 같은 source·destination·format 조합으로 반복할 때는 매 tile마다 다시 초기화하지 않는다.

### Vector engine(SFPU) 경로

[공식 SFPU 설명](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#vector-engine-sfpu)에 따르면 SFPU는 `sin`, `exp`, `relu`, typecast 같은 elementwise 연산에 적합한 vector engine이다. SFPU 경로에서는 먼저 input tile을 `Dst`에 놓는다. SFPU가 일부 element를 `LReg`로 읽어 연산한 뒤 결과를 다시 `Dst`에 저장한다.

```text
input CB --Unpacker--> Dst --load--> LReg --SFPU instructions--> LReg --store--> Dst
```

SFPI(SFPU Interface)는 SFPU program을 C++ 형태로 작성하기 위한 library와 compiler다. [공식 SFPI 소개](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/custom_sfpi.html)에 따르면 Blackhole의 SFPU는 32-wide vector에서 FP32·INT32 연산과 lane predication을 지원한다. `VD`는 vector destination의 약자로, 보통 SFPU instruction이 결과를 기록할 LReg index를 고른다. 다만 operand의 정확한 읽기·쓰기 방향은 instruction마다 ISA를 확인해야 한다.

FPU와 SFPU의 구분은 “tile 연산 대 scalar 연산”이 아니다. 두 engine 모두 tile data를 처리하지만 내부 primitive와 register 경로가 다르다. SFPU도 instruction 한 번에 여러 lane을 처리하며, FPU도 matrix multiplication 외의 연산을 실행할 수 있다.

## 32-bit Tensix instruction encoding

LLK의 macro와 SFPI compiler가 만드는 backend 명령은 계산할 data가 아니라 **Tensix instruction word**다. Math RISC-V는 이 word를 Tensix thread에 넣고, opcode가 가리키는 FPU 또는 SFPU가 실행한다.

```text
Compute API
    |
    +--> FPU LLK --------> MVMUL + FPU fields ----------+
    |                                                   |
    +--> SFPU LLK/SFPI -> SFPLOAD + SFPU fields --------+
                                                        |
                                                        v
                                             32-bit instruction word
                                                        |
                                                        v
                                             Math RISC-V pushes to T1
                                                        |
                                      +-----------------+-----------------+
                                      |                                   |
                                      v                                   v
                               Matrix engine FPU                   Vector engine SFPU
```

기준 Blackhole LLK source의 [`TT_OP`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h#L11-L12)는 instruction word를 다음처럼 조합한다.

```cpp
TT_OP(opcode, params) = (opcode << 24) + params
```

`INSTRUCTION_WORD` macro는 이 값을 inline assembly에 넘긴다. 아래 bit 배치와 16진수는 **logical Tensix instruction word**를 나타낸다.

```text
bit      31                    24 23                         0
        +------------------------+----------------------------+
field   | opcode: 8 bits         | opcode-specific fields     |
        +------------------------+----------------------------+
```

상위 8-bit `opcode`가 명령을 구분하고, 하위 24 bit는 register·mode·immediate처럼 opcode별 operand를 담는다. FPU와 SFPU instruction 모두 이 큰 형식을 사용하지만 하위 field의 이름과 의미는 서로 다르다.

| Engine | 예시 | Opcode | 하위 field가 정하는 것 |
| --- | --- | ---: | --- |
| FPU | [`MVMUL`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h#L366-L371) | `0x26` | `Dst` field, address modifier, instruction mode, data-valid control 등 |
| SFPU | [`SFPLOAD`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h#L670-L676) | `0x70` | LReg index, data format, address modifier, immediate offset 등 |

같은 32-bit 형식을 쓴다는 사실이 두 instruction의 실행 방식까지 같다는 뜻은 아니다. Opcode가 선택한 engine의 ISA가 하위 field를 해석하며, 실행 중에는 `RWCs[CurrentThread]` 같은 thread별 state도 함께 참조할 수 있다.

## LLK 코드를 읽는 순서

FPU와 SFPU 중 어느 경로를 보더라도 다음 순서로 읽으면 역할이 섞이지 않는다.

1. Public compute API가 어떤 init 함수와 operation을 호출하는지 확인한다.
2. Unpacker가 input을 `SrcA`·`SrcB`로 보내는지, `Dst`로 바로 보내는지 확인한다.
3. LLK 또는 SFPI가 생성하는 backend instruction과 실행 engine을 확인한다.
4. 결과가 `Dst`의 어느 tile slot에 남는지 확인한다.
5. `tile_regs_*`와 packer가 `Dst`의 소유권과 output format을 어떻게 처리하는지 확인한다.

```text
API call
   |
   v
init and data format setup
   |
   v
Unpacker destination: SrcA/SrcB or Dst
   |
   v
backend instruction: FPU or SFPU
   |
   v
result in Dst
   |
   v
Packer to output CB
```

이 구분을 먼저 세우면 `Dst`가 FPU에는 주로 destination인데 SFPU에는 source이자 destination인 이유와, SFPI가 SFPU에만 등장하는 이유를 자연스럽게 이해할 수 있다.

## 제약과 가정

- 이 글의 tile 설명은 일반적인 32×32 tile과 16×16 face 네 개를 기준으로 한다.
- `LReg` 폭, instruction field, RWC 동작은 architecture에 따라 달라질 수 있다. 세부 분석은 Blackhole A0와 위에 고정한 source revision을 기준으로 해야 한다.
- `fp32_dest_acc_en=true`는 `Dst`의 element storage 폭을 32 bit로 설정하지만 모든 FPU 연산이 FP32 정밀도로 수행된다는 뜻은 아니다.
- Compute API는 여러 hardware 차이를 추상화한다. 성능이나 정확성을 instruction 수준에서 설명하려면 init 설정, data format과 실제 backend instruction을 함께 확인해야 한다.

## 핵심 정리

1. RISC-V T0~T2가 instruction stream과 dataflow를 제어하고, unpacker·FPU·SFPU·packer가 전달받은 명령을 실행한다.
2. Unpacker는 L1 tile을 compute register로 옮기고, packer는 `Dst` 결과를 L1으로 돌려보낸다.
3. FPU는 주로 `SrcA`·`SrcB`에서 읽어 `Dst`에 기록하고, SFPU는 `Dst`와 내부 `LReg` 사이에서 vector를 옮겨 연산한다.
4. FPU와 SFPU의 backend instruction은 모두 32-bit word지만 opcode별 operand와 실행 의미는 서로 다르다.
5. LLK 코드는 init, unpack destination, 실행 engine, `Dst` 결과, pack 순서로 추적하면 읽기 쉽다.

## 참고 자료

- [TT-Metalium compute engine과 register dataflow](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html)
- [TT-Metalium tile 구조](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/tiles.html)
- [TT-Metalium custom SFPU와 SFPI 소개](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/custom_sfpi.html)
- [TT-Metalium Low Level Kernels 문서](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.html)
- [TT-Metalium Lab 1 Tensix core 구조](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html#kernel-types-and-data-flow)
- [Blackhole Tensix instruction push와 thread model](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/BabyRISCV/PushTensixInstruction.md)
- [Blackhole Matrix Unit(FPU) ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/MatrixUnit.md)
- [Blackhole Vector Unit(SFPU) ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md)
- [Blackhole LLK instruction encoding source](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h#L11-L12)
