---
title: "Tenstorrent NoC 기초"
date: 2026-08-18 09:44:51 +0900
categories: [Tenstorrent, noc]
tags: [noc]
description: "NoC의 NIU·router와 mesh·torus부터 Tenstorrent의 NOC0·NOC1, Metalium 2D matmul의 multicast까지 핵심 구조를 설명합니다."
render_with_liquid: false
---

## 문서 범위

- NoC 기본 개념: local resource, NIU, router, link, packet과 hop
- Topology: 2D mesh와 2D torus, Tenstorrent의 `NOC0`·`NOC1`
- Programming model: TTNN 2D matmul의 `in0` multicast와 동기화
- 기준 구현: `tt-metal` commit [`69096826694c`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e)
- 기준 NoC ISA: `tt-isa-documentation` commit [`b7738d9ac14a`](https://github.com/tenstorrent/tt-isa-documentation/tree/b7738d9ac14a34a4033d60dde9463466b23082e1)
- 공식 문서 확인일: 2026-08-18

이 글은 TT-Metalium kernel에서 데이터가 어디로, 어떤 경로를 통해 이동하는지 이해하는 데 필요한 개념에 집중한다.

여기서 **NoC tile**은 router와 NIU가 놓인 칩 격자의 한 위치다.

## NoC는 칩 안의 packet network다

NoC(Network on Chip)는 한 칩 안의 계산 장치, 메모리와 I/O를 잇는 packet-switched network다. 연결 대상이 많아지면 하나의 공용 bus가 모든 통신을 중재하기 어렵다. NoC는 여러 router와 link로 통신을 분산해 서로 다른 경로의 전송을 동시에 진행할 수 있게 한다.

NoC 자체는 데이터를 저장하거나 계산하지 않는다. 출발지의 요청을 packet으로 만들어 목적지까지 전달한다.

```text
Logical transaction directions

+-------------+    +------------+    +------------+    +------------+    +-------------+
| source      |--->| source NIU |--->| NoC fabric |--->| target NIU |--->| destination |
| resource    |<---|            |<---| routers    |<---|            |<---| resource    |
+-------------+    +------------+    +------------+    +------------+    +-------------+

                         request packet(s) --->
                         <--- response / acknowledgement packet(s)
```

| 구성 요소 | 역할 |
| --- | --- |
| Local resource | Tensix L1, DRAM, PCIe subsystem처럼 실제 read·write를 처리한다. |
| NIU | 각 tile의 local NoC endpoint다. Tile의 transaction·data와 NoC packet 사이를 연결한다. |
| Router | NoC grid의 각 위치에서 packet을 다음 link로 전달한다. |
| Link | NIU와 router 또는 인접 router를 잇는 물리 전송로 한 구간이다. |
| Hop | Packet이 link 하나를 지나는 단계다. |
| Transaction | Read, write, atomic처럼 소프트웨어가 요청한 작업이다. |
| Packet·flit | Packet은 함께 routing되는 단위이고, flit은 link를 통해 전송되는 더 작은 단위다. |

Read는 목적지로 가는 request와 데이터를 싣고 돌아오는 response가 필요하다. Write는 local 데이터를 목적지로 보내며, 설정에 따라 완료 acknowledgement를 받을 수 있다. 큰 transaction 하나가 여러 packet으로 나뉠 수도 있다.

Router는 packet을 목적지 방향의 다음 link로 전달한다. 여러 packet이 같은 link에 몰리면 차례를 기다리므로 지연이 생길 수 있다.

## 2D mesh와 2D torus

2D mesh는 router를 행과 열로 배치하고 상하좌우 이웃을 연결한다. 바깥쪽 router에는 격자 밖으로 나가는 link가 없다.

```text
2D mesh

R00-----R10-----R20
 |       |       |
R01-----R11-----R21
 |       |       |
R02-----R12-----R22

No wrap links at the edges
```

2D torus는 같은 격자의 좌우 끝과 상하 끝을 각각 이어 각 행과 열을 ring으로 닫는다.

```text
2D torus, shown as two wrapped cross-sections

       +-----------------+
       |                 |
       v                 |
row:  R00----->R10----->R20

       +-----------------+
       |                 |
       v                 |
col:  R00----->R01----->R02

Every row and column closes into a ring
```

Torus에서는 반대편 edge로 wrap할 수 있어 일부 출발지·목적지 사이의 hop 수가 줄어든다. 대신 wrap link와 순환 경로를 안전하게 제어할 routing 규칙이 필요하다.

## Tenstorrent의 NOC0과 NOC1

Blackhole에는 같은 tile 집합을 연결하는 `NOC0`와 `NOC1`이 있다. 여기서 NoC는 단일 hardware module이나 bus line의 이름이 아니다. 공식 문서는 [한 NoC를 NIU와 router가 link로 연결된 2D torus 전체](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/NoC/README.md#L5-L13)로 정의한다. 따라서 `NOC0`과 `NOC1`은 각각 칩 전체에 펼쳐진 하나의 물리 network이며, 서로 완전히 분리되어 있다.

Blackhole의 기본 진행 방향과 unicast 경로는 다음과 같다.

| NoC | 진행 방향 | Unicast route |
| --- | --- | --- |
| `NOC0` | 오른쪽과 아래쪽 | 오른쪽으로 이동한 뒤 아래쪽으로 이동한다. |
| `NOC1` | 왼쪽과 위쪽 | 위쪽으로 이동한 뒤 왼쪽으로 이동한다. |

각 방향의 끝에서는 torus link를 통해 반대편으로 wrap한다. 출발지와 목적지, 선택한 NoC가 정해지면 [기본 route](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/NoC/RoutingPaths.md)도 정해진다. Router가 매 hop마다 칩 전체의 congestion을 보고 임의의 최단 경로를 다시 찾는 구조는 아니다.

### Tensix tile의 RISC-V와 NoC endpoint

Blackhole의 Tensix tile에는 [RISC-V core 다섯 개](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/docs/source/tt-metalium/tt_metal/labs/matmul/lab1/lab1.rst#L439-L445)가 있다.

| RISC-V core | 역할 | 흔한 예제 배치 |
| --- | --- | --- |
| BRISC, `DataMovementProcessor::RISCV_0` | Data-movement kernel 실행 | Reader, `NOC0` |
| NCRISC, `DataMovementProcessor::RISCV_1` | Data-movement kernel 실행 | Writer, `NOC1` |
| `TRISC0`·`TRISC1`·`TRISC2` | Compute kernel의 unpack·math·pack 제어 | 해당 없음 |

다음 그림은 2D torus의 한 행만 단순화한 것이다. A와 B는 각각 Metalium에서 하나의 Tensix core로 다루는 tile이며, 그림은 논리적 연결 관계를 나타낸다.

```text
NOC0 fabric, one row = NIU0s + R0 routers + NOC0 links

...  ---> [R0@A] ---- link ---->  [R0@B] ---> ...
            |                       |
          [NIU0]                  [NIU0]
            | tile interface        | tile interface
    +-------+--------+      +-------+--------+
    | Tensix tile A  |      | Tensix tile B  |
    | BRISC / NCRISC |      | BRISC / NCRISC |
    | local L1 / CBs |      | local L1 / CBs |
    +-------+--------+      +-------+--------+
            | tile interface        | tile interface
          [NIU1]                  [NIU1]
            |                       |
... <--- [R1@A] <--- link -----   [R1@B] <--- ...

NOC1 fabric, one row = NIU1s + R1 routers + NOC1 links
```

| 용어 | 물리적으로 가리키는 범위 |
| --- | --- |
| `NOC0`, `NOC1` | 같은 번호의 NIU·router·link를 합친 칩 전체 network다. |
| `NIU0`, `NIU1` | [Tensix tile에 붙은](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/TensixTile/README.md#L3-L7) local NoC endpoint다. |
| `R0@A`, `R1@A` | Grid 위치 A의 개별 `NOC0`·`NOC1` router다. |
| Link | NIU와 router 또는 두 router 사이의 물리 선로 한 구간이다. |

NIU는 tile의 local address space와 router를 잇는 endpoint block이며, 그림에서는 구분을 위해 tile box 밖에 두었다. BRISC와 NCRISC가 NIU에 read·write 명령을 넣으면, NIU가 L1과 packet 사이에서 payload를 옮기고 router와 link가 packet을 운반한다. 여러 tile이 같은 NoC에 동시에 packet을 보낼 수 있으며, 같은 output link를 공유하면 contention이 생긴다. DRAM·PCIe·Ethernet tile도 각자의 NIU를 통해 이 network에 연결된다.

일반적인 matmul 예제는 BRISC reader와 `NOC0`로 입력을 읽고, NCRISC writer와 `NOC1`로 출력을 쓴다. Circular buffer(CB)는 local L1에 있다.

```text
Common matmul mapping

BRISC reader --NoC command--> local NIU0 --read request via NOC0--> DRAM A/B
input CB <==== local NIU0 <==== read response via NOC0 <============= DRAM A/B

input CB ----> matmul compute ----> output CB

NCRISC writer --NoC command--> local NIU1
output CB ====> local NIU1 ==== write request via NOC1 ====> DRAM C
```

`via NOC0`은 단일 module이 아니라 `NOC0`에 속한 NIU·router·link 경로를 뜻한다. 위 reader·writer 조합은 [Metalium 예제의 기본 배치](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab3/lab3.html#data-movement-processors-and-noc-selection)일 뿐 하드웨어 규칙은 아니다. [`DataMovementConfig`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/api/tt-metalium/kernel_types.hpp#L22-L67)에서 processor와 NoC를 따로 선택할 수 있다.

## Metalium의 NoC 데이터 이동: 2D matmul의 `in0` multicast 예제

TTNN의 2D matmul은 출력 행렬을 2D worker grid에 나눈다. 같은 행의 core는 같은 `in0` block을 사용하므로, 한 core가 block을 읽어 같은 행의 나머지 core에 multicast한다. 아래 설명은 [`matmul_multicore_reuse_mcast_2d_program_factory.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp#L260-L338)의 일반적인 비전치 배치를 기준으로 한다.

```text
2D worker grid, in0 path

row 0: in0 --> [S + C] --+--> [R + C, col 1]
                         +--> [R + C, col 2]

row 1: in0 --> [S + C] --+--> [R + C, col 1]
                         +--> [R + C, col 2]

S: in0 sender on the left column
R: in0 receiver
C: matmul compute on every work core
```

Factory는 왼쪽 열에 [`reader_bmm_tile_layout_in0_sender_padding.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp)를, 나머지 core에 [`reader_bmm_tile_layout_in0_receiver.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp)를 배치한다. Sender도 자신의 `in0` block으로 matmul을 수행하므로 자기 L1로 다시 복사하지 않고 receiver만 multicast 대상으로 삼는다.

두 `in0` data-movement kernel은 [`RISCV_1`에서 실행되도록 설정](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp#L763-L822)된다. 즉 reader라는 이름이지만 NCRISC에서 실행된다. 같은 factory는 `in1` reader와 output writer를 겸하는 kernel을 `RISCV_0`에 둔다. 사용할 NoC도 kernel별로 지정하므로 reader와 `NOC0`, writer와 `NOC1`의 조합은 고정 규칙이 아니다.

### 참고 : Circular buffer의 producer·consumer 동기화

Circular buffer(CB)는 host program이 정한 크기의 local L1 영역을 ring으로 재사용한다. Producer와 consumer는 다음 API로 빈 공간과 읽을 수 있는 tile을 구분한다.

```text
Producer                         Local CB state                     Consumer
--------                         --------------                     --------
cb_reserve_back(cb, n) --wait--> free slots >= n
write(get_write_ptr(cb)) ------> reserved back slots
cb_push_back(cb, n) ---------->  n tiles become visible -------->   cb_wait_front(cb, n) returns
                                                                    read(get_read_ptr(cb))
                                 n slots become free <------------- cb_pop_front(cb, n)

                         write and read pointers wrap
```

[`cb_reserve_back()`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/cb_reserve_back.html)과 [`cb_wait_front()`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/cb_wait_front.html)는 조건을 만족할 때까지 기다릴 뿐 tile data를 옮기지 않는다. 실제 쓰기는 `noc_async_read`나 `pack_tile`, 읽기는 `matmul_block`이나 `copy_tile` 같은 별도 연산이 수행한다. [`cb_push_back()`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/cb_push_back.html)은 쓴 tile을 consumer에 공개하고 write pointer를 옮긴다. [`cb_pop_front()`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/cb_pop_front.html)은 L1의 bytes를 지우는 대신 읽은 slot을 다시 쓸 수 있게 하고 read pointer를 옮긴다. 이 순서가 producer의 overflow와 consumer의 underflow를 막는다.

이 동기화는 한 core의 local CB를 공유하는 producer와 consumer 사이에서 동작한다. 다른 core의 CB로 multicast할 때는 아래 예제처럼 각 receiver의 공간이 준비됐는지 NoC semaphore로 따로 확인해야 한다.

### 한 `in0` block이 이동하는 순서

Factory는 [모든 대상 core에 `in0`용 semaphore 두 개](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp#L1120-L1126)를 만든다. Semaphore는 각 core의 L1에 놓인 작은 값이며, local code뿐 아니라 다른 core도 NoC를 통해 갱신할 수 있다. 같은 ID를 사용해도 하나의 전역 변수를 공유하는 것은 아니다.

| 동기화 요소 | 역할 |
| --- | --- |
| `sender_sem` | Sender core의 준비 counter다. 각 receiver가 remote atomic으로 1씩 올리고, sender는 값이 receiver 수에 도달할 때까지 기다린다. |
| `receiver_sem` | 각 receiver core의 도착 flag다. Receiver가 `INVALID`로 초기화하고, sender가 data 뒤에 `VALID`를 multicast한다. |
| [`async_read_barrier()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/inc/api/dataflow/noc.h#L599-L615) | Sender core가 선택한 NoC로 발행한 `in0` read의 완료를 기다린다. Receiver의 준비 상태까지 확인하지는 않는다. |

따라서 sender는 read barrier와 `sender_sem` 조건을 모두 통과해야 한다. 전자는 **데이터를 읽어 왔는지**, 후자는 **모든 receiver에 쓸 공간이 있는지**를 확인한다.

`cb_in0`은 각 core의 local CB다. 같은 CB ID를 쓰더라도 sender와 각 receiver에는 별도의 instance가 있다. Sender도 matmul에 참여하므로 자신이 읽은 block을 자기 compute kernel에 넘기고, receiver는 multicast로 받은 block을 자기 compute kernel에 넘긴다.

```text
Each receiver core                    Sender core
------------------                    -----------
own L1: local cb_in0                  own L1: local cb_in0
reserve_back(local cb_in0)            reserve_back(local cb_in0)
receiver_sem = INVALID                async_read(in0 -> local cb_in0)
                                      async_read_barrier()
atomic up(sender_sem) --------------+
                                    +-> wait(sender_sem == num_dests)
reserved cb_in0 slot <================= multicast sender-local block
wait(receiver_sem == VALID) <---------- multicast VALID
push_back(local cb_in0)               push_back(local cb_in0)
          |                                     |
          v                                     v
receiver compute                      sender compute
```

1. Receiver의 `reserve_back()`은 local `cb_in0`에 빈 slot이 생길 때까지 기다린다. 공간을 확보하면 `receiver_sem`을 `INVALID`로 바꾸고 sender core의 `sender_sem`을 remote atomic으로 올린다.
2. Sender는 자신의 `cb_in0`에도 공간을 확보하고 `in0` tile read를 비동기로 발행한다. `async_read_barrier()`는 이 sender가 발행한 read가 L1에 도착할 때까지 기다린다. 그 뒤 `sender_sem == num_dests`를 확인하고 counter를 다음 block을 위해 0으로 되돌린다.
3. Data와 모든 receiver의 준비가 확인되면 sender는 자기 local `cb_in0`의 block을 각 receiver의 `cb_in0`용으로 예약된 L1 slot에 multicast한다. 실제로 옮기는 것은 CB 객체나 pointer가 아니라 block을 구성하는 tile data다. 이어서 각 receiver의 `receiver_sem`에 `VALID`를 multicast한다.
4. Receiver가 `VALID`를 확인한 시점에는 tile data가 이미 예약된 L1 slot에 있지만 compute kernel에는 아직 보이지 않는다. Sender와 각 receiver가 자기 local `cb_in0`을 `push_back()`하면 해당 slot이 각 core의 compute kernel에 공개된다. 같은 CB를 두 번 push하는 것이 아니다.

핵심은 양방향 handshake다. Receiver에서 sender로 가는 `sender_sem` atomic은 **공간이 준비됐다**는 뜻이고, sender에서 receiver로 오는 `receiver_sem = VALID`는 **데이터가 도착했다**는 뜻이다. Semaphore는 상태만 전달하며 실제 `in0` block은 NoC multicast가 옮긴다.

## 핵심 정리

1. NoC는 local resource의 transaction을 NIU가 packet으로 만들고 router가 여러 hop에 걸쳐 전달하는 on-chip network다.
2. Mesh는 edge에서 끝나고, torus는 각 행과 열의 끝을 이어 ring으로 닫는다.
3. Tenstorrent의 `NOC0`와 `NOC1`은 같은 tile을 반대 방향으로 연결하는 독립된 두 physical 2D torus이며, BRISC·NCRISC와는 별도 hardware다.
4. CB의 `reserve`·`wait`는 공간이나 data를 기다리고, `push`·`pop`은 쓴 tile을 공개하거나 읽은 slot을 반환한다.
5. 2D matmul의 `in0` sender는 block을 한 번 읽고 같은 행의 receiver에 multicast한다.
6. Receiver는 CB 공간을 먼저 확보하고 준비 신호를 보낸다. Sender가 data와 완료 신호를 보낸 뒤 각 core가 CB를 compute에 공개한다.

## 참고 자료

- [TT-Metalium Lab 3: NoC와 multicast](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab3/lab3.html)
- [TT-Metalium Lab 1: Tensix core와 RISC-V processor](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html)
- [TT-Metalium kernel의 memory와 NoC access](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html)
- [TT-Metalium NoC ordering](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/data_movement/ordering.html)
- [TTNN 2D matmul program factory](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp)
- [2D matmul `in0` sender kernel](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp)
- [2D matmul `in0` receiver kernel](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp)
- [Blackhole A0 NoC ISA 개요](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/NoC/README.md)
- [Blackhole A0 routing path](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/NoC/RoutingPaths.md)
- [Blackhole A0 PCIe tile의 AXI·NoC bridge](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/BlackholeA0/PCIExpressTile/README.md)
- [Benini와 De Micheli, Networks on Chips: A New SoC Paradigm](https://doi.org/10.1109/2.976921)
