---
title: "TT-Metalium Runtime 구조: TTNN 호출에서 Tensix 커널 실행까지"
date: 2026-08-21 16:50:53 +0900
categories: [Tenstorrent, runtime]
tags: [tt-metal, tt-metalium, runtime, fast-dispatch]
description: "공식 tt-metal 소스를 따라 TTNN operation이 Program과 MeshWorkload를 거쳐 Fast Dispatch firmware와 Tensix worker kernel로 실행되는 경로를 정리한다."
render_with_liquid: false
---

`tt-metal`에서 흔히 runtime이라고 부르는 부분을 찾으려면 어디를 봐야 할까?
결론부터 말하면 하나의 `runtime/` 디렉터리를 찾으면 안 된다. TTNN의 operation
실행부, TT-Metalium의 host runtime, device에 상주하는 dispatch firmware, Tensix의
worker firmware, 그리고 하위의 LLRT와 UMD가 함께 runtime 경로를 이룬다.

이 글은 이 경로를 위에서 아래로 연결한다. 특히 일반적인 TTNN operation 하나가
호출된 뒤 어떤 host 객체를 거치고, 어떤 command가 device로 전달되며, 마지막에
reader·compute·writer kernel이 어떻게 시작되는지를 공식 소스 기준으로 추적한다.

## 문서 범위

- 기준 소스는 Tenstorrent 공식 `tt-metal` 저장소의
  [`6909682`](https://github.com/tenstorrent/tt-metal/commit/69096826694cac0e8bbde0050e38a3e411a6d91e)
  커밋이다. 커밋 시각은 2026-08-12이며, 문서 확인일은 2026-08-21이다.
- 로컬 파생 저장소는 경로를 대조하는 용도로만 사용했다. 동작 설명과 링크는 모두
  공식 upstream 자료를 근거로 한다.
- 실행 경로는 한 host에 직접 연결된 device를 `1x1 MeshDevice`로 열고, 기본값인
  command queue 하나에서 일반 workload를 Fast Dispatch로 실행하는 경우를 중심으로
  설명한다.
- worker 쪽 세부 설명은
  [`tt-1xx` firmware](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/firmware/src/tt-1xx)
  기준이다. `tt-2xx`, multi-host Fabric, service core workload, 여러 sub-device,
  trace replay, profiler, DRAM-backed command queue는 범위에서 제외한다.
- 이 글에서 runtime은 공식 제품명 하나가 아니라, operation 제출부터 완료 확인까지의
  실행 계층을 묶어 부르는 용어다. TT-Metalium 자체의 입문 흐름은 공식
  [`METALIUM_GUIDE.md`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/METALIUM_GUIDE.md#L280-L542)를
  먼저 읽으면 좋다.

## 먼저 보는 결론

runtime을 읽을 때 가장 중요한 구분은 다음 네 가지다.

1. TTNN의 `device_operation`은 operation을 검증하고 output tensor를 만들며,
   program cache를 조회한 뒤 `MeshWorkload`를 제출한다.
2. TT-Metalium host runtime은 kernel, circular buffer, semaphore, runtime argument를
   담은 `Program`을 compile하고 dispatch command sequence로 바꾼다.
3. Fast Dispatch firmware인 `cq_prefetch`와 `cq_dispatch`는 host queue의 command를
   가져와 worker L1에 설정값과 launch message를 쓰고 GO signal을 보낸다.
4. Tensix worker firmware는 launch message를 해석해 BRISC, NCRISC, TRISC에서 실제
   operation kernel을 시작하고, 끝나면 dispatcher에 completion을 알린다.

따라서 `reader_kernel.cpp`, `compute_kernel.cpp`, `writer_kernel.cpp` 같은 파일은
operation의 payload이지 범용 runtime 자체는 아니다. 반대로 `cq_prefetch.cpp`,
`cq_dispatch.cpp`, `brisc.cc`, `trisc.cc`는 여러 operation을 실행시키는 runtime
firmware에 가깝다.

```text
HOST

+---------+      +------------------+
| TTNN Op |----->| Device Operation |
+---------+      +------------------+
                         |
                         v
                +------------------+
                | Program Cache    |
                | Miss: Build      |
                | Hit: Patch RTAs  |
                +------------------+
                         |
                         v
+------------+   +--------------+   +--------------------+
| Program(s) |-->| MeshWorkload |-->| FDMeshCommandQueue |
+------------+   +--------------+   +--------------------+
                                             |
                                             v
                                   +------------------+
                                   | System Memory CQ |
                                   +------------------+
                                             |
===================== Host / Device Boundary =====================
                                             v
                                   +-------------+
                                   | cq_prefetch |
                                   +-------------+
                                             |
                                             v
                                   +-------------+
                                   | cq_dispatch |
                                   +-------------+
                                             |
                                    Launch Message + GO
                                             |
                                             v
                                   +------------------+
                                   | Tensix Worker FW |
                                   +------------------+
                                     |       |       |
                                     v       v       v
                                   Reader  Compute  Writer
                                     \_______|_______/
                                             |
                                       Completion
```

## runtime은 어느 디렉터리에 있는가

공식 커밋에는 runtime 전체를 대표하는 최상위 `runtime/` 디렉터리가 없다. 무엇을
확인하려는지에 따라 출발점이 달라진다.

| 확인하려는 것 | 먼저 볼 공식 경로 | 역할 |
| --- | --- | --- |
| TTNN operation 제출과 cache | [`ttnn/api/ttnn/device_operation.hpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/api/ttnn/device_operation.hpp#L254-L410) | output 생성, cache hit/miss, runtime argument 갱신, enqueue |
| operation별 program 생성 | [`ttnn/cpp/ttnn/operations`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations) | core 배치, kernel, CB, compile/runtime argument 구성 |
| 공개 Metalium API | [`tt_metal/api/tt-metalium`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/api/tt-metalium) | `Program`, `MeshDevice`, buffer, enqueue API |
| mesh 실행 계층 | [`tt_metal/distributed`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed) | `MeshWorkload`, Fast/Slow `MeshCommandQueue` |
| command 생성 | [`tt_metal/impl/program`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/program) | runtime args, config, binary, launch, GO command 조립 |
| host command queue | [`tt_metal/impl/dispatch`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch) | hugepage queue와 completion 상태 관리 |
| device dispatch firmware | [`tt_metal/impl/dispatch/kernels`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/kernels) | host command fetch, decode, NoC write, launch |
| Tensix worker firmware | [`tt_metal/hw/firmware/src`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/firmware/src) | operation kernel의 준비, 시작, 완료 보고 |
| device I/O 하위 경계 | [`tt_metal/llrt`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/llrt)와 [TT-UMD](https://github.com/tenstorrent/tt-umd/tree/2f923261d5a383cacaa22b2c41a75612c12cf344) | device 탐색, 좌표 변환, DMA/MMIO 접근 |

TTNN operation 하나를 따라가려면 첫 번째 행에서 시작해 아래로 내려가면 된다.
Metalium C++ program을 직접 작성했다면 `tt_metal/distributed`부터 읽으면 된다.
dispatch hang이나 launch 문제를 디버깅한다면 host 쪽
`fd_mesh_command_queue.cpp`와 device 쪽 `cq_prefetch.cpp`, `cq_dispatch.cpp`를 함께
봐야 한다.

## runtime의 핵심 객체

### `Program`

`Program`은 한 kernel binary를 뜻하지 않는다. 어느 logical core에서 어떤 data
movement·compute kernel을 실행할지, 어떤 circular buffer와 semaphore를 쓸지,
compile-time argument와 runtime argument를 어떻게 배치할지를 담는 host-side 실행
설명이다. enqueue 전 compile 과정에서 kernel binary와 L1 배치 정보가 구체화된다.

### `MeshWorkload`

`MeshWorkload`는 `Program`을 `MeshCoordinateRange`에 연결한다. 한 workload 안에서
서로 다른 program의 device range는 겹칠 수 없으며, 일부 device를 비워 두는 것은
가능하다. 이 계약은 공식
[`mesh_workload.hpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/api/tt-metalium/mesh_workload.hpp#L20-L36)에
명시돼 있다.

single-chip 코드도 이 abstraction을 그대로 쓴다. 즉, device 하나도 `1x1 mesh`의
한 coordinate range로 취급한다.

### `MeshCommandQueue`

`MeshCommandQueue`는 workload와 buffer transfer를 순서대로 제출하는 경계다.
`MeshDevice`는 runtime option에 따라
[`FDMeshCommandQueue` 또는 `SDMeshCommandQueue`를 생성](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/mesh_device.cpp#L1478-L1499)한다.
여기서 FD는 Fast Dispatch, SD는 Slow Dispatch를 뜻한다.

다음 코드는 이 세 객체의 관계만 남긴 축약 예다.

```cpp
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

auto device = MeshDevice::create_unit_mesh(/*device_id=*/0);
auto& cq = device->mesh_command_queue(/*cq_id=*/0);

Program program = CreateProgram();
// CreateKernel(...), CreateCircularBuffer(...), SetRuntimeArgs(...)

MeshWorkload workload;
workload.add_program(MeshCoordinateRange(device->shape()), std::move(program));

EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
Finish(cq);
```

공식 guide의 vector-add 예제도 동일하게 `Program`을 `MeshWorkload`에 넣어
enqueue한다. 공개 함수 선언은
[`distributed.hpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/api/tt-metalium/distributed.hpp#L34-L113)에서
확인할 수 있다.

## device를 열 때 준비되는 것

Fast Dispatch는 첫 operation 호출 때 갑자기 생기지 않는다. `MeshDevice`를 만드는
과정에서 host queue와 device firmware가 먼저 준비된다.

1. `MeshDevice::create_unit_mesh()`가 device 하나를 mesh abstraction으로 연다. 이
   API의 `num_command_queues` 기본값은
   [`1`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/api/tt-metalium/mesh_device.hpp#L305-L320)이다.
2. `CommandQueueInitializer`가 MMIO device에 대응하는 host-side command queue를
   초기화한다. 소스 주석도 이 queue를
   [`hugepage에 연결된 hardware command queue`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/device/firmware/command_queue_initializer.cpp#L79-L89)로
   설명한다.
3. `DispatchKernelInitializer`가 dispatch topology의 program을 만들고 compile한 뒤
   device command queue를 초기화한다. 구체적인 순서는
   [`compile_dispatch_kernels()`와 `init_device_command_queues()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/device/firmware/dispatch_kernel_initializer.cpp#L119-L184)에
   드러난다.
4. 일반 hardware 경로에서 `SystemMemoryManager`는 LLRT의 `host_dma_address()`로
   얻은 hugepage를 command queue 시작 주소로 사용한다. DRAM-backed queue는 별도
   niche 경로이며 더 느리다고
   [`system_memory_manager.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/system_memory_manager.cpp#L122-L220)에
   명시돼 있다.

device 쪽 `cq_prefetch`와 `cq_dispatch`는 반복 loop를 도는 상주 firmware다. 반면
operation의 reader·compute·writer kernel은 workload마다 선택되고 launch된다. 이
차이를 놓치면 dispatch core의 kernel과 worker core의 kernel을 같은 종류로 오해하기
쉽다.

## TTNN operation 한 번이 실행되는 전체 경로

이제 `ttnn.matmul()` 같은 operation이 호출됐다고 가정하고 한 단계씩 내려가 보자.

### 1. TTNN이 output과 cache key를 만든다

operation별 wrapper는 정규화된 attribute와 tensor argument를
`ttnn::device_operation::launch<...>()`에 넘긴다. 예를 들어 matmul은
[`MatmulDeviceOperation`을 launch](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L2840-L2876)한다.

공통 `launch()`는 다음 일을 한다.

- input tensor가 device tensor이고 할당돼 있는지 확인한다.
- output tensor를 먼저 만든다.
- input 또는 operation attribute에서 `MeshDevice`를 찾는다.
- operation adapter로 넘겨 program cache hit/miss를 처리한다.

이 함수의 주석은 반환된 output이 할당돼 있더라도 operation 완료 전에는 값이 바로
준비되지 않는다고 명시한다. 관련 경로는
[`device_operation.hpp`의 `launch()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/api/ttnn/device_operation.hpp#L435-L503)에서
볼 수 있다.

### 2. cache miss면 workload를 만들고, hit면 runtime state를 갱신한다

program cache가 활성화돼 있으면 operation attribute와 tensor argument로 key를
계산한다. cache miss에서는 operation이 program factory를 선택해 새로운
`MeshWorkload`를 만든다. matmul만 해도 program config에 따라 여러 factory 중 하나를
고르는 것을
[`select_program_factory()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L2127-L2150)에서
확인할 수 있다.

cache hit에서는 cached workload를 그대로 제출하기 전에
`override_runtime_arguments()`를 호출한다. shape와 program 구조는 같더라도 이번
호출의 input/output buffer 주소는 달라질 수 있기 때문이다. 공통 cache-hit 경로는
[`handle_mesh_adapter_cache_hit()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/api/ttnn/device_operation.hpp#L254-L289),
실제 matmul의 주소 갱신 예는
[`override_runtime_arguments_impl()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp#L3048-L3095)에서
볼 수 있다.

마지막에는 workload 안의 모든 `Program`에 같은 runtime ID를 넣고
`EnqueueMeshWorkload(..., false)`를 호출한다. 즉, 일반 TTNN operation 제출은 이
지점에서 비동기 Metalium queue로 넘어간다.

### 3. `EnqueueMeshWorkload`가 compile, binary load, command 생성을 연결한다

정상 Fast Dispatch 경로에서 공식
[`EnqueueMeshWorkload()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/distributed.cpp#L25-L129)는
아래 세 단계를 순서대로 호출한 뒤 실제 mesh command queue에 넘긴다.

1. `compile()`: kernel binary를 JIT compile하고, circular buffer를 할당·검증하고,
   L1 내부 구조의 상대 offset을 확정한다.
2. `load_binaries()`: 이 workload를 처음 enqueue할 때 kernel binary용 DRAM
   `MeshBuffer`를 만들고 binary 데이터를 queue에 쓴다.
3. `generate_dispatch_commands()`: 각 `Program`을 device dispatch command sequence로
   바꾼다.

이 세 단계의 구현은
[`mesh_workload.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/mesh_workload.cpp#L100-L223)에
연속해서 있다. binary 상태는 첫 제출에서 `InFlight`, dispatcher에 command를 쓴
뒤 `Committed`가 되며, 이후 재제출에서는 같은 binary buffer를 다시 만들지 않는다.

### 4. `FDMeshCommandQueue`가 현재 device 상태를 반영한다

미리 만든 command sequence에도 enqueue 시점에만 알 수 있는 값이 있다. 예를 들면
worker completion count, L1 kernel-config ring buffer의 이번 slot, launch-message write
pointer, 현재 command queue ID다.

[`FDMeshCommandQueue::enqueue_mesh_workload()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/fd_mesh_command_queue.cpp#L382-L555)는
다음을 수행한다.

- workload가 사용할 worker 수와 GO signal 방식을 계산한다.
- L1 kernel-config ring buffer 공간을 예약한다.
- 조건이 맞으면 prefetcher cache의 hit와 offset을 조회한다.
- cached command sequence의 동적 필드를 현재 상태에 맞게 patch한다.
- workload가 배치된 각 physical device의 system-memory queue에 command를 쓴다.
- launch-message write pointer와 예상 worker completion count를 전진시킨다.

따라서 TTNN program cache와 dispatch command cache는 같은 것이 아니다. 전자는
operation 수준에서 `MeshWorkload` 생성을 재사용하고, 후자는 이미 조립한 device
command의 정적 부분을 재사용한다.

### 5. host가 command sequence를 issue queue에 쓴다

`ProgramImpl::generate_dispatch_commands()`는 sub-device manager 상태를 key로
`ProgramCommandSequence`를 cache한다. 최초에는 preamble, stall, runtime args,
program config, binary transfer, launch message, GO signal용 command를 조립한다.
이 구조는
[`program.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/program/program.cpp#L2256-L2294)에서
확인할 수 있다.

실제로 queue에 기록하는 순서는 다음과 같다.

```text
Preamble
  -> Optional Stall
  -> Runtime Arguments
  -> Program Config
  -> Optional Stall
  -> Binary Transfer or Wait Barrier
  -> Launch Message
  -> GO Signal
```

이 순서는
[`write_program_command_sequence()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/program/dispatch.cpp#L2884-L2992)에
그대로 나타난다. `SystemMemoryManager`는 issue queue 공간을 예약하고 command bytes를
쓴 다음 write pointer를 전진시킨다. 이어 fetch queue entry를 써서 device
prefetcher에 새 command가 있음을 알린다. 관련 공개 내부 API는
[`system_memory_manager.hpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/system_memory_manager.hpp#L65-L107)에
모여 있다.

### 6. `cq_prefetch`가 command를 가져온다

device의 prefetch firmware는 fetch queue를 확인하고 host-visible issue queue에서
command batch를 가져온다. host와 device가 같은 dispatch core에 있는 단순 topology도
있고 relay stage가 분리되는 topology도 있으므로, 실제 core 수와 연결 방식은
architecture와 mesh 구성에 따라 달라진다.

공통적인 역할은 같다. `cq_prefetch`가 prefetch command를 해석하고 inline, linear,
paged payload를 downstream command-data buffer로 relay한다. host-side variant의 반복
loop와 variant 선택은
[`cq_prefetch.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp#L2967-L3167)에
있다.

### 7. `cq_dispatch`가 worker L1을 설정하고 GO를 보낸다

`cq_dispatch`는 relay된 dispatch command를 decode한다. command switch에는 linear,
paged, packed write, wait, GO signal 등이 있다. 이 write command들이 runtime args,
circular-buffer config, semaphore, kernel text, launch message를 필요한 worker L1
위치로 전달한다.

[`process_cmd_d()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/kernels/cq_dispatch.cpp#L1297-L1420)는
command 종류를 보여 준다. GO command를 처리할 때는 필요한 이전 worker completion
count를 먼저 기다린 뒤 worker grid에 signal을 multicast 또는 unicast한다. 이
동기화는
[`process_go_signal_mcast_cmd()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/kernels/cq_dispatch.cpp#L1165-L1212)에서
볼 수 있다.

### 8. Tensix worker firmware가 operation kernel을 시작한다

worker Tensix의 BRISC firmware는 GO signal 또는 preload된 launch message를 기다린다.
유효한 launch를 받으면 다음 순서로 움직인다.

1. launch message에서 enable mask와 kernel-config base를 읽는다.
2. circular buffer interface와 NoC 상태를 설정한다.
3. NCRISC와 세 TRISC를 깨우고, BRISC에 배치된 data-movement kernel도 시작한다.
4. 각 processor가 kernel text offset에 있는 operation kernel entry를 호출한다.
5. 모든 processor의 완료를 기다린 뒤 GO mailbox를 `DONE`으로 바꾼다.
6. dispatcher의 worker-completion counter를 atomic으로 갱신하고 launch-message read
   pointer를 전진시킨다.

전체 control loop는
[`brisc.cc`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/firmware/src/tt-1xx/brisc.cc#L354-L596)에
있다. compute 쪽 TRISC도 자신의 GO를 기다린 뒤 launch message에서 CB와 runtime-arg
base를 설정하고 kernel entry를 호출한다. 이 흐름은
[`trisc.cc`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/firmware/src/tt-1xx/trisc.cc#L108-L224)에서
확인할 수 있다.

### 9. completion이 host까지 돌아온다

worker가 dispatcher의 completion count를 올리면 뒤의 wait와 GO command가 진행될 수
있다. host가 `Finish()`를 호출하거나 blocking enqueue를 사용하면
`FDMeshCommandQueue`는 host로 record event를 보내고 outstanding completion read가
끝날 때까지 기다린다. 구현은
[`finish_nolock()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/fd_mesh_command_queue.cpp#L686-L729)에
있다.

비동기 enqueue가 “실행 완료”를 뜻하지 않는 이유가 여기 있다. enqueue는 host
queue에 명령을 넣는 데까지이며, 결과를 읽거나 재사용하기 전 필요한 동기화는
command ordering, event, blocking read, `Finish()` 중 적절한 수단으로 보장해야 한다.

## runtime argument가 kernel까지 가는 경로

runtime argument는 이름 그대로 compile 뒤 실행할 때 바뀔 수 있는 값이다. 대표적인
예는 input/output buffer 주소, tile 수, 시작 offset이다.

```text
SetRuntimeArgs
      |
      v
Host Kernel Object
      |
      v
Packed Dispatch Write
      |
      v
Worker L1 Kernel Config
      |
      v
Firmware RTA Base
      |
      v
get_arg_val<T>(index)
```

host의 `SetRuntimeArgs()`는 즉시 device에 쓰지 않는다. 먼저 `Program` 안의 `Kernel`
객체가 core별 argument vector를 보관한다. API entry와 저장 호출은
[`tt_metal.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/host_api/tt_metal.cpp#L1791-L1848)에
있다.

dispatch command를 조립할 때 이 vector는 L1 destination과 함께 packed-write
command가 된다. 시작점은
[`generate_runtime_args_cmds()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/program/dispatch.cpp#L510-L570)다.
worker firmware는 launch message에 들어 있는 offset으로 `rta_l1_base`와
`crta_l1_base`를 설정한다. operation kernel의 `get_arg_val<T>()`와
`get_common_arg_val<T>()`는 그 L1 주소에서 4-byte 값을 읽는다. 공식 API 설명은
[`Runtime Arguments`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/host_apis/runtime_args/runtime_args.html)와
[`get_arg_val`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/kernel_args/get_arg_val.html)에
정리돼 있다.

| 종류 | 결정 시점 | 변경의 의미 | kernel에서 읽는 방법 |
| --- | --- | --- | --- |
| `compile_args`, `defines` | `CreateKernel` | 다른 compiled instantiation과 binary를 만들 수 있음 | `get_compile_time_arg_val()`, generated define |
| unique runtime args | enqueue 전, core별 | binary를 바꾸지 않고 core별 값 갱신 | `get_arg_val<T>()` |
| common runtime args | enqueue 전, kernel 공통 | 같은 kernel이 놓인 core에 공통 값 제공 | `get_common_arg_val<T>()` |

legacy `SetRuntimeArgs()`로 이미 만든 unique argument 목록을 갱신할 때는 argument
개수를 바꿀 수 없다. common runtime args도 같은 API로는 한 번만 설정하며, 재사용
코드는 `GetRuntimeArgs()` 또는 `GetCommonRuntimeArgs()`로 기존 storage를 갱신한다.
이 제약은
[`Kernel::set_runtime_args()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/kernels/kernel.cpp#L725-L809)에
명시돼 있다.

## cache는 한 층이 아니다

runtime을 읽다 보면 모두 cache라고 불리지만 수명과 목적이 다른 구조가 여러 개
나온다.

| cache 또는 상태 | 소유 계층 | 재사용하는 것 | 매 실행에 바뀌는 것 |
| --- | --- | --- | --- |
| TTNN program cache | `MeshDevice` / `device_operation` | factory 결과와 `MeshWorkload` | tensor 주소 등 runtime state |
| kernel build cache | JIT build 계층 | 같은 compile config의 binary artifact | runtime args에는 영향 없음 |
| cached program command sequence | `ProgramImpl` | 조립된 dispatch command의 정적 구조 | L1 slot, counters, write pointers, queue ID |
| program binary status | `MeshWorkloadImpl` | 이미 할당하고 보낸 DRAM binary buffer | `NotSent` → `InFlight` → `Committed` |
| prefetcher cache | `FDMeshCommandQueue` / device prefetcher | 크기가 맞는 kernel transfer data | cache hit 여부와 offset |

compile-time 값과 runtime 값의 build-cache 경계는
[`Kernel::compute_hash()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/kernels/kernel.cpp#L528-L625)에도
드러난다. kernel source, compile args, defines는 hash에 들어가지만 runtime argument의
값 자체는 build 대상이 아니다.

중요한 점은 program cache hit여도 `EnqueueMeshWorkload()` 이하가 사라지는 것은
아니라는 사실이다. cached workload의 runtime args를 고친 뒤 동일한 compile/load/
generate entry를 지나며, 각 단계 내부의 상태와 cache가 불필요한 작업을 줄인다.
“cache hit이므로 dispatch 비용이 0”이라고 이해하면 실제 실행 경로와 맞지 않는다.

## Fast Dispatch와 Slow Dispatch

[공식 guide](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/METALIUM_GUIDE.md#L514-L540)는
production workload에는 Fast Dispatch를 사용하고, dispatch 자체의 문제를 분리해야
할 때만 Slow Dispatch를 쓰도록 권장한다.

| 구분 | Fast Dispatch | Slow Dispatch |
| --- | --- | --- |
| host 제출 | DMA-visible command queue에 command sequence 기록 | host가 program과 transfer를 직접 진행 |
| device 역할 | 전용 prefetch/dispatch firmware가 command 처리 | Fast Dispatch queue firmware를 우회 |
| 비동기성 | enqueue 후 host와 device가 겹쳐 진행 가능 | 기본 디버그 경로는 단계별 host 동기화 |
| 순서 | queue 하나 안에서는 in-order | host가 순차적으로 제어 |
| 주 용도 | 일반 실행과 성능 측정 | dispatch 문제와 kernel 문제의 분리 |

Slow Dispatch는 device를 열기 전에 환경 변수로 선택한다.

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 ./your_metal_program
```

구현상 Slow Dispatch queue는 worker가 idle인지 확인하고 program을 직접 launch하며,
`Finish()`에서 core와 memory barrier를 기다린다. 관련 코드는
[`sd_mesh_command_queue.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/sd_mesh_command_queue.cpp#L276-L415)에
있다.

공식 guide에는 queue 0을 compute, queue 1을 transfer에 쓰는 2-queue 예제가 나온다.
다만 이 기준 커밋의 `MeshDevice::create_unit_mesh()` API 기본값은 queue 하나다. 두
queue를 쓰려면 `num_command_queues`를 명시해서 device를 만들고, queue 사이의
dependency는 `MeshEvent`로 표현해야 한다. “모든 device가 기본으로 두 queue를
연다”고 가정하면 현재 API와 다르다.

## LLRT와 UMD는 어디에 연결되는가

Metalium 아래에는 LLRT와
[UMD(User Mode Driver)](https://github.com/tenstorrent/tt-umd/blob/2f923261d5a383cacaa22b2c41a75612c12cf344/README.md#L1-L15)가
있다. 이들은
operation factory처럼 compute 계획을 만들지 않는다. device를 발견하고 열며, logical
coordinate를 hardware coordinate로 바꾸고, host DMA memory와 device read/write를
제공하는 하위 계층이다.

`tt_metal/llrt/tt_cluster.cpp`의 `Cluster::open_driver()`는 silicon target에서
[`tt::umd::Cluster`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/llrt/tt_cluster.cpp#L392-L397)를
생성한다. direct core access도 좌표를 변환한 뒤 UMD의 DMA 또는 device read/write를
호출한다. 그 경계는
[`Cluster::write_core()`와 `read_core()`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/llrt/tt_cluster.cpp#L857-L910)에서
명확히 보인다.

Fast Dispatch의 steady state에서는 host가 operation command마다 direct
`write_core()`를 반복하는 것이 아니다. UMD를 통해 준비된 DMA-visible hugepage에
host가 command를 쌓고, device prefetch firmware가 가져가는 것이 핵심이다. Slow
Dispatch나 초기화, 일부 직접 I/O 경로를 읽을 때는 LLRT/UMD 호출이 더 직접적으로
나타난다.

## 소스를 읽는 추천 순서

runtime 전체를 처음 읽는다면 다음 순서가 가장 덜 끊긴다.

1. [`METALIUM_GUIDE.md`의 Running code와 Fast dispatch](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/METALIUM_GUIDE.md#L280-L542)로
   `Program`, runtime args, queue의 공개 모델을 익힌다.
2. 관심 operation의 entry와 program factory를 찾는다. matmul이라면
   [`matmul_device_operation.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L2127-L2150)에서
   시작한다.
3. [`device_operation.hpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/api/ttnn/device_operation.hpp#L364-L410)에서
   program cache hit/miss와 enqueue 경계를 본다.
4. [`distributed.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/distributed.cpp#L116-L129)와
   [`mesh_workload.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/mesh_workload.cpp#L100-L223)에서
   compile, binary load, command 생성을 연결한다.
5. [`fd_mesh_command_queue.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/fd_mesh_command_queue.cpp#L382-L555)에서
   enqueue 시점의 ring-buffer와 completion state를 본다.
6. [`impl/program/dispatch.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/program/dispatch.cpp#L2218-L2290)에서
   host command bytes가 어떻게 조립되는지 본다.
7. [`cq_prefetch.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp#L2967-L3167)와
   [`cq_dispatch.cpp`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/kernels/cq_dispatch.cpp#L1297-L1420)에서
   device dispatch command의 소비 과정을 본다.
8. 마지막으로 [`brisc.cc`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/firmware/src/tt-1xx/brisc.cc#L388-L592)와
   [`trisc.cc`](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/firmware/src/tt-1xx/trisc.cc#L133-L224)로
   GO 이후 실제 kernel entry가 호출되는 지점을 확인한다.

이 순서로 보면 “operation이 어떤 kernel을 만들었는가”와 “runtime이 그 kernel을
어떻게 실행했는가”를 분리해서 추적할 수 있다.

## 제약과 해석 시 주의점

- Fast Dispatch topology는 architecture, local/remote device, mesh와 Fabric 구성에
  따라 prefetch·dispatch stage가 합쳐지거나 추가 relay를 가질 수 있다. 위 그림은
  역할을 보여 주는 최소 모델이다.
- 현재 `EnqueueMeshWorkload()`에는 claimed service core만 대상으로 하는 workload를
  Slow Dispatch로 보내는 별도 분기가 있다. 이 글은 그 경로가 아닌 일반 worker-grid
  workload를 다룬다.
- `MeshWorkload`, sub-device, trace, prefetcher cache는 계속 개발 중인 내부 구조와
  맞물린다. 다른 커밋을 읽을 때는 이 문서의 line anchor보다 symbol 이름으로 다시
  검색하는 편이 안전하다.
- async API의 반환, device execution 완료, host가 결과를 읽을 수 있는 시점은 서로
  다르다. 성능 분석과 correctness debugging에서 이 세 시점을 구분해야 한다.
- reader·compute·writer는 관례적인 세 역할이지 모든 operation이 정확히 세 source
  file이나 동일한 core 배치를 가져야 한다는 뜻은 아니다.

## 핵심 정리

- TT-Metalium runtime은 한 폴더가 아니라 TTNN, Metalium host, dispatch firmware,
  worker firmware, LLRT/UMD를 잇는 실행 경로다.
- TTNN에서 시작하면 `device_operation.hpp`, Metalium에서 시작하면
  `distributed.cpp`와 `fd_mesh_command_queue.cpp`가 첫 관문이다.
- `Program`은 core별 실행 설명이고, `MeshWorkload`는 program을 mesh range에 매핑하며,
  `MeshCommandQueue`가 이를 제출한다.
- Fast Dispatch는 host hugepage의 command queue와 device의 `cq_prefetch`/
  `cq_dispatch` firmware로 host와 device 실행을 겹친다.
- worker firmware는 launch message와 GO signal을 받아 operation kernel을 호출하고
  completion을 dispatcher에 돌려준다.
- runtime args는 host `Kernel` 객체에서 dispatch packed write를 거쳐 worker L1에
  도달하며, program cache hit에서는 주로 buffer 주소 같은 값만 갱신된다.
- cache는 TTNN program cache, build cache, command-sequence cache, binary 상태,
  prefetcher cache로 나누어 이해해야 한다.

## 참고 자료

- [TT Architecture and Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/METALIUM_GUIDE.md)
- [TT-Metalium Runtime Arguments API](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/host_apis/runtime_args/runtime_args.html)
- [TT-Metalium `get_arg_val` API](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/kernel_args/get_arg_val.html)
- [TTNN device operation 실행과 program cache](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/ttnn/api/ttnn/device_operation.hpp)
- [Mesh workload enqueue 경로](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/distributed.cpp)
- [Fast Dispatch mesh command queue](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/distributed/fd_mesh_command_queue.cpp)
- [Host program dispatch command 생성](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/program/dispatch.cpp)
- [Device prefetch firmware](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp)
- [Device dispatch firmware](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/impl/dispatch/kernels/cq_dispatch.cpp)
- [Tensix `tt-1xx` BRISC worker firmware](https://github.com/tenstorrent/tt-metal/blob/69096826694cac0e8bbde0050e38a3e411a6d91e/tt_metal/hw/firmware/src/tt-1xx/brisc.cc)
- [TT-UMD official repository](https://github.com/tenstorrent/tt-umd/tree/2f923261d5a383cacaa22b2c41a75612c12cf344)
