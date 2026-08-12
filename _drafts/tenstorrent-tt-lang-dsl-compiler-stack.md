---
title: "TT-Lang은 무엇인가: DSL부터 TT-Metal과 자동 튜닝까지"
date: 2026-08-12 10:41:59 +0900
categories: [Tenstorrent, compiler]
tags: [tt-lang, dsl, mlir, tt-metal]
description: "TT-Lang의 Python DSL 모델, MLIR 기반 컴파일 경로, TT-Metalium과의 관계, Triton식 자동 튜닝 지원 여부를 소스 코드 기준으로 분석합니다."
render_with_liquid: false
---

## 문서 범위

- TT-Lang 기준: `6b2c56d48b317b53fc43a43dd66bc9144fd894d7`, `v1.1.7-34-g6b2c56d48`
- 기준 브랜치와 시점: `main`, 2026-08-12
- 결합된 TT-Metal 버전: `v0.75.0`, submodule commit `d9a68815f5fcf08a5bfbffb6f1f811823fba8edd`
- 로컬 dependency 상태: LLVM과 TT-Metal submodule worktree는 초기화되지 않음
- 교차 확인 자료: TT-Lang·TT-Metalium·Triton 공식 문서와 위 commit의 소스 코드
- 확인일: 2026-08-12

이 글은 TT-Lang을 처음 접했을 때 생기는 세 가지 질문에 답한다.

1. DSL은 무엇이며 TT-Lang은 왜 DSL인가?
2. TT-Lang 아래에는 무엇이 있고, 그 최하위 소프트웨어 계층이 TT-Metal인가?
3. Triton의 `@triton.autotune`과 같은 자동 튜닝 기능을 제공하는가?

결론부터 요약하면, TT-Lang은 **Python에 내장된 Tenstorrent 사용자 정의 연산용 DSL**이다. Python 소스를 MLIR로 바꾸고 TT-Metal 커널 C++을 생성한 뒤, `ttnn.generic_op`과 TT-Metalium 런타임으로 실행한다. 따라서 소프트웨어 스택의 기반에는 `tt-metal` 저장소가 있다. 다만 하드웨어까지 포함하면 TT-Metalium 아래에 TT-LLK, 펌웨어와 Tensix 하드웨어가 더 있다. 분석한 commit에는 Triton처럼 여러 설정을 실측해 최적값을 자동 선택하는 공개 autotuner가 없다.

## DSL이란 무엇인가

DSL(Domain-Specific Language)은 특정 문제 영역을 표현하도록 범위를 좁힌 언어다. Python이나 C++처럼 여러 종류의 프로그램을 만들 수 있는 범용 언어와 달리, DSL은 한 영역의 개념을 짧고 명확하게 표현하는 데 집중한다.

예를 들어 SQL은 테이블 조회와 갱신을 표현하는 DSL이다. 정규 표현식은 문자열 패턴을 표현하는 DSL이다. 같은 관점에서 TT-Lang은 Tenstorrent 커널의 계산, data movement, 메모리 배치, 동기화와 코어 간 통신을 표현한다.

DSL이라고 해서 반드시 독립된 문법과 컴파일러 실행 파일을 가져야 하는 것은 아니다. TT-Lang은 Python 문법 안에 들어가는 **내장형 DSL(embedded DSL)**이다. 사용자는 `@ttl.operation`, `@ttl.compute`, `@ttl.datamovement` 데코레이터와 `ttl.copy`, `DataflowBuffer`, `PipeNet` 같은 객체를 사용한다. 덧셈이나 행렬곱도 Python의 `+`, `@`로 적는다. 그러나 데코레이터 안의 모든 Python 표현을 자유롭게 실행하는 것은 아니다. 컴파일러가 이해하는 제한된 Python AST(Abstract Syntax Tree)만 허용한다. [TT-Lang 명세](https://docs.tenstorrent.com/tt-lang/specs/TTLangSpecification.html)도 Python host language의 제한된 부분집합이라는 점을 DSL의 근거로 든다.

다음 코드는 저장소의 elementwise add 예제를 핵심 구조만 남겨 단일 코어용으로 줄인 것이다.

```python
@ttl.operation(grid=(1, 1))
def add(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(2, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            out_dfb.reserve() as out_blk,
        ):
            out_blk.store(a_blk + b_blk)

    @ttl.datamovement()
    def read():
        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
            ttl.copy(a[0:2, 0:1], a_blk).wait()
            ttl.copy(b[0:2, 0:1], b_blk).wait()

    @ttl.datamovement()
    def write():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0:2, 0:1]).wait()
```

겉모습은 Python이지만 각 표현에는 하드웨어 의미가 붙는다.

| TT-Lang 표현 | 의미 |
| --- | --- |
| `@ttl.operation(grid=...)` | 어떤 Tensix 코어 격자에 사용자 정의 연산을 배치할지 선언한다. |
| `@ttl.datamovement()` | DRAM·L1·코어 사이에서 데이터를 옮기는 커널을 정의한다. |
| `@ttl.compute()` | 타일 또는 블록 계산을 수행하는 compute 커널을 정의한다. |
| `DataflowBuffer` | 코어 L1의 버퍼와 producer/consumer 동기화 경계를 표현한다. |
| `reserve()` / `wait()` | 생산할 공간을 확보하거나 소비할 데이터를 기다린다. |
| `ttl.copy(...).wait()` | 비동기 전송을 시작하고 완료를 기다린다. |
| `out_blk.store(a_blk + b_blk)` | 블록 덧셈 결과를 출력 블록에 기록한다. |

현재 명세에는 두 가지 연산 작성 방식이 있다. Unified-body 방식은 하나의 본문을 쓰면 컴파일러가 문장을 compute와 data movement 스레드로 나눈다. 위 예제와 같은 explicit multi-kernel 방식은 작성자가 `compute`와 두 data movement 커널을 직접 구분한다. 정밀한 스레드 제어가 필요할 때 후자를 사용한다.

### Runtime argument와 compile-time argument

TT-Lang 연산의 parameter는 runtime에 전달되는 TT-NN tensor다. 반면 block shape, loop bound, scalar coefficient, grid 같은 값은 바깥 Python scope에서 capture되는 compile-time 값이다. 같은 연산을 shape, dtype, layout, memory space와 compiler option이 같은 tensor로 다시 호출하면 compile cache를 재사용한다.

이 구분은 뒤에서 살펴볼 튜닝과도 연결된다. 예를 들어 block 크기를 바꾸려면 보통 서로 다른 compile-time 값을 capture한 operation factory를 만들어야 한다. Runtime scalar argument로 block 크기를 자유롭게 넘기고 하나의 커널이 모두 처리하는 모델이 아니다.

## TT-NN, TT-Lang, TT-Metalium 사이의 위치

TT-Lang은 TT-NN과 TT-Metalium 사이의 중간 단계다. 여기서 “중간”은 소스 코드의 추상화 수준을 뜻한다.

| 계층 | 주된 작업 단위 | 사용자가 제어하는 범위 | 대표 용도 |
| --- | --- | --- | --- |
| TT-NN | tensor operation | 이미 최적화된 연산과 tensor layout을 선택한다. | 모델 작성과 배포 |
| TT-Lang | 사용자 정의 연산의 block, dataflow와 grid | 계산, data movement, DFB, pipe를 표현하고 일부 자원 관리는 컴파일러에 맡긴다. | 커널 결합과 새 사용자 정의 연산 |
| TT-Metalium | 호스트 프로그램과 C++ device kernel | CB, semaphore, NoC, 커널 배치와 runtime argument를 낮은 수준에서 직접 구성한다. | 하드웨어를 세밀하게 제어하는 커널 개발 |

TT-NN은 `matmul`, `add`, `relu`처럼 완성된 tensor operation을 제공한다. TT-Lang의 주된 동기는 이런 연산 여러 개를 하나의 사용자 정의 연산으로 결합하거나, TT-NN에 없는 패턴을 구현하는 것이다. TT-Metalium은 같은 일을 더 낮은 수준에서 할 수 있지만 CB index, NoC address, DST register와 커널 설정을 훨씬 직접적으로 다뤄야 한다.

TT-Lang도 모든 일을 자동화하지는 않는다. 사용자는 tensor를 block으로 나누는 방법, grid, DFB shape와 buffering, data movement 순서 등을 설계한다. 컴파일러는 이 표현에서 compute API 호출, 일부 NoC 주소 계산, DFB physical index, DST register allocation, 동기화와 lowering 세부를 계산한다. 그래서 TT-NN보다 낮고 TT-Metalium보다 높다.

## 한 Tensix core에서 무엇이 실행되는가

TT-Metalium의 일반적인 연산은 Reader, Compute, Writer 파이프라인으로 구성된다. TT-Lang의 explicit multi-kernel 방식도 이 구조를 그대로 드러낸다.

```text
+------+  +-----------+  +-----------+  +---------+  +------------+  +-----------+  +------+
| DRAM |->| Reader DM |->| input DFB |->| Compute |->| output DFB |->| Writer DM |->| DRAM |
+------+  +-----------+  | in L1     |  +---------+  | in L1      |  +-----------+  +------+
                         +-----------+               +------------+
```

그림은 데이터 의존 관계를 단순화한 것이다. DFB(Dataflow Buffer)는 TT-Lang의 논리적 추상화이며 lowering 뒤에는 TT-Metal circular buffer와 동기화 연산으로 구체화된다. 이중 버퍼링을 사용하면 Compute가 현재 block을 처리하는 동안 Reader가 다음 block을 채울 수 있다.

## Python에서 hardware까지의 실제 경로

### 전체 경로

로컬 소스를 따라가면 실행 경로는 다음과 같다.

```text
+---------------------------+
| Python + ttl.operation    |
+-------------+-------------+
              |
              | inspect source, parse Python AST
              v
+---------------------------+
| TTL / TTCore MLIR         |
| ops, types, layouts       |
+-------------+-------------+
              |
              | fusion, DFB, config, DST, loop, pipe passes
              v
+---------------------------+
| TTKernel MLIR             |
| hardware API semantics    |
+-------------+-------------+
              |
              | lower to EmitC, translate to C++
              v
+---------------------------+
| Metal kernel C++          |
| reader / compute / writer |
+-------------+-------------+
              |
              | KernelDescriptor + ProgramDescriptor
              v
+---------------------------+
| ttnn.generic_op           |
| TT-Metalium JIT/runtime   |
+-------------+-------------+
              |
              v
+---------------------------+
| TT-LLK / firmware         |
+-------------+-------------+
              |
              v
+---------------------------+
| Tensix hardware           |
+---------------------------+
```

각 단계를 자세히 살펴보자.

### 1. Python AST를 TTL MLIR로 만든다

`TTLGenericCompiler`는 연산 내부 thread function의 소스를 `ast.parse`로 읽고 TT-Lang이 지원하는 node를 MLIR operation으로 만든다. 이 단계에서 tensor shape와 layout은 TTCore type으로, `ttl.copy`, DFB, block 계산은 TTL dialect operation으로 표현된다. Python을 그대로 device에서 실행하는 것도 아니고, 일반 Python bytecode 전체를 컴파일하는 것도 아니다.

TT-Lang이 MLIR을 쓰는 이유는 추상화를 여러 단계로 나누기 쉽기 때문이다. 처음부터 C++ 문자열을 만드는 대신 상위 데이터 흐름의 의미를 IR에 보존한 상태에서 검증과 변환을 수행할 수 있다.

### 2. TTL pass가 자원과 실행 형태를 결정한다

실제 pipeline은 단일 변환이 아니라 여러 pass의 연속이다. [현재 소스의 pipeline](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/lib/Dialect/TTL/Pipelines/TTLPipelines.cpp)에서 중요한 작업을 역할별로 묶으면 다음과 같다.

| 단계 | 대표 작업 |
| --- | --- |
| 상위 단계 정규화 | loop state materialization, 빠진 copy wait와 CB synchronization 삽입 |
| Compute 형성 | tensor-level expression fusion, tile-level `ttl.compute` 생성 |
| DFB와 Pipe 계획 | intermediate DFB 생성, PipeTransport grouping, lifetime 분석, physical DFB index 할당, L1 budget 검증 |
| Compute 설정 | target 제약에 맞는 FPU/SFPU strategy, DST element width, unpack mode 결정 |
| DST와 loop lowering | DST register 할당, subblock 분할, 명시적 loop와 scheduling 생성 |
| 하드웨어 lowering | TTL operation을 `ttkernel.*` operation으로 변환 |
| C++ 준비 | init 삽입, canonicalization, affine lowering, TTKernel을 EmitC로 변환 |

예를 들어 소스의 `a_blk + b_blk`는 처음에는 block 또는 tensor 수준의 `ttl.add`다. 이후 tile 단위 연산으로 바뀌고, TTKernel 단계에서는 `ttkernel.add_binary_tile` 계열 operation이 된다. 마지막 C++에는 TT-Metal compute API의 `add_binary_tile` 호출이 나타난다.

중요한 점은 이 과정의 “자동 선택”이 곧 autotuning은 아니라는 것이다. FPU와 SFPU 중 가능한 실행 전략을 고르거나 DFB index를 graph coloring으로 배치하는 일은 compile-time 제약 조건 풀이와 heuristic이다. 후보 커널을 하드웨어에서 여러 번 실행해 가장 빠른 결과를 고르는 과정은 아니다.

### 3. EmitC에서 TT-Metal kernel C++을 생성한다

TTKernel은 하드웨어 커널 API에 가까운 dialect다. [C++ backend](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/lib/Target/TTKernel/TTKernelToCpp.cpp)는 이를 EmitC로 낮춘 뒤 각 thread를 `kernel_main` 함수로 감싼 독립 C++ 소스로 변환한다. Compute 커널에는 `api/compute/common.h`와 필요한 compute header를, data movement 커널에는 `api/dataflow/dataflow_api.h`를 넣는다. 사용한 operation에 따라 `compute_kernel_api.h`, reduce API, debug print header나 experimental LLK snippet도 추가한다.

따라서 TT-Lang의 최종 산출물은 자체 virtual machine용 bytecode가 아니다. TT-Metal JIT가 컴파일할 **Metal device kernel C++**이다.

### 4. `ttnn.generic_op`으로 host program을 구성한다

[Python runner](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/python/ttl/kernel_runner.py)는 생성된 Reader, Compute, Writer C++ 경로와 core range, compile-time/runtime argument, compute configuration을 `ttnn.KernelDescriptor`로 묶는다. DFB allocation은 `CircularBufferDescriptor`, 전체 프로그램은 `ttnn.ProgramDescriptor`가 된다. 마지막 호출은 다음과 같다.

```python
return ttnn.generic_op(io_tensors, program)
```

이 때문에 계층 그림에서 TT-NN의 위치가 두 번 보일 수 있다. 위쪽에서는 사용자가 built-in TT-NN 연산과 TT-Lang 사용자 정의 연산을 섞어 모델을 작성한다. 아래쪽 host launch에서는 TT-Lang이 TT-NN의 저수준 `generic_op` descriptor API를 연결부로 사용한다. 그렇다고 TT-Lang 커널이 다시 `ttnn.add`나 `ttnn.matmul` graph로 변환되는 것은 아니다. 실제 실행 본체는 이미 생성된 Metal C++ 커널이다.

### 5. 첫 호출 compile과 cache

Operation wrapper는 tensor의 logical·padded shape, dtype, memory space, memory layout, tile shape, grid, target architecture와 compiler option으로 cache key를 만든다. 같은 key의 다음 호출은 `CompiledTTNNKernel`을 재사용한다. TT-Metalium 쪽에도 JIT와 program cache가 있다.

이것은 **compile cache**이지 **tuning result cache**가 아니다. Cache miss에서 여러 설정을 실행해 최적 후보를 찾는 것이 아니라, 주어진 option으로 커널 하나를 컴파일한다.

## 맨 아래에는 TT-Metal이 있는가

짧게 답하면 **TT-Lang의 실행 기반에는 TT-Metal이 있다.** 다만 이름과 경계를 정확히 구분해야 한다.

- `tt-metal`: TT-NN, TT-Metalium runtime, device kernel API, 펌웨어와 관련 구성 요소가 들어 있는 GitHub 저장소 이름이다.
- TT-Metalium: 그 저장소가 제공하는 저수준 host·kernel programming framework와 API의 제품명이다.
- TT-LLK(Low Level Kernel): 하드웨어 세대별 compute primitive 구현이다.
- Tensix 하드웨어: 소프트웨어 스택 아래의 실제 실행 장치다.

TT-Lang 저장소의 `.gitmodules`는 `third-party/tt-metal`을 공식 submodule로 선언한다. 분석한 commit의 `third-party/tt-metal-version`은 `TT_METAL_TAG="v0.75.0"`과 `TTNN_PYPI="0.75.0"`을 고정한다. [Build script](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/cmake/modules/BuildTTMetal.cmake)는 이 소스에서 `ttnn`과 `ttnncpp` target 및 runtime library를 만들고 펌웨어를 미리 컴파일한다. 즉 TT-Metal은 단순한 참고 코드가 아니라 build와 실행 dependency다.

그렇지만 “맨 아래”를 하드웨어까지 포함한 말로 이해하면 TT-Metalium이 끝은 아니다. 생성된 C++은 TT-Metal device API와 LLK를 거쳐 RISC-V·Tensix engine에서 실행되고, host command는 runtime과 device driver를 통해 하드웨어에 전달된다. LLVM/MLIR은 이 수직 runtime stack의 아래가 아니라, 옆에서 C++을 생성하는 컴파일러 기반 기술이다.

## Simulator 경로는 별개다

`tt-lang-sim`은 TT device와 전체 compiler stack 없이 pure Python으로 연산을 실행하는 functional simulator다. 따라서 simulator 경로는 다음처럼 하드웨어 컴파일 경로를 우회한다.

```text
TT-Lang Python source ---> functional simulator ---> Torch-backed result

TT-Lang Python source ---> MLIR/C++ ---> TT-Metalium ---> Tensix hardware
```

Simulator는 DFB state, synchronization, deadlock, tensor 결과와 data movement 구조를 빠르게 검증하는 데 유용하다. 하지만 Metal C++을 생성하거나 실제 device cycle을 재현하는 performance simulator는 아니다. Simulator가 제공하는 trace와 통계도 하드웨어에서 후보 설정의 실행 시간을 비교하는 autotuner와는 다르다.

## Triton처럼 자동 튜닝하는가

### Triton autotune이 하는 일

Triton의 [`@triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html)은 `triton.Config` 후보 목록과 `key`를 받는다. Key 값이 달라지면 후보를 실제로 평가하고 가장 빠른 설정을 고른다. 여러 후보를 실행하므로 output을 reset하거나 restore하는 hook도 제공하며, 선택 결과를 재사용할 수 있다. 이것이 여기서 말하는 **실측 기반 autotuning**이다.

### 현재 TT-Lang의 결론: 동등한 기능은 없다

분석한 TT-Lang commit에는 다음을 한데 제공하는 공개 데코레이터나 runtime API가 없다.

1. Block shape, grid, buffering, compiler flag 등으로 후보 공간을 선언한다.
2. Input signature별로 후보를 컴파일한다.
3. 하드웨어에서 warmup과 반복 측정을 수행한다.
4. 정확성을 보존하면서 가장 빠른 후보를 고른다.
5. 최적 후보를 signature와 target별로 cache한다.

저장소 전체에서 tuner 관련 경로를 확인하면 오히려 반대 근거가 분명하다. [Pipe optimization design 문서](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/docs/development/PipeOptimizations.md)는 autotuning을 static cost model의 **대안**으로 열거한 뒤, 현재 선택은 static per-Pipe rewrite라고 명시한다.

### 이름이 비슷하지만 다른 기능

| TT-Lang 기능 | 자동으로 하는 일 | Triton식 autotuning인가 |
| --- | --- | --- |
| `TTLANG_AUTO_PROFILE=1` | 각 source line에 signpost를 넣고 cycle report를 출력한다. | 아니다. 측정만 하고 설정을 바꾸지 않는다. |
| `TTLANG_PERF_DUMP=1` | NoC traffic과 kernel wall time을 요약한다. | 아니다. |
| Compile cache | 같은 tensor signature와 option의 compiled kernel을 재사용한다. | 아니다. 후보 비교와 최적값 선택이 없다. |
| `fp32_dest_acc_en=None` | Target capability와 operation requirement로 합법적인 DST width를 결정한다. | 아니다. Constraint resolution이다. |
| `--ttl-pipe-batch-tiles 0` | Compiler policy로 grouping 크기를 자동 선택한다. | 아니다. 하드웨어 benchmark search가 아니다. |
| DFB exact coloring search | Resource limit을 만족하는 DFB index allocation을 탐색한다. | 아니다. 성능 설정 search가 아니다. |
| `CompilerOptions` | 사용자가 optimization pass와 hardware option을 선택한다. | 수동 설정 interface다. |
| Matmul `plan_matmul` | 수동 선정한 shape table 또는 정적 score로 block·grid plan을 하나 고른다. | 일반 autotuner가 아니다. |
| `/ttl-optimize` Claude skill | Agent가 profile을 읽고 코드 수정을 제안·검증하는 workflow다. | Runtime library의 autotuner가 아니다. |

특히 `auto-profiling`의 “auto”는 instrumentation을 자동으로 넣는다는 뜻이다. `autotuning`의 “auto”처럼 설정 탐색과 선택을 자동화한다는 뜻이 아니다.

### Matmul benchmark가 보여 주는 현재 방식

[`benchmarks/matmul/`](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/benchmarks/matmul/README.md)에는 `block_cfg=(bm, bn, bk)`와 `part_cfg=(Mp, Np, Kp)`를 고르는 planner와 하드웨어 benchmark가 있다. 이 코드는 튜닝에 필요한 부품을 보여 주지만 제품 수준 autotuner는 아니다.

- 알려진 shape는 과거 sweep에서 사람이 고른 `SHAPE_PLANS` table을 사용한다.
- 나머지 shape는 core utilization, padding overhead, core별 iteration 수를 score로 계산해 plan 하나를 고른다.
- `sweep.py`는 그 plan을 실행하고 `ttnn.matmul`과 비교한다.
- 첫 연산 호출 때 가능한 plan을 모두 benchmark해 최적 후보를 고르는 경로는 없다.

따라서 현재 TT-Lang에서 Triton식 작업을 하려면 operation factory로 여러 compile-time 설정을 만들고, 별도의 benchmark harness로 compile·warmup·측정·정확성 검사를 수행한 뒤 결과를 저장해야 한다. Built-in profiler와 compiler option은 이 harness를 만드는 재료이지, harness 자체는 아니다.

## TT-Lang과 Triton을 같은 종류로 봐도 되는가

둘 다 Python embedded DSL이며 사용자 정의 accelerator kernel을 작성한다는 점에서는 같은 범주다. 그러나 programming model은 크게 다르다.

| 관점 | TT-Lang | Triton |
| --- | --- | --- |
| Hardware model | Tensix core의 compute와 data movement kernel, DFB, NoC를 드러낸다. | GPU program instance가 pointer와 block 단위 tensor 연산을 수행한다. |
| Data movement | Tensor slice, DFB, copy, pipe를 명시적으로 작성한다. | Load/store와 compiler lowering 중심으로 표현한다. |
| Concurrent kernel 구조 | Reader·Compute·Writer 역할과 synchronization이 핵심이다. | 하나의 JIT kernel body와 program grid가 중심이다. |
| Backend | TTL/TTKernel MLIR에서 TT-Metal C++을 생성한다. | Triton IR/MLIR 계열을 GPU target code로 낮춘다. |
| Tuning | Profiling, compiler option, 정적 heuristic과 수동 sweep을 제공한다. | `@triton.autotune`이 후보 benchmark와 선택을 통합한다. |

따라서 TT-Lang을 “Tenstorrent용 Triton”이라고만 부르면 Python DSL이라는 공통점은 전달되지만, DFB와 Reader·Compute·Writer로 구성되는 dataflow model과 현재 autotuning 차이는 놓치게 된다. TT-Lang은 TT-NN과 TT-Metalium 사이에서 커널 결합을 쉽게 만드는 언어라는 설명이 더 정확하다.

## 현재 snapshot을 읽을 때의 주의점

TT-Lang은 빠르게 변하는 project다. 분석한 commit은 release tag `v1.1.7`에서 34 commit 뒤에 있으며, 바로 전날 LLVM main과 TT-Metal `v0.75.0`으로 dependency가 올라갔다. 문서의 기능 표에도 simulator만 지원하거나 아직 compiler가 지원하지 않는 항목이 있다. 특히 다음을 구분해야 한다.

- README의 vision과 현재 구현.
- Functional simulator 지원과 하드웨어 compiler 지원.
- Compiler가 자동으로 합법적인 resource plan을 찾는 기능과 실측 autotuning.
- TT-Lang 소스가 생성하는 device kernel과 이를 launch하는 TT-NN `generic_op` adapter.

다른 revision을 분석한다면 `third-party/tt-metal-version`, compiler option 목록, `TTLPipelines.cpp`, `ttl_api.py`의 runtime path, functionality matrix와 benchmark directory를 다시 확인해야 한다.

## 핵심 정리

- DSL은 특정 문제 영역에 집중한 언어다. TT-Lang은 제한된 Python syntax로 Tenstorrent 사용자 정의 연산을 표현하는 embedded DSL이다.
- TT-Lang은 TT-NN보다 낮고 TT-Metalium보다 높은 추상화 수준에 있다. 사용자는 block, data movement와 synchronization을 설계하고 compiler는 resource allocation과 lowering 일부를 맡는다.
- 실제 compile 경로는 `Python AST -> TTL/TTCore MLIR -> TTKernel -> EmitC -> Metal C++`이다.
- Host에서는 생성된 kernel을 descriptor로 묶어 `ttnn.generic_op`을 호출한다. TT-Metalium이 JIT compile과 device dispatch를 담당한다.
- `tt-metal`은 TT-Lang의 실제 build·runtime dependency다. 그 아래에는 TT-LLK, 펌웨어와 Tensix 하드웨어가 있다.
- Functional simulator는 compiler와 TT-Metal을 우회하는 정확성 검증 경로다. 하드웨어 성능을 자동 탐색하지 않는다.
- 현재 snapshot에는 Triton `@triton.autotune`과 동등한 실측 기반 autotuner가 없다. Profiling, compile cache, compiler heuristic, 수동 benchmark는 각각 유용하지만 autotuning과는 다른 기능이다.

## 참고 자료

- Tenstorrent, [TT-Lang Introduction](https://docs.tenstorrent.com/tt-lang/overview.html).
- Tenstorrent, [TT-Lang Specification](https://docs.tenstorrent.com/tt-lang/specs/TTLangSpecification.html).
- Tenstorrent, [TT-Lang Performance Tools](https://docs.tenstorrent.com/tt-lang/reference/performance-tools.html).
- Tenstorrent, [TT-Lang Compiler Options](https://docs.tenstorrent.com/tt-lang/reference/compiler-options.html).
- Tenstorrent, [TT-Metalium Getting Started와 software stack](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html).
- Triton, [`triton.autotune` API](https://triton-lang.org/main/python-api/generated/triton.autotune.html).
- TT-Lang snapshot, [`README.md`](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/README.md)와 [language specification source](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/docs/sphinx/specs/TTLangSpecification.md).
- TT-Lang snapshot, [Python AST frontend](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/python/ttl/_src/ttl_ast.py), [Python compile path](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/python/ttl/ttl_api.py), [MLIR pass pipeline](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/lib/Dialect/TTL/Pipelines/TTLPipelines.cpp).
- TT-Lang snapshot, [TTKernel C++ emitter](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/lib/Target/TTKernel/TTKernelToCpp.cpp), [Metal header mapping](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/include/ttlang/Target/TTKernel/TTKernelIncludesMap.h), [`ttnn.generic_op` runner](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/python/ttl/kernel_runner.py).
- TT-Lang snapshot, [TT-Metal build integration](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/cmake/modules/BuildTTMetal.cmake)과 [pinned version](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/third-party/tt-metal-version).
- TT-Lang snapshot, [Compute Kernel Configuration design](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/docs/development/ComputeKernelConfiguration.md)과 [Pipe optimization alternatives](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/docs/development/PipeOptimizations.md).
- TT-Lang snapshot, [matmul benchmark 설명](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/benchmarks/matmul/README.md), [static planner](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/benchmarks/matmul/config.py), [benchmark sweep](https://github.com/tenstorrent/tt-lang/blob/6b2c56d48b317b53fc43a43dd66bc9144fd894d7/benchmarks/matmul/sweep.py).
