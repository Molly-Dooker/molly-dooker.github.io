---
title: "Tenstorrent TT 컴파일러 스택: TT-Forge부터 TT-Metal까지"
date: 2026-08-14 14:27:19 +0900
categories: [Tenstorrent, compiler]
tags: [tt-forge, tt-xla, tt-forge-onnx, tt-mlir, tt-metal]
description: "TT-XLA와 TT-Forge-ONNX가 TT-MLIR에서 합류하고 TTNN runtime과 TT-Metal로 이어지는 현재 Tenstorrent 컴파일러 스택을 소스 코드 기준으로 분석합니다."
render_with_liquid: false
---

## 문서 범위

- 전체 프로젝트 기준: `tt-forge` commit [`4208372bc007`](https://github.com/tenstorrent/tt-forge/tree/4208372bc007a3e839388f1d846057b1d926165a)
- PyTorch·JAX frontend 기준: `tt-xla` commit [`1c9f93e45db9`](https://github.com/tenstorrent/tt-xla/tree/1c9f93e45db98d81814e850a748f35a53db2a0ab)
- ONNX·TensorFlow·PaddlePaddle frontend 기준: `tt-forge-onnx` commit [`4bef980334fe`](https://github.com/tenstorrent/tt-forge-onnx/tree/4bef980334fe7bdcd741cb343ce2fc978d748758)
- 공통 compiler 기준: `tt-mlir` commit [`71046369d603`](https://github.com/tenstorrent/tt-mlir/tree/71046369d603b97fd6a8dd8b947ca8588ac2a74f)
- deprecated frontend 비교 기준: `tt-torch` commit [`f5e3955f8a0a`](https://github.com/tenstorrent/tt-torch/tree/f5e3955f8a0a5169d39a241e8ab689270a97a339)
- 하위 실행 계층 교차 확인: `tt-metal`, 이 저장소의 TT-Lang·program factory·LLK 문서
- 확인일: 2026-08-14

이 글은 Tenstorrent의 현행 model compiler가 framework graph를 어떤 중간 표현으로 바꾸고, 그 결과를 TT-NN과 TT-Metalium으로 어떻게 실행하는지 설명한다. 저장소 이름이 비슷한 `tt-forge`, `tt-forge-fe`, `tt-forge-onnx`의 관계와 deprecated된 경로도 함께 구분한다.

분석의 기준은 각 저장소의 README만이 아니라 실제 compile entry point, MLIR pass pipeline, FlatBuffer 생성 코드와 runtime operation handler다. Model별 지원 범위, 성능 수치, compiler 설치와 실제 hardware 실행은 다루지 않는다. TT-Forge 계열은 빠르게 변하므로 위 commit을 벗어난 revision에서는 dependency pin과 pipeline option을 다시 확인해야 한다.

## 먼저 보는 결론

현재 표준 model graph 경로는 둘이다.

1. PyTorch와 JAX는 `TT-XLA`를 거쳐 StableHLO로 들어간다.
2. ONNX, TensorFlow와 PaddlePaddle은 `TT-Forge-ONNX`가 TT-TVM으로 가져온 뒤 Forge graph를 거쳐 TTIR로 들어간다.

두 경로는 TT-MLIR의 TTIR에서 합류한다. 공통 경로는 `TTIR -> TTNN dialect -> .ttnn FlatBuffer -> TTNN backend runtime -> TT-NN -> TT-Metalium`이다. TT-Lang은 model graph 전체를 받는 또 하나의 frontend가 아니라, 사용자 정의 operation의 device kernel을 만드는 별도 DSL(Domain-Specific Language) 경로다.

```text
Framework frontends

PyTorch -> PyTorch/XLA --+
                         +--> TT-XLA / PJRT --> VHLO --> StableHLO -------+
JAX ------> JAX/XLA ----+                                               |
                                                                         |
ONNX / TensorFlow / PaddlePaddle                                         |
                 |                                                       |
                 v                                                       v
          +-----------------+          +------------- TT-MLIR -------------+
          | TT-Forge-ONNX   |--------->| TTIR -> TTNN dialect -> FlatBuffer |
          | TT-TVM importer |  TTIR    | StableHLO -> TTIR                  |
          | Forge Graph     |          +------------------+-----------------+
          +-----------------+                             |
                                                          v
                                                   .ttnn executable
                                                          |
                                                          v
                                                TTNN backend runtime
                                                          |
                                                          v
                                           TT-NN API / Program Factory
                                                          |
                                                          v
                                             TT-Metalium JIT / runtime
                                                          |
                                                          v
                                      Firmware / LLK / SFPI / Tensix hardware
```

이 그림에서 compile-time과 runtime의 경계는 `.ttnn` 부근이다. `.ttnn`은 최종 Tensix instruction binary가 아니라 TTNN operation과 tensor 정보를 직렬화한 실행 파일이다. Runtime이 이 operation을 실제 TT-NN C++ API 호출로 바꾸고, TT-Metalium이 필요한 device kernel을 JIT compile하고 dispatch한다.

## 비슷한 이름부터 구분한다

### TT-Forge는 전체 프로젝트의 이름이다

[`tt-forge` README](https://github.com/tenstorrent/tt-forge/blob/4208372bc007a3e839388f1d846057b1d926165a/README.md#tt-forge-sub-projects)는 TT-Forge를 TT-Metalium 위에 구축한 open-source AI compiler stack이라고 정의한다. 여기에는 frontend, TT-MLIR, TT-Lang, 학습 recipe와 model library가 함께 들어간다.

따라서 architecture에서 `TT-Forge`를 StableHLO와 TT-MLIR 사이에 있는 단일 compiler pass로 그리면 부정확하다. `tt-forge` 저장소는 전체 프로젝트의 문서, demo, 배포와 model 생태계를 묶는 중심점이다. 실제 framework graph 변환은 `tt-xla` 또는 `tt-forge-onnx`가 담당한다.

| 프로젝트 | 주된 역할 | 현행 위치 |
| --- | --- | --- |
| TT-Forge | 전체 compiler 생태계와 배포 단위 | Umbrella project |
| TT-XLA | PyTorch·JAX frontend와 PJRT plugin | 주 frontend, single·multi-chip |
| TT-Forge-ONNX | ONNX·TensorFlow·PaddlePaddle용 TVM 기반 frontend | Single-chip frontend |
| TT-MLIR | 공통 MLIR compiler와 runtime serialization | Middle/backend |
| TT-Lang | 사용자 정의 고성능 kernel용 Python DSL | Early-preview custom-kernel 경로 |
| TT-Blacksmith | 학습 recipe와 experiment | Compiler stage가 아님 |
| TT-Forge-Models | Model loader와 CI coverage | Compiler stage가 아님 |

### tt-forge-fe는 현재 tt-forge-onnx다

현재 GitHub의 [`tenstorrent/tt-forge-fe`](https://github.com/tenstorrent/tt-forge-fe) 주소는 `tenstorrent/tt-forge-onnx`로 이동한다. 별도의 현행 frontend 두 개가 병렬로 존재하는 것이 아니라 repository 이름이 바뀐 것이다. 다만 Python package와 source namespace에는 `forge.compile`, `ForgeModule`, `ForgeGraphModule`처럼 기존 이름이 남아 있다. 오래된 문서나 image의 `Forge FE`도 대체로 이 계보를 가리킨다.

### TT-Torch와 TT-BUDA는 현행 주 경로가 아니다

[`tt-torch` README](https://github.com/tenstorrent/tt-torch/blob/f5e3955f8a0a5169d39a241e8ab689270a97a339/README.md)는 프로젝트가 deprecated되었고 TT-XLA를 사용하라고 명시한다. TT-Torch는 PyTorch 2와 torch-mlir를 연결해 StableHLO를 TT-MLIR로 넘기던 이전 frontend다. 소스와 과거 문서를 읽을 때는 유용하지만 새 architecture의 PyTorch 진입점으로 두지 않는다.

[`tt-buda`](https://github.com/tenstorrent/tt-buda)는 archived된 초기 compiler/runtime stack이다. 현재 TT-XLA·TT-Forge-ONNX·TT-MLIR 경로와 하나의 직렬 pipeline으로 섞어 설명하지 않는 편이 정확하다.

## PyTorch와 JAX는 TT-XLA로 들어간다

TT-XLA는 OpenXLA의 PJRT device plugin interface를 구현한다. PyTorch와 JAX가 서로 다른 Python frontend를 쓰더라도 compiler와 runtime으로 들어가는 공통 접점은 `pjrt_plugin_tt.so`다.

### PyTorch 경로

TT-XLA는 [`@register_backend(name="tt")`](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/python_package/tt_torch/backend/backend.py#L825-L837)로 `torch.compile` backend를 등록한다. Backend는 Dynamo와 AOTAutograd가 만든 FX graph를 전처리하고 PyTorch/XLA bridge로 넘긴다. 현재 기본 경로는 [`extract_compiled_graph`](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/python_package/tt_torch/backend/backend.py#L598-L603)를 통해 XLA graph를 추출한다.

```text
PyTorch Module
      |
      v
torch.compile backend="tt"
      |
      v
Dynamo / AOTAutograd / FX preprocessing
      |
      v
PyTorch/XLA graph
      |
      v
TT PJRT plugin
```

### JAX 경로

JAX 쪽 Python package는 [`register_plugin("tt", ...)`](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/python_package/jax_plugin_tt/__init__.py#L20-L31)으로 같은 PJRT shared library를 등록한다. JAX의 `jit`과 XLA lowering 결과가 이 plugin으로 들어가므로, PyTorch와 JAX는 framework capture 단계는 달라도 TT-XLA의 native compiler 경로를 공유한다.

### PJRT가 받은 VHLO를 TTNN executable로 바꾼다

TT-XLA의 [`ModuleBuilder::buildModule`](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/pjrt_implementation/src/api/module_builder/module_builder.cc#L322-L526)은 다음 순서로 tt-mlir library의 pipeline을 조합한다.

1. PJRT가 넘긴 versioned HLO MLIR, 즉 VHLO(Versioned StableHLO)를 parse한다.
2. StableHLO deserialize pipeline으로 VHLO를 StableHLO로 되돌린다.
3. StableHLO와 Shardy 관련 frontend pass를 실행한다.
4. StableHLO를 TTIR로 변환한다.
5. 공통 TTIR-to-TTNN pipeline과 runtime용 tail pipeline을 실행한다.
6. 최종 TTNN module을 `.ttnn` FlatBuffer로 직렬화한다.

VHLO는 framework와 compiler 사이에서 StableHLO program의 version 호환성을 유지하기 위한 wire 형식이다. TT-XLA가 VHLO를 독자적인 최종 IR로 쓰는 것은 아니다. 실제 최적화와 lowering은 deserialize한 StableHLO, TTIR과 TTNN dialect에서 진행한다.

TT-XLA는 [`third_party/CMakeLists.txt`](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/third_party/CMakeLists.txt#L5-L8)에서 호환되는 TT-MLIR commit을 고정한다. TT-MLIR은 StableHLO와 runtime, TTNN backend, OpModel 지원을 켠 상태로 함께 build된다. TT-XLA는 단순히 StableHLO text를 외부 executable에 넘기는 wrapper가 아니라 compiler와 runtime library를 포함한 framework adapter다.

## ONNX 계열은 TT-Forge-ONNX로 들어간다

[`tt-forge-onnx` README](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/README.md#what-is-this-repo)는 이 frontend가 TT-TVM을 사용하며 ONNX, TensorFlow와 PaddlePaddle을 대상으로 한다고 설명한다. PyTorch 입력도 지원하지만, 현행 공식 권장은 PyTorch와 JAX에 TT-XLA를 사용하는 것이다. TT-Forge-ONNX의 공개 범위는 현재 single-chip이다.

실제 compile path는 다음과 같다.

```text
ONNX / TensorFlow / PaddlePaddle
                |
                v
         TT-TVM import
                |
                v
        Python ForgeModule
                |
                v
     Forge graph transformations
                |
                v
          ForgeGraphModule
                |
                v
        C++ MLIRGenerator
                |
                v
              TTIR
```

Python [`compile.py`](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/forge/forge/compile.py#L640-L648)는 framework module을 `convert_to_forge_module`로 변환한다. 이 함수는 TT-TVM과 연결된 `generate_forge_module`을 호출한다. 이후 graph pass, constant evaluation, autograd 관련 pass와 graph 분할을 거친 뒤 C++ `run_mlir_compiler`로 이동한다.

C++ [`MLIRGenerator::emit_mlir`](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/forge/csrc/passes/lower_to_mlir.cpp#L178-L218)는 `ForgeGraphModule`의 operation을 TTIR operation으로 직접 만든다. 따라서 이 경로는 StableHLO를 거치지 않는다. TTIR을 만든 뒤에는 TT-MLIR의 TTIR-to-TTNN pipeline을 실행하고 [`ttnnToFlatbuffer`](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/forge/csrc/passes/mlir_compiler.cpp#L123-L143)를 호출한다.

이 snapshot의 TT-Forge-ONNX는 pipeline 이름으로 `ttir-to-ttnn-backend-pipeline`을 사용한다. 이 frontend가 고정한 TT-MLIR은 해당 이름을 `ttir-to-ttnn-runtime-pipeline`의 [deprecated 호환 alias](https://github.com/tenstorrent/tt-mlir/blob/883ef57adf1ce4c6c4b33deacc5af7d1f09e5e1d/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp#L819-L834)로 등록한다. 이름은 다르지만 별도의 backend가 두 개 존재한다는 의미는 아니다.

## 두 frontend는 TTIR에서 합류한다

TT-MLIR은 MLIR(Multi-Level Intermediate Representation) 기반의 공통 compiler다. TT-XLA는 StableHLO를 넘기고, TT-Forge-ONNX는 이미 만든 TTIR을 넘긴다. 어떤 frontend에서 왔든 model graph가 TTIR에 도달한 뒤에는 같은 TTNN lowering 기반을 사용한다.

### Dialect의 역할

| Dialect | 추상화 수준 | 주된 역할 |
| --- | --- | --- |
| TTCore | 공통 device·layout·grid type | System description과 tensor 배치의 공통 기반 |
| TTIR | Framework와 runtime 사이의 tensor graph | Frontend가 합류하는 고수준 operation 표현 |
| TTNN | TT-NN API에 가까운 tensor operation | Layout, memory configuration, sharding과 runtime 호출 표현 |
| D2M | Direct-to-Metal lowering용 중간 표현 | 선택한 subgraph의 data movement와 compute kernel 구체화 |
| TTKernel | Device kernel API에 가까운 operation | TT-Metal kernel C++ 생성에 필요한 kernel 표현 |
| TTMetal | 낮은 수준의 host·device program 표현 | Direct TTMetal backend와 실험 경로의 기반 |

이 표를 위에서 아래로 항상 순서대로 통과하는 pipeline으로 읽으면 안 된다. 현행 model compiler의 기본 경로는 `TTIR -> TTNN`이다. D2M, TTKernel과 TTMetal은 custom kernel 또는 direct-to-metal 경로에서 사용한다.

### TTIR-to-TTNN 공통 pipeline

현재 [`createTTIRToTTNNCommonPipeline`](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp#L306-L532)은 대략 다음 일을 수행한다.

1. TTIR element type을 정규화하고 decomposition, canonicalization, fusion과 constant evaluation 준비 pass를 실행한다.
2. TTNN layout을 만들고 TTIR operation을 TTNN operation으로 변환한다.
3. TTNN composite resolution, decomposition, fusion과 hardware 제약 workaround를 적용한다.
4. OpModel을 사용할 수 있으면 operation 제약을 검사하면서 memory layout을 전파하고 L1 sharding과 spill을 조정한다.
5. 선택적으로 D2M subgraph를 만들고 custom kernel 경로로 낮춘다.
6. Layout conversion을 구체적인 operation으로 분해하고 deallocation을 삽입한다.
7. Runtime target에서는 CPU로 옮긴 부분을 LLVM IR로 낮추고 FlatBuffer 변환이 가능한 TTNN IR을 만든다.

여기서 optimizer가 선택하는 것은 단순한 operation 순서만이 아니다. Tensor를 interleaved 또는 sharded layout으로 둘지, L1과 DRAM 사이에서 어떻게 이동할지, operation이 지원하는 layout인지도 compile 결정에 포함된다. 그래서 TTIR-to-TTNN 구간이 framework-independent graph와 TT-Metal runtime 사이의 핵심 경계다.

## .ttnn은 최종 기계어가 아니다

TT-MLIR의 [FlatBuffer 문서](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/docs/src/flatbuffers.md)는 세 확장자를 구분한다.

| 확장자 | 의미 | 현재 상태 |
| --- | --- | --- |
| `.ttsys` | Target system description | Compiler에 target 정보를 제공 |
| `.ttnn` | TTNN backend runtime용 compiled binary | 현행 model path의 실행 형식 |
| `.ttb` | TTMetal backend runtime용 compiled binary | 문서상 unsupported |

여기서 compiled binary라는 표현을 Tensix machine code로 해석하면 안 된다. `.ttnn` FlatBuffer에는 program, input과 output tensor, TTNN operation, layout과 configuration이 들어간다. TT-MLIR runtime의 [`program_executor.cpp`](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/runtime/lib/ttnn/program_executor.cpp#L223-L264)는 program의 operation 목록을 순서대로 방문한다.

각 runtime handler는 실제 TT-NN C++ API를 부른다. 예를 들어 serialized matmul operation의 handler는 [`::ttnn::matmul`](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/runtime/lib/ttnn/operations/matmul/matmul.cpp#L67-L93)을 호출한다. 이 지점부터 실행 책임은 `tt-metal` 저장소에 들어 있는 TT-NN implementation과 TT-Metalium으로 넘어간다.

```text
.ttnn FlatBuffer
       |
       v
TT-MLIR TTNN runtime
       |
       v
TT-NN C++ operation
       |
       +--> Layout / memory handling
       |
       +--> Program Factory
                  |
                  v
          TT-Metal Program
                  |
                  +--> Host command dispatch --> UMD / KMD --> Firmware / Device
                  |
                  +--> Kernel C++ / LLK / SFPI --> JIT --> RISC-V binary --> Tensix
```

UMD(User Mode Driver)와 KMD(Kernel Mode Driver)는 host command가 device에 도달하는 제어 경로다. Device kernel 쪽에서는 TT-Metalium C++과 LLK(Low-Level Kernel)·SFPI(SFPU Interface) 호출을 compile하고, compute kernel을 unpack, math와 pack용 RISC-V binary로 나눈다. 이 binary가 runtime에 Tensix instruction을 각 compute thread로 보낸다.

이 저장소의 [TTNN matmul program factory 문서](/posts/tenstorrent-matmul-program-factories/)는 `::ttnn::matmul` 아래에서 operation 조건에 따라 program factory를 선택하고 Reader·Compute·Writer kernel을 구성하는 구간을 설명한다. [Tenstorrent LLK 기초](/posts/tenstorrent-llk-basics/)는 그보다 아래에서 compute kernel이 RISC-V와 Tensix engine으로 이어지는 과정을 다룬다.

## TT-Lang은 별도의 custom-kernel 경로다

Standalone TT-Lang은 제한된 Python AST(Abstract Syntax Tree)를 MLIR로 만들고 TT-Metal kernel C++을 생성한다.

```text
TT-Lang Python
      |
      v
TTL / TTCore MLIR
      |
      v
TTKernel MLIR
      |
      v
EmitC -> TT-Metal kernel C++
      |
      v
ttnn.generic_op
      |
      v
TT-Metalium JIT / runtime
```

따라서 TT-Lang을 TT-XLA나 TT-Forge-ONNX와 같은 model frontend로 놓으면 역할을 혼동한다. TT-XLA와 TT-Forge-ONNX는 model graph를 TTIR과 TTNN operation으로 바꾼다. TT-Lang은 사용자가 block, data movement와 grid를 설계한 custom operation을 device kernel로 바꾼다. Standalone compile 과정은 이 저장소의 `_drafts/tenstorrent-tt-lang-dsl-compiler-stack.md`에서 별도로 분석한다.

### TT-XLA model 안에 TT-Lang operation을 넣는 경로

최신 TT-XLA source에는 두 경로를 연결하는 초기 integration도 들어 있다. [`@tt_torch.tt_lang_operation`](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/python_package/tt_torch/tt_lang.py#L203-L231)으로 감싼 operation을 XLA tensor에 적용하면 `stablehlo.custom_call @tt.tt_lang_op`을 만든다.

```text
TT-Lang operation in PyTorch model
               |
               v
stablehlo.custom_call @tt.tt_lang_op
               |
               v
          ttir.tt_lang_op
               |
               v
          ttnn.tt_lang_op
               |
               v
      TT-Lang kernel resolver
               |
               v
    C++ sources + kernel metadata
               |
               v
       TTNN GenericOp FlatBuffer
               |
               v
          ttnn.generic_op
```

TT-XLA의 ModuleBuilder는 TTIR-to-TTNN 뒤에 [`TTNNResolveTTLangKernels`와 `TTNNLowerTTLangToGeneric`](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/pjrt_implementation/src/api/module_builder/module_builder.cc#L1139-L1161)을 실행한다. Resolver는 Python의 [`resolve_operation`](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/python_package/tt_torch/tt_lang.py#L947-L988)을 호출해 kernel C++ source, circular buffer, semaphore, core range와 runtime argument 정보를 artifact로 만든다. Runtime에서는 이를 `ttnn.generic_op`으로 실행하고 TT-Metalium JIT가 source를 device binary로 compile한다.

코드는 실제 lowering과 resolver를 구현하고 있지만 TT-Lang 자체가 early preview이고 integration 문서와 구현이 빠르게 바뀌고 있다. 따라서 이 경로를 모든 TT-XLA model이 통과하는 기본 단계가 아니라 선택적인 custom operation 기능으로 보는 편이 안전하다.

### D2M도 기본 직렬 단계가 아니다

TT-MLIR의 D2M(Direct-to-Metal)은 compiler가 선택한 TTNN subgraph를 kernel 수준으로 낮추는 별도 경로다. `enable-create-d2m-subgraphs`를 켜면 pipeline은 대상 subgraph를 만들고, [`TTNN -> TTIR -> D2M -> TTKernel -> TTNN GenericOp`](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp#L711-L726) 경로를 실행한 뒤 결과를 주변 TTNN graph에 다시 합친다.

TT-Lang과 D2M은 모두 최종적으로 custom kernel과 `GenericOp`을 사용할 수 있지만 출발점이 다르다. TT-Lang에서는 사용자가 Python DSL로 kernel 구조를 작성한다. D2M에서는 compiler가 기존 TTNN subgraph를 골라 kernel 형태로 낮춘다.

## Build와 runtime dependency는 version pin으로 연결된다

TT-Forge stack은 모든 저장소의 최신 `main`을 임의로 조합하는 구조가 아니다. 각 frontend가 TT-MLIR revision을 고정하고, TT-MLIR이 다시 TT-Metal revision을 고정한다.

2026-08-14에 확인한 dependency 관계는 다음과 같다.

```text
tt-xla 1c9f93e45db9
    |
    +--> tt-mlir 71046369d603 --+
                                  |
                                  +--> tt-metal 5beed318d0f0
                                  |
tt-forge-onnx 4bef980334fe        |
    |                             |
    +--> tt-mlir 883ef57adf1c ----+
    |
    +--> tt-tvm 40b41d685df0

tt-lang 6b2c56d48b31
    |
    +--> tt-metal v0.75.0 / d9a68815f5fc
```

TT-XLA의 CMake는 TT-MLIR `71046369d603`을 직접 지정한다. TT-Forge-ONNX는 git submodule로 [TT-MLIR `883ef57adf1c`](https://github.com/tenstorrent/tt-mlir/tree/883ef57adf1ce4c6c4b33deacc5af7d1f09e5e1d)와 [TT-TVM `40b41d685df0`](https://github.com/tenstorrent/tt-tvm/tree/40b41d685df0397182367aeb72b2448800ac3018)을 고정한다. 두 TT-MLIR revision의 [`third_party/CMakeLists.txt`](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/third_party/CMakeLists.txt#L1-L6)는 모두 TT-Metal `5beed318d0f0`을 가리킨다.

TT-Lang draft의 snapshot은 별도로 TT-Metal `v0.75.0`과 submodule commit `d9a68815f5fc`을 사용한다. TT-Lang이 만든 kernel artifact를 TT-XLA·TT-MLIR runtime에 넣으려면 생성 코드가 TT-MLIR이 고정한 TT-Metal API와 맞아야 한다. 분석이나 재현 환경에는 frontend commit 하나만 적지 말고 TT-MLIR과 TT-Metal pin까지 함께 남겨야 한다.

## 기존 문서와 연결되는 위치

현재 이 저장소의 Tenstorrent 문서는 compiler stack의 아래쪽을 비교적 자세히 다룬다.

| 기존 문서 | 전체 stack에서 담당하는 구간 |
| --- | --- |
| `_drafts/tenstorrent-tt-lang-dsl-compiler-stack.md` | TT-Lang Python부터 TTKernel, EmitC와 `ttnn.generic_op`까지 |
| [TTNN matmul program factory](/posts/tenstorrent-matmul-program-factories/) | TT-NN operation에서 TT-Metal program factory와 device kernel 구성까지 |
| [Tenstorrent LLK 기초](/posts/tenstorrent-llk-basics/) | TT-Metal compute kernel 아래의 RISC-V, LLK, SFPI와 Tensix engine |
| SDPA prefill·decode 문서 | 구체적인 TT-NN operation과 program factory의 dataflow |

이 글은 기존 문서에 없던 위쪽과 가운데를 채운다. 구체적으로 framework frontend 선택, StableHLO와 TTIR의 합류, TTIR-to-TTNN lowering, `.ttnn`과 runtime의 경계, TT-Lang·D2M 분기를 하나의 지도에 놓는다.

## 제약과 확인이 더 필요한 부분

- 이 분석은 위 commit의 source를 정적으로 대조한 결과다. 각 frontend를 build하거나 실제 model을 hardware에서 실행해 pipeline dump를 수집하지는 않았다.
- TT-XLA의 TT-Lang integration과 TT-MLIR의 D2M 경로는 변화가 빠르다. 기능 상태는 문서 표현보다 source의 pass 등록과 test를 함께 확인해야 한다.
- TT-Forge-ONNX는 snapshot에서 deprecated alias인 `ttir-to-ttnn-backend-pipeline`을 사용한다. TT-MLIR revision을 바꿀 때 alias 유지 여부를 확인해야 한다.
- `.ttnn` 안에 기록되는 operation과 metadata는 schema revision에 따라 바뀔 수 있다. FlatBuffer schema와 runtime을 서로 다른 TT-MLIR version으로 섞지 않아야 한다.
- TT-NN operation마다 program factory 구성과 JIT 시점이 같다고 가정하면 안 된다. Operation별 구현은 `tt-metal` source에서 따로 확인해야 한다.
- Official architecture diagram의 `TTIR -> TTKernel` 또는 `TTIR -> TTMetal` 화살표는 가능한 target이나 custom-kernel 분기를 나타낸다. 모든 model의 기본 실행 순서로 해석하지 않는다.

## 핵심 정리

- TT-Forge는 단일 compiler pass가 아니라 TT-XLA, TT-Forge-ONNX, TT-MLIR, TT-Lang과 주변 project를 묶는 전체 생태계다.
- 현재 PyTorch와 JAX의 주 frontend는 TT-XLA다. TT-Torch는 deprecated되었다.
- 기존 `tt-forge-fe` 저장소는 현재 TT-Forge-ONNX이며, TT-TVM을 사용해 ONNX·TensorFlow·PaddlePaddle model을 Forge graph로 가져온다.
- TT-XLA는 StableHLO를 TTIR로 낮추고, TT-Forge-ONNX는 Forge graph에서 TTIR을 직접 만든다. 두 경로는 TTIR에서 합류한다.
- 현행 공통 model path는 `TTIR -> TTNN dialect -> .ttnn FlatBuffer -> TTNN runtime -> TT-NN -> TT-Metalium`이다.
- `.ttnn`은 최종 Tensix machine code가 아니다. Runtime이 serialized TTNN operation을 실제 TT-NN API로 실행하며 TT-Metalium이 device program과 kernel을 준비한다.
- TTKernel과 TTMetal dialect는 모든 model이 차례로 거치는 필수 단계가 아니다. TT-Lang, D2M과 direct backend 같은 custom-kernel 분기에서 주로 나타난다.
- 호환성은 repository 이름보다 frontend가 고정한 TT-MLIR과 TT-Metal commit chain으로 확인해야 한다.

## 참고 자료

- Tenstorrent, [TT-Forge snapshot README와 sub-project 구성](https://github.com/tenstorrent/tt-forge/blob/4208372bc007a3e839388f1d846057b1d926165a/README.md).
- Tenstorrent, [TT-XLA snapshot README](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/README.md), [PyTorch backend](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/python_package/tt_torch/backend/backend.py), [JAX plugin](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/python_package/jax_plugin_tt/__init__.py).
- TT-XLA snapshot, [VHLO-to-StableHLO와 TTIR-to-TTNN ModuleBuilder](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/pjrt_implementation/src/api/module_builder/module_builder.cc), [FlatBuffer runtime submit](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/pjrt_implementation/src/api/flatbuffer_loaded_executable_instance.cc).
- Tenstorrent, [TT-Forge-ONNX snapshot README](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/README.md), [Python compile stages](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/forge/forge/compile.py).
- TT-Forge-ONNX snapshot, [Forge graph-to-TTIR lowering](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/forge/csrc/passes/lower_to_mlir.cpp), [TTIR-to-TTNN과 FlatBuffer 생성](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/forge/csrc/passes/mlir_compiler.cpp), [runtime submit](https://github.com/tenstorrent/tt-forge-onnx/blob/4bef980334fe7bdcd741cb343ce2fc978d748758/forge/csrc/runtime/runtime.cpp).
- Tenstorrent, [TT-MLIR snapshot README](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/README.md), [dialect overview](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/docs/src/dialects-overview.md).
- TT-MLIR snapshot, [TTIR-to-TTNN pipeline](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp), [FlatBuffer 형식](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/docs/src/flatbuffers.md), [TTNN program executor](https://github.com/tenstorrent/tt-mlir/blob/71046369d603b97fd6a8dd8b947ca8588ac2a74f/runtime/lib/ttnn/program_executor.cpp).
- TT-XLA snapshot, [TT-Lang operation integration](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/python_package/tt_torch/tt_lang.py)과 [TT-Lang lowering pass 연결](https://github.com/tenstorrent/tt-xla/blob/1c9f93e45db98d81814e850a748f35a53db2a0ab/pjrt_implementation/src/api/module_builder/module_builder.cc#L1139-L1161).
- Tenstorrent, [TT-Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/5beed318d0f0d1c6212e605947fb0be80c9e0a1d/METALIUM_GUIDE.md)와 [Getting Started software stack](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html).
- OpenXLA, [StableHLO](https://github.com/openxla/stablehlo)와 [PJRT C API overview](https://openxla.org/xla/pjrt).
- Tenstorrent, [deprecated TT-Torch](https://github.com/tenstorrent/tt-torch/blob/f5e3955f8a0a5169d39a241e8ab689270a97a339/README.md)와 [archived TT-BUDA](https://github.com/tenstorrent/tt-buda).
