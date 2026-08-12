---
title: "MLIR와 XLA 컴파일러 구조: Hugging Face Llama에서 가속기까지"
date: 2026-08-21 11:14:47 +0900
categories: [DeepLearning, compiler]
tags: [mlir, xla, stablehlo, pjrt, pytorch-xla, llama]
description: "Hugging Face에서 불러온 Llama의 PyTorch 연산이 graph capture, HLO와 StableHLO, XLA 최적화, PJRT runtime을 거쳐 TPU와 GPU 가속기에서 실행되는 과정을 설명합니다."
render_with_liquid: false
---

## 문서 범위

- MLIR 기준: `llvm-project` commit [`bd439d56eca9`](https://github.com/llvm/llvm-project/tree/bd439d56eca9a2a95e6f3747fb25200dd39c7e6f/mlir)
- StableHLO 기준: `stablehlo` commit [`3c9301c116f9`](https://github.com/openxla/stablehlo/tree/3c9301c116f97fbaf696959a0ede32712aec587b)
- XLA·PJRT 기준: `xla` commit [`3abee12b628f`](https://github.com/openxla/xla/tree/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf)
- PyTorch 기준: `pytorch` commit [`98402d6df1bb`](https://github.com/pytorch/pytorch/tree/98402d6df1bb92ff02cf41bb4d239cb48faf78f9)
- PyTorch/XLA 기준: `xla` commit [`41398bfff334`](https://github.com/pytorch/xla/tree/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf)
- Llama 구현 기준: `transformers` commit [`945dac9117cb`](https://github.com/huggingface/transformers/tree/945dac9117cb54196888c0e6c08035792a98c485)
- 주 실행 환경: Hugging Face `AutoModelForCausalLM`으로 Llama 계열 checkpoint를 읽고 PyTorch/XLA와 PJRT로 TPU inference를 실행하는 경우
- 보조 비교: XLA:GPU와 일반 PyTorch CUDA 경로
- 확인일: 2026-08-21

이 글은 모델을 불러온 뒤 한 번의 `forward`가 가속기용 executable이 되기까지의 컴파일러·런타임 구조를 설명한다. 학습용 backward graph와 optimizer lowering, 특정 TPU 세대의 비공개 instruction encoding, 모델별 성능 수치는 다루지 않는다. 여러 가속기를 사용할 때 필요한 SPMD(Single Program, Multiple Data) 분할은 전체 경로를 이해하는 데 필요한 범위까지만 살펴본다.

가장 중요한 전제는 **PyTorch model이 항상 MLIR이나 XLA를 거치지는 않는다**는 점이다. 이 글의 본선은 tensor를 `xla` device로 옮기고 PyTorch/XLA를 선택한 경우다. 일반 CUDA tensor를 eager mode로 실행하거나 기본 `torch.compile` backend인 TorchInductor를 사용하면 다른 compiler stack을 지난다.

## 먼저 보는 결론

MLIR, StableHLO, XLA와 PJRT는 같은 대상을 가리키는 이름이 아니다.

| 구성 요소 | 역할 | 이 글에서의 위치 |
| --- | --- | --- |
| PyTorch·Transformers | Model 구조와 tensor 연산을 표현하는 frontend | Llama Python code와 ATen operation |
| MLIR | 여러 추상화 수준의 IR과 pass를 만드는 compiler infrastructure | StableHLO와 일부 backend IR의 기반 |
| StableHLO | Framework와 compiler 사이의 portable ML operation set | Version이 있는 공개 교환 형식 |
| HLO | XLA가 최적화에 사용하는 내부 graph IR | StableHLO import 뒤 또는 PyTorch/XLA 직접 경로에서 생성 |
| XLA | HLO graph를 분석·최적화하고 target executable을 만드는 compiler | Fusion, layout, sharding, scheduling, code generation |
| PJRT | Compiler와 device runtime을 framework에서 분리하는 device API | Compile, buffer transfer, loaded executable, execute |
| `libtpu`·GPU runtime | Target compiler와 device 제어 구현 | TPU 또는 GPU에 executable을 load하고 dispatch |
| TPU·GPU | 실제 tensor program을 실행하는 가속기 | HBM, matrix/vector unit, interconnect |

OpenXLA가 제시하는 일반적인 공개 경로는 다음과 같다. StableHLO는 MLIR dialect이지만, XLA 내부 HLO는 MLIR 기반 IR이 아니다. XLA는 StableHLO를 받으면 내부 HLO로 바꾼 뒤 target별 pipeline을 실행한다. 이 구분은 OpenXLA의 [용어 문서](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/terminology.md)에도 명시돼 있다.

```text
+-------------------+      +-------------------+      +-------------------+
| ML framework      |----->| StableHLO         |----->| XLA internal HLO  |
| frontend graph    |      | MLIR dialect      |      | optimizer IR      |
+-------------------+      +-------------------+      +---------+---------+
                                                               |
                                                               v
                                                     +-------------------+
                                                     | target backend    |
                                                     | code generation   |
                                                     +---------+---------+
                                                               |
                         +-------------------+                 v
                         | framework runtime |<------>| PJRT executable   |
                         | and device buffer |        | and device API    |
                         +-------------------+        +---------+---------+
                                                               |
                                                               v
                                                     +-------------------+
                                                     | accelerator       |
                                                     | TPU / GPU / CPU   |
                                                     +-------------------+
```

다만 **현재 PyTorch/XLA의 live 실행은 이 그림과 완전히 같은 직선 경로만 사용하지 않는다**. 기준 소스에서 `LoweringContext`는 Lazy IR node를 `xla::XlaOp`로 낮춰 `xla::XlaComputation`을 만들고, 기본적으로 이 HLO computation을 PJRT의 `CompileAndLoad`에 직접 넘긴다. `XLA_STABLEHLO_COMPILE`을 켠 선택 경로에서만 HLO를 StableHLO MLIR module로 바꿔 PJRT에 전달한다. 해당 분기는 [`pjrt_computation_client.cpp`](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/runtime/pjrt_computation_client.cpp#L638-L661)에서 확인할 수 있다.

따라서 “PyTorch/XLA로 실행하면 반드시 `Torch dialect -> StableHLO -> HLO` 순서를 눈에 보이는 단계로 모두 통과한다”라고 단정하면 부정확하다. StableHLO는 OpenXLA의 공개·portable 경계이고, PyTorch/XLA의 기존 bridge에는 HLO를 직접 넘기는 구현 경로도 남아 있다.

## MLIR은 compiler 하나가 아니라 IR infrastructure다

MLIR(Multi-Level Intermediate Representation)은 고수준 dataflow graph부터 target-specific code까지 여러 추상화 수준을 한 framework에서 표현하도록 설계됐다. [MLIR Language Reference](https://github.com/llvm/llvm-project/blob/bd439d56eca9a2a95e6f3747fb25200dd39c7e6f/mlir/docs/LangRef.md#high-level-structure)에 따르면 graph node에 해당하는 `Operation`, edge에 해당하는 SSA `Value`, 그리고 `Block`과 `Region`이 기본 구조를 이룬다.

MLIR의 핵심 확장 단위는 **dialect**다. Dialect는 operation, type과 attribute의 의미를 한 namespace 아래 정의한다. 예를 들어 같은 MLIR module 안에서도 다음처럼 서로 다른 수준을 표현할 수 있다.

- `stablehlo.dot_general`: framework와 compiler가 공유할 수 있는 tensor contraction
- `linalg.matmul`: structured linear algebra operation
- `scf.for`: structured control flow
- `gpu.launch`: GPU execution hierarchy
- `llvm.call`: LLVM IR에 가까운 호출
- `tt.*`, `iree.*` 같은 외부 project dialect: 특정 compiler나 accelerator의 의미

Compiler는 rewrite와 conversion pass로 dialect 내부를 최적화하거나 다른 dialect로 낮춘다. 이때 **lowering**은 같은 program 의미를 더 구체적인 추상화로 바꾸는 과정이고, **optimization**은 같은 추상화 수준에서도 계산량·memory traffic·launch 수를 줄이는 변환이다. 실제 pipeline에서는 둘이 섞여 실행될 수 있다.

MLIR 자체가 Llama를 TPU binary로 만드는 독립 compiler는 아니다. StableHLO, Triton과 여러 accelerator compiler가 MLIR의 IR 구조와 pass infrastructure를 사용한다. XLA도 StableHLO 입구와 일부 GPU code generation에서 MLIR을 활용하지만, 중심 optimizer IR인 HLO까지 MLIR이라고 부르지는 않는다.

## StableHLO, VHLO와 HLO를 구분한다

### StableHLO는 portable operation contract다

[StableHLO specification](https://github.com/openxla/stablehlo/blob/3c9301c116f97fbaf696959a0ede32712aec587b/docs/spec.md)은 StableHLO를 ML model의 high-level operation set으로 정의한다. `add`, `broadcast_in_dim`, `dot_general`, `gather`, `reduce`, `while`처럼 tensor program의 의미를 표현하지만, 특정 TPU의 tile 크기나 GPU thread block까지 정하지는 않는다.

StableHLO program에는 operation뿐 아니라 tensor shape, element type, reduction region과 dimension attribute가 들어간다. 예를 들어 Llama의 projection은 source에서 `nn.Linear`이지만, portable IR에서는 contracting dimension이 명시된 `stablehlo.dot_general`과 필요하면 `stablehlo.add`로 표현할 수 있다. 이 단계는 “무엇을 계산하는가”를 보존하고 “어느 MXU 또는 GPU block에서 어떻게 계산하는가”는 backend에 남긴다.

### VHLO는 serialization 호환 계층이다

VHLO(Versioned StableHLO)는 StableHLO program을 version별로 직렬화하기 위한 dialect다. StableHLO의 [compatibility 문서](https://github.com/openxla/stablehlo/blob/3c9301c116f97fbaf696959a0ede32712aec587b/docs/compatibility.md)는 일반 MLIR text가 아니라 정해진 API로 만든 portable artifact에 호환성 보장을 적용한다. 사람이 읽는 `.mlir` text와 장기 저장용 portable bytecode를 같은 대상으로 보면 안 된다.

### HLO는 XLA 내부 optimizer IR이다

HLO(High Level Optimizer)는 XLA compiler가 직접 사용하는 graph representation이다. StableHLO와 operation 이름·의미가 닮았지만 자체 C++ object model, text format과 protobuf 표현을 사용한다. XLA는 StableHLO를 import하면 곧바로 내부 HLO로 바꾸고, 수백 개의 analysis와 rewrite pass를 HLO 위에서 실행한다.

StableHLO가 framework와 compiler 사이의 비교적 안정적인 계약이라면 HLO는 XLA implementation과 함께 변하는 내부 작업 형식에 가깝다. Model 배포 형식으로 StableHLO를 선택할 수는 있지만, 임의의 optimized HLO dump를 장기 호환 artifact로 간주하면 안 된다.

## Llama를 load하는 일과 compile하는 일은 다르다

`AutoModelForCausalLM.from_pretrained()`가 하는 핵심 작업은 config로 Python module을 만들고 checkpoint의 parameter를 읽어 `state_dict`에 채우는 일이다. 이 호출만으로 XLA가 실행되거나 accelerator executable이 생기지는 않는다.

고정한 Transformers source에서 `LlamaForCausalLM`은 `LlamaModel`과 vocabulary projection인 `lm_head`를 가진다. `LlamaModel`은 embedding, decoder layer 목록, 마지막 RMSNorm과 rotary embedding을 구성한다. 실제 구조는 [`modeling_llama.py`](https://github.com/huggingface/transformers/blob/945dac9117cb54196888c0e6c08035792a98c485/src/transformers/models/llama/modeling_llama.py#L284-L357)에서 확인할 수 있다.

```text
token text
    |
    v
+-------------------+
| tokenizer on CPU  |
+---------+---------+
          |
          v
      input_ids
          |
          v
+-------------------+
| token embedding   |
+---------+---------+
          |
          v
+-------------------+    repeat N layers
| decoder layer     |<-------------------+
| RMSNorm           |                    |
| self attention    |                    |
| residual add      |                    |
| RMSNorm + MLP     |--------------------+
+---------+---------+
          |
          v
+-------------------+
| final RMSNorm     |
+---------+---------+
          |
          v
+-------------------+
| LM head           |
+---------+---------+
          |
          v
       logits
```

Tokenizer는 문자열을 token ID로 바꾸는 host-side 전처리다. 일반적으로 이 문자열 처리까지 XLA graph에 넣지 않는다. Compiler가 보는 입력은 `input_ids`, `attention_mask`, `position_ids`, KV cache와 model parameter 같은 tensor다.

`model.to(xla_device)`는 parameter와 buffer를 XLA device에 놓도록 요청한다. PyTorch/XLA는 lazy execution을 사용하므로 실제 transfer와 materialization 시점은 동기화와 graph 실행에 따라 지연될 수 있다. 이 이동 역시 Llama 계산 graph를 최적화하는 compile 자체와는 다르다.

다음 코드는 컴파일 단위를 분명히 보이기 위한 최소 구조다. 실제 실행에는 서로 호환되는 `torch`·`torch_xla` wheel, TPU runtime, 모델 접근 권한과 충분한 장치 메모리가 필요하다. `use_cache=False`를 사용한 이유는 먼저 단일 `forward`만 떼어 보기 위해서다.

```python
import os

os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch
import torch_xla
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "<llama-checkpoint>"
device = torch_xla.device()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
).eval().to(device)

batch = tokenizer(
    "MLIR and XLA",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128,
)
batch = {name: tensor.to(device) for name, tensor in batch.items()}

compiled_model = torch.compile(model, backend="openxla")

with torch.inference_mode():
    logits = compiled_model(
        **batch,
        use_cache=False,
        return_dict=False,
    )[0]

torch_xla.sync(wait=True)
```

`torch.compile()` 호출 자체가 모든 compile을 즉시 끝낸다고 해석하면 안 된다. 일반적으로 첫 입력을 넣을 때 Dynamo guard와 FX graph가 확정되고, PyTorch/XLA가 graph를 lowering하며, PJRT compile이 일어난다. 같은 graph와 호환되는 shape가 다시 들어오면 cache한 executable을 재사용한다.

## PyTorch graph를 잡는 두 경로

PyTorch/XLA에서는 전통적인 LazyTensor tracing과 `torch.compile(..., backend="openxla")` 경로를 구분해야 한다. 두 경로 모두 아래에서는 PyTorch/XLA lowering과 XLA compiler를 사용하지만, Python program에서 compile할 graph를 정하는 방법이 다르다.

### LazyTensor 경로

XLA tensor에 PyTorch operation을 적용하면 dispatcher가 XLA backend 구현을 선택한다. Operation은 즉시 TPU kernel 하나를 launch하는 대신 XLATensor가 가리키는 Lazy IR node를 만든다. 필요한 값에 도달할 때까지 node를 누적하므로 여러 PyTorch operation을 하나의 graph로 볼 수 있다.

`torch_xla.sync()`, device tensor의 CPU 복사, 값을 요구하는 일부 연산 같은 materialization 지점이 graph compile·execution을 촉발한다. PyTorch/XLA 공식 [overview](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/learn/xla-overview.md#pytorchxla-overview)는 tracing, HLO compile, cache와 execution을 이 순서로 설명한다.

```text
PyTorch op on XLA tensor
          |
          v
XLATensor Lazy IR node
          |
          +----> more PyTorch ops ----+
          |                           |
          +<--------------------------+
          |
          v
sync or materialization barrier
          |
          v
lower + compile + execute
```

이 방식은 Python control flow를 실제로 실행하면서 tensor operation만 기록한다. 따라서 Llama의 `for decoder_layer in self.layers`는 기본적으로 layer 수만큼 펼쳐진 큰 graph가 된다. PyTorch/XLA의 [`scan_layers`](https://docs.pytorch.org/xla/master/features/scan.html)는 동형 decoder layer의 body를 XLA `while`로 표현해 compile할 graph 크기를 줄이는 선택지지만, 지원 operation과 attention kernel 제약을 먼저 확인해야 한다.

### `torch.compile`의 OpenXLA backend

`torch.compile`에서는 TorchDynamo가 Python bytecode를 관찰해 tensor program을 FX graph로 만든다. Shape, dtype, Python constant와 module state에 guard를 붙이고, guard가 유지되는 호출은 같은 compiled result를 재사용한다.

PyTorch의 현재 [`openxla` backend 등록 코드](https://github.com/pytorch/pytorch/blob/98402d6df1bb92ff02cf41bb4d239cb48faf78f9/torch/_dynamo/backends/torchxla.py#L29-L55)는 AOTAutograd wrapper를 사용하고 첫 호출에서 PyTorch/XLA의 `extract_compiled_graph`를 부른다. PyTorch/XLA bridge는 이 FX graph를 XLA tensor로 실행해 기존 LazyTensor lowering으로 compile한다. 즉 이 경로는 `FX -> 독립된 Torch MLIR dialect -> StableHLO`로 고정된 pipeline이 아니라 `FX -> PyTorch/XLA Lazy IR -> XLA computation` 경로다.

```text
Python Llama module
        |
        v
TorchDynamo guards + FX graph
        |
        v
AOTAutograd graph wrapper
        |
        v
PyTorch/XLA graph extraction
        |
        v
XLATensor Lazy IR lowering
        |
        v
XLA computation + PJRT
```

지원되지 않는 operation이 있으면 bridge는 지원 가능한 FX partition과 CPU fallback partition을 나눌 수 있다. 기준 소스의 [`partition_fx_graph_for_cpu_fallback`](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/_dynamo/dynamo_bridge.py#L689-L747)이 이 역할을 한다. Fallback은 정확성을 유지하는 데 도움이 되지만 device-host transfer와 graph 분할을 일으켜 성능 비용이 따른다.

## XLATensor IR에서 XLA computation까지

PyTorch/XLA의 `LoweringContext`는 필요한 output에서 Lazy IR graph의 post-order를 구한다. 각 XLA node의 `CheckedLower()`가 하나 이상의 `xla::XlaOp`를 만들고, builder는 output을 묶어 `xla::XlaComputation`으로 완성한다. 실제 반복은 [`LoweringContext::LowerNode`](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/lowering_context.cpp#L260-L299)와 [`BuildXla`](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/lowering_context.cpp#L214-L247)에 나타난다.

이 변환에서 model parameter와 runtime input은 computation parameter가 될 수 있고, 작은 scalar나 compile 시점에 결정된 값은 constant로 들어갈 수 있다. `from_pretrained()`로 읽은 모든 가중치가 무조건 machine code 안에 상수로 박히는 것은 아니다. 일반 실행에서는 executable과 별도의 device buffer로 관리하는 것이 중요하다. Portable export에서도 weight를 parameter로 둘지 constant로 inline할지는 export 방식과 option에 따라 달라진다.

### Llama operation은 어떤 HLO가 되는가

다음 표는 Llama source와 대표적인 HLO·StableHLO operation의 대응을 보여 준다. 실제 graph dump는 Transformers의 attention backend, decomposition, dtype, shape, XLA version과 custom kernel 사용 여부에 따라 달라진다. 이 표를 정확한 일대일 치환 규칙으로 읽으면 안 된다.

| Llama 계산 | 대표적인 graph 표현 | Backend가 노리는 실행 형태 |
| --- | --- | --- |
| Token embedding | `gather` | Embedding 가중치에서 행 읽기 |
| Q/K/V와 output projection | `dot_general`, `add` | Matrix multiplication과 bias fusion |
| RMSNorm | `convert`, `multiply`, `reduce`, `add`, `rsqrt` | Reduction과 elementwise fusion |
| RoPE | `slice`, `negate`, `concatenate`, `multiply`, `add` | Vector elementwise fusion |
| Attention score | `dot_general`, scale, mask `add` | Batched matrix multiplication |
| Softmax | `reduce` max, `subtract`, `exponential`, `reduce` sum, `divide` | Fused softmax 또는 target kernel |
| Attention value product | `dot_general` | Batched matrix multiplication |
| SwiGLU MLP | 두 projection, activation, `multiply`, down projection | GEMM과 elementwise fusion |
| Residual connection | `add` | Producer·consumer fusion 후보 |
| KV cache update | `dynamic_update_slice`, `scatter` 또는 custom operation | 기존 cache buffer 갱신 |
| LM head | `dot_general` | Hidden state와 vocabulary weight 곱 |

예를 들어 [`LlamaRMSNorm.forward`](https://github.com/huggingface/transformers/blob/945dac9117cb54196888c0e6c08035792a98c485/src/transformers/models/llama/modeling_llama.py#L52-L67)는 FP32 변환, 제곱, 마지막 축 평균, epsilon 덧셈, `rsqrt`, 곱셈과 원래 dtype 변환을 차례로 쓴다. Source에서는 여러 PyTorch operation이지만 XLA는 전체 producer-consumer graph를 보고 중간 tensor를 HBM에 쓰지 않는 fusion을 만들 수 있다.

Attention도 source의 함수 이름 하나가 accelerator instruction 하나로 바뀌는 구조가 아니다. Eager attention을 쓰면 QK matmul, mask, softmax와 PV matmul이 여러 graph operation으로 드러난다. `scaled_dot_product_attention`이나 custom kernel 경로를 선택하면 frontend가 decomposition, `stablehlo.composite` 또는 `custom_call` 같은 경계를 남길 수도 있다. 어떤 경로든 최종 지원 여부와 fusion은 target backend가 결정한다.

## StableHLO는 live 실행과 portable export에서 위치가 다르다

StableHLO를 이해할 때 다음 두 목적을 분리해야 한다.

1. **Compiler input**: Framework가 StableHLO module을 PJRT compiler에 넘겨 현재 target executable을 만든다.
2. **Portable export**: Framework와 별도로 저장·전달할 StableHLO artifact를 만들고, 나중에 XLA나 다른 consumer가 compile한다.

PyTorch/XLA의 현재 live default는 앞에서 본 것처럼 HLO `XlaComputation`을 PJRT에 직접 넘긴다. 반면 PyTorch/XLA의 [StableHLO export 문서](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/features/stablehlo.md#torch-export-to-stablehlo)는 `torch.export`의 `ExportedProgram`을 `torchax`로 변환해 StableHLO MLIR을 얻는 별도 흐름을 설명한다.

```text
Live execution
PyTorch/XLA Lazy IR -> XlaComputation / HLO -> PJRT CompileAndLoad

Portable export
PyTorch -> ExportedProgram / FX -> torchax -> StableHLO artifact
                                              |
                                              v
                                      XLA or other compiler
```

Portable StableHLO는 target-independent tensor semantics를 보존하지만 다음 항목까지 보장하지는 않는다.

- 특정 TPU 또는 GPU의 최종 machine code
- 같은 fusion과 tile 선택
- 같은 memory layout과 buffer address
- 서로 다른 backend에서 bitwise-identical한 부동소수점 결과
- Python tokenizer, `generate()`의 모든 host control flow와 model metadata

즉 StableHLO bytecode는 “가속기가 바로 실행하는 binary”가 아니라 “호환되는 compiler가 읽을 program”이다.

## XLA compiler 안에서 일어나는 일

[XLA architecture 문서](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/architecture.md#how-it-works)는 StableHLO import, target-independent HLO 최적화, target-specific HLO 최적화와 code generation을 큰 단계로 나눈다. 실제 XLA에는 수백 개의 HLO pass가 있으며 target과 option에 따라 순서와 적용 여부가 달라진다.

다음 그림은 고정된 pass 순서가 아니라 각 구간의 책임을 요약한 것이다.

```text
StableHLO module or HLO computation
                 |
                 v
+-----------------------------------+
| import, verify, shape information |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
| HLO graph optimization            |
| simplify, CSE, fold, rewrite      |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
| partition and target optimization |
| sharding, fusion, library match   |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
| physical execution plan           |
| layout, schedule, buffer assign   |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
| target codegen and executable     |
+-----------------+-----------------+
                  |
                  v
             PJRT runtime
```

### Graph 정규화와 단순화

Algebraic simplification, constant folding, CSE(Common Subexpression Elimination), dead code elimination과 operation decomposition이 graph를 정리한다. 예를 들어 reshape 연쇄를 지우거나 compile-time constant 계산을 미리 끝내고, 같은 표현식을 한 번만 계산하도록 합칠 수 있다. XLA의 [HLO pass 문서](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/hlo_passes.md)는 `AlgebraicSimplifier`, rematerialization 등 대표 pass의 목적을 설명한다.

### Fusion은 operation 수보다 memory traffic을 줄인다

Llama에는 RMSNorm, RoPE, activation, mask와 residual처럼 elementwise·reduction operation이 많다. 이 operation을 각각 launch하고 매번 HBM에 intermediate tensor를 쓰면 계산보다 memory traffic과 launch overhead가 더 커질 수 있다.

Fusion은 producer와 consumer를 한 kernel 또는 한 target program으로 묶어 intermediate를 register나 on-chip memory에 유지한다. 다만 모든 이웃 operation을 합치는 것은 아니다. Register pressure, 중복 계산, layout, 여러 consumer, library call 경계와 target별 cost model이 fusion 가능성과 이득을 제한한다.

### Layout과 tiling은 logical shape 아래의 문제다

PyTorch에서 `hidden_states`가 `batch x sequence x hidden`이라는 사실만으로 physical memory order와 tile shape가 정해지지는 않는다. Backend는 matrix unit이 선호하는 block, contiguous dimension, alignment와 memory hierarchy를 고려해 layout을 선택한다. 필요하면 physical transpose나 copy를 삽입한다.

Tiling은 큰 `dot_general`을 가속기의 matrix unit과 local memory에 맞는 block으로 나눈다. TPU 문서는 XLA가 matrix multiplication을 MXU에 맞는 작은 block으로 변환한다고 설명한다. 이 단계가 같은 StableHLO `dot_general`을 TPU 세대와 shape별로 다른 executable로 만드는 이유다.

### Scheduling과 buffer assignment

Graph 전체를 알고 있으면 XLA는 value의 마지막 사용 시점을 분석해 같은 memory 영역을 다시 쓸 수 있다. Live range가 겹치지 않는 intermediate buffer를 재사용하고, memory pressure가 크면 일부 값을 다시 계산하는 rematerialization을 선택할 수 있다.

Scheduling은 compute, copy와 collective의 순서를 정한다. 여러 device에서는 통신과 계산을 겹치려는 schedule도 중요하다. 최종 executable에는 단순한 kernel 목록뿐 아니라 buffer, launch, dependency와 collective 실행 계획이 함께 필요하다.

### Backend code generation은 target마다 다르다

TPU backend는 HLO를 TPU executable로 낮추며, 실제 compiler와 runtime은 `libtpu`가 제공하는 PJRT implementation에 포함된다. 공개 문서만으로 모든 TPU instruction과 scheduling detail을 확인할 수는 없으므로, 이 글에서는 PJRT가 반환하는 loaded executable 아래를 세대 독립적인 개념 수준으로 설명한다.

XLA:GPU는 공개된 경로가 더 구체적이다. [XLA:GPU architecture](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/gpu_architecture.md)는 optimized library call, TritonIR code generation과 LLVM 기반 native emitter를 함께 사용한다고 설명한다. Matmul·softmax fusion은 TritonIR로 낮아질 수 있고, 다른 operation은 MLIR emitter를 거쳐 LLVM IR와 PTX가 되거나 cuBLAS·cuDNN·NCCL 호출로 바뀔 수 있다.

따라서 “XLA의 모든 backend가 `StableHLO -> Linalg -> LLVM IR` 순서로 내려간다”는 하나의 고정 그림도 정확하지 않다. 공통 HLO 최적화 뒤의 IR과 library 경계는 target backend가 선택한다.

## PJRT가 compiler와 실행을 잇는다

PJRT는 framework가 device별 compiler와 runtime을 같은 방식으로 다루게 하는 API다. 이름에 runtime이 들어가지만 실행만 담당하지 않는다. Client와 topology 조회, host-to-device buffer 생성, compile·load, executable 실행과 비동기 completion을 함께 제공한다.

[PJRT C++ API overview](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/pjrt/cpp_api_overview.md)는 다음 객체를 구분한다.

| PJRT 객체 | 역할 |
| --- | --- |
| `PjRtClient` | Device, memory space, compile과 transfer의 진입점 |
| `PjRtDevice` | 특정 device와 addressability 표현 |
| `PjRtBuffer` | Plugin이 관리하는 device-side data handle |
| `PjRtExecutable` | 직렬화할 수 있는 compiled artifact |
| `PjRtLoadedExecutable` | 현재 client에서 즉시 실행할 수 있게 load된 executable |
| `PjRtFuture` | 비동기 transfer·execution 완료 상태 |

PyTorch/XLA의 compile 함수는 device assignment와 SPMD option을 만든 뒤 `PjRtClient::CompileAndLoad`를 호출한다. 실행할 때는 input `PjRtBuffer`를 모아 loaded executable의 `ExecuteSharded` 또는 여러 device용 `Execute`를 부르고, 결과도 새 `PjRtBuffer` handle로 받는다. 단일 device 경로는 [`ExecuteComputation`](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/runtime/pjrt_computation_client.cpp#L737-L808)에서 확인할 수 있다.

이 구조 덕분에 PyTorch frontend는 TPU compiler 내부와 driver call을 직접 알 필요가 없다. 반대로 accelerator vendor는 PJRT plugin 안에서 compile, buffer와 execute contract를 구현할 수 있다. PyTorch/XLA의 [custom hardware plugin 문서](https://docs.pytorch.org/xla/master/contribute/plugins.html)는 plugin이 compiler와 runtime을 가진 `PjRtClient`를 제공해야 한다고 설명한다.

## TPU에서는 무엇이 실제로 실행되는가

TPU를 기준으로 host와 device의 책임을 나누면 다음과 같다.

```text
Host CPU
+---------------------------------------------------------------+
| tokenizer | Python | FX/Lazy tracing | XLA compile request     |
| PJRT client | input transfer | launch | output synchronization |
+-------------------------------+-------------------------------+
                                |
                                v
TPU device
+---------------------------------------------------------------+
| HBM buffers                                                   |
|   |                                                           |
|   +--> matrix program --> MXU                                 |
|   +--> vector program --> vector unit                         |
|   +--> control/address --> scalar unit                        |
|                                                               |
| on-chip data movement, synchronization, collectives           |
+---------------------------------------------------------------+
```

Google Cloud의 [TPU architecture 문서](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)는 TensorCore가 MXU(Matrix Multiplication Unit), vector unit과 scalar unit으로 구성된다고 설명한다. Llama의 큰 projection과 attention matmul은 주로 MXU 활용 대상이고, activation·normalization·softmax 같은 계산에는 vector 연산이 중요하다. Scalar unit은 control flow와 address 계산을 맡는다.

이 대응도 source operation 하나를 hardware unit 하나에 고정하는 표는 아니다. Fusion된 program은 matrix 연산 전후의 vector 계산과 data movement를 함께 포함할 수 있다. Backend는 HBM traffic을 줄이도록 tile과 schedule을 정하고, PJRT runtime은 compile된 program과 buffer를 device에 dispatch한다.

Output tensor는 실행이 끝나자마자 반드시 host memory로 돌아오지 않는다. `PjRtBuffer`로 device에 남아 다음 graph의 입력이 될 수 있다. `.cpu()`, Python `print`, `.item()`처럼 실제 host 값이 필요한 순간에 transfer와 synchronization이 생긴다. Autoregressive loop에서 불필요한 host read를 피해야 하는 이유다.

## `generate()`는 보통 graph 하나가 아니다

`LlamaForCausalLM.forward()` 한 번과 `model.generate()` 전체를 같은 compile 단위로 보면 inference 동작을 놓치게 된다. Autoregressive generation은 먼저 prompt 전체를 처리하는 prefill을 실행하고, 이후 새 token을 하나씩 만드는 decode를 반복한다.

```text
prompt tokens
      |
      v
+-------------------+
| prefill forward   |
| build KV cache    |
+---------+---------+
          |
          v
     next-token logits
          |
          v
+-------------------+
| token selection   |
+---------+---------+
          |
          v
new token + KV cache
          |
          v
+-------------------+
| decode forward    |-----> updated KV cache
+---------+---------+              |
          |                        |
          v                        |
     next-token logits             |
          |                        |
          +---- repeat ------------+
```

Prefill과 decode는 shape와 workload가 다르다.

- Prefill은 query length가 prompt length라서 큰 attention과 matrix multiplication을 수행한다.
- Decode는 보통 새 query token 하나와 지금까지의 KV cache를 읽는다.
- Token selection이 Python 또는 CPU에서 일어나면 매 token마다 device-host synchronization이 생길 수 있다.
- Python의 stopping condition과 sampling logic은 별도로 graph에 넣지 않는 한 host control flow로 남는다.

Transformers의 현재 Llama 소스는 `use_cache=True`인 호출에서 cache가 없으면 `DynamicCache`를 만든다. 각 attention layer는 이 cache의 `past_key_values.update()`를 부른다. 관련 코드는 [`LlamaModel.forward`](https://github.com/huggingface/transformers/blob/945dac9117cb54196888c0e6c08035792a98c485/src/transformers/models/llama/modeling_llama.py#L367-L417)와 [`LlamaAttention.forward`](https://github.com/huggingface/transformers/blob/945dac9117cb54196888c0e6c08035792a98c485/src/transformers/models/llama/modeling_llama.py#L243-L281)에 있다.

### KV cache shape가 compile cache에 영향을 준다

Dynamic cache는 token이 늘 때마다 sequence dimension도 커질 수 있다. XLA executable cache는 graph뿐 아니라 shape와 compile environment의 영향을 받으므로 새 shape가 계속 나타나면 compile variant가 늘어난다. PyTorch/XLA의 [recompilation guide](https://docs.pytorch.org/xla/release/r2.6/perf/recompilation.html)는 padding과 bucketization으로 shape 종류를 제한하는 이유를 설명한다.

Hugging Face의 [cache 전략 문서](https://huggingface.co/docs/transformers/kv_cache)는 `DynamicCache`와 최대 길이를 미리 할당하는 `StaticCache`를 구분한다. Static cache는 decode graph shape를 고정하기 쉽지만 사용하지 않는 cache 영역까지 memory를 차지하고 mask해야 한다. 따라서 다음 trade-off가 생긴다.

| 전략 | Compile 관점 | Memory·계산 관점 |
| --- | --- | --- |
| Dynamic cache | 길이 변화가 graph 또는 shape variant를 늘릴 수 있음 | 현재 길이에 맞춰 증가 |
| Static cache | 고정 decode graph를 재사용하기 쉬움 | 최대 길이만큼 선할당, 불필요한 영역 mask |
| Length bucket | 몇 개 executable로 입력 길이를 묶음 | Bucket 안의 padding overhead |

일반 Transformers가 Static cache와 함께 자동으로 적용하는 기본 `torch.compile` 동작을 OpenXLA path와 같다고 가정해서는 안 된다. XLA를 쓰려면 compile할 prefill·decode 함수를 분명히 나누고 `backend="openxla"`를 지정한 뒤, 해당 Transformers·PyTorch/XLA version에서 cache update가 지원되는지 확인해야 한다.

### Decoder layer 수와 token 수는 서로 다른 반복이다

Llama에는 두 종류의 반복이 있다.

1. 한 `forward` 안에서 decoder layer를 순회하는 model-depth loop
2. `generate()`가 새 token을 만들며 `forward`를 반복하는 token loop

첫 번째는 기본 tracing에서 layer 수만큼 graph에 펼쳐질 수 있고 `scan_layers` 같은 compiler-visible loop로 바꿀 여지가 있다. 두 번째는 cache, sampling과 종료 조건을 carried state로 가진다. 전체 generation을 하나의 `while` graph로 만들려면 이 state와 control flow까지 tensor program으로 다시 표현해야 한다. 단순히 `torch.compile(model.forward)`만 호출해서 두 반복이 모두 하나의 device loop가 되지는 않는다.

## 여러 가속기에서는 sharding이 추가된다

한 chip에 들어가지 않는 Llama는 model parameter, activation과 KV cache를 여러 device에 나눠야 한다. `.to("xla")`만으로 적절한 tensor parallelism이 자동 완성되지는 않는다. PyTorch/XLA SPMD에서는 logical device mesh와 tensor별 partition spec을 주고, XLA가 sharding annotation을 사용해 partitioned program과 collective를 만든다.

```text
global Llama graph + sharding annotations
                  |
                  v
+----------------------------------------+
| XLA SPMD partitioner                   |
| per-device slices + required transfers |
+--------------------+-------------------+
                     |
          +----------+----------+
          |                     |
          v                     v
+-------------------+  +-------------------+
| device 0 program  |  | device 1 program  |
| local buffers     |  | local buffers     |
+---------+---------+  +---------+---------+
          |      all-reduce / all-gather   |
          +<------------------------------>+
```

예를 들어 projection weight를 hidden dimension에 따라 나누면 각 device가 local matmul을 계산한 뒤 결과를 `all-reduce`하거나 `all-gather`해야 할 수 있다. 어느 축을 나누고 어디에 collective를 넣을지는 weight와 activation sharding 조합에 따라 달라진다.

PyTorch/XLA의 [SPMD user guide](https://docs.pytorch.org/xla/master/spmd.html)는 `xr.use_spmd()`, `Mesh`와 `mark_sharding()`으로 annotation을 붙이는 흐름을 설명한다. XLA compiler는 이를 바탕으로 single-device program을 partition하고, PJRT는 topology와 device assignment에 맞춰 shard buffer와 executable을 실행한다.

Compiler가 통신을 삽입한다고 해서 sharding 선택이 공짜가 되는 것은 아니다. Llama에서는 다음을 함께 봐야 한다.

- Parameter가 각 device memory에 들어가는가
- Q/K/V head와 hidden dimension을 어느 mesh axis에 나누는가
- Layer마다 activation reshard가 생기는가
- KV cache가 batch, head와 sequence 중 어느 축으로 분산되는가
- ICI와 host 간 network에서 collective가 병목이 되는가
- Prefill과 decode에 같은 sharding이 모두 적합한가

## 일반 PyTorch CUDA 경로와 혼동하지 않는다

`torch.compile`이라는 API 이름만으로 XLA 사용 여부가 정해지지 않는다. PyTorch의 [API 문서](https://docs.pytorch.org/docs/stable/generated/torch.compile)는 기본 backend가 `inductor`라고 명시한다.

| 코드와 device | 대표 경로 |
| --- | --- |
| CUDA tensor, compile 없음 | PyTorch dispatcher -> ATen CUDA·library kernel -> GPU |
| CUDA tensor, `torch.compile(model)` | Dynamo -> AOTAutograd -> TorchInductor -> Triton·C++/LLVM -> GPU |
| XLA tensor, lazy mode | PyTorch/XLA Lazy IR -> HLO -> PJRT -> XLA device |
| XLA tensor, `backend="openxla"` | Dynamo·FX -> PyTorch/XLA bridge -> HLO -> PJRT -> XLA device |
| Export용 StableHLO | `torch.export`·torchax -> StableHLO artifact -> consumer compiler |

XLA:GPU를 사용하면 XLA가 NVIDIA GPU용 PTX와 library call을 만든다. 그러나 이것은 일반 CUDA tensor에서 기본 TorchInductor를 쓰는 경로와 다른 선택이다. 따라서 profiler에서 Triton kernel을 봤다는 사실만으로 XLA를 사용했다고 판단할 수도 없다. TorchInductor와 XLA:GPU 모두 서로 다른 pipeline에서 Triton을 사용할 수 있다.

## 어디를 관찰하면 되는가

Compiler stack을 이해하려면 Python source만 읽지 말고 각 경계의 artifact와 metric을 확인해야 한다.

### PyTorch와 graph capture

- `torch._dynamo.explain()` 또는 `TORCH_LOGS`로 graph break와 guard를 확인한다.
- PyTorch/XLA metric에서 `CachedCompile`, `UncachedCompile`, `CompileTime`과 CPU fallback `aten::` counter를 확인한다.
- 첫 호출의 compile 시간과 warm execution 시간을 분리해 측정한다.

### HLO dump

XLA 공식 [HLO dump 문서](https://openxla.org/xla/hlo_dumps)는 `XLA_FLAGS`로 compile 전후 HLO와 backend artifact를 저장하는 방법을 설명한다.

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" \
PJRT_DEVICE=TPU \
python run_llama.py
```

Dump에서는 다음을 비교한다.

1. 같은 prompt bucket에서 graph hash와 executable을 재사용하는가
2. Q/K/V projection과 RMSNorm 주변에 어떤 fusion이 생겼는가
3. Prefill과 decode가 별도 module로 compile됐는가
4. 불필요한 `copy`, transpose와 layout conversion이 있는가
5. SPMD에서 collective와 reshard가 어디에 들어갔는가

### StableHLO export

Portable boundary를 확인하려면 live execution dump와 별도로 `torch.export`·torchax StableHLO export를 사용한다. 여기서 얻은 MLIR은 XLA optimization 뒤 HLO dump와 같은 artifact가 아니다. StableHLO에는 framework-independent semantics가 남고, optimized HLO에는 target과 pass 결과가 더 많이 반영된다.

### Device profile

최종 성능은 IR 모양만으로 확정할 수 없다. Device profile에서 matrix unit utilization, HBM bandwidth, collective, input transfer, compile·launch 사이의 idle gap을 확인해야 한다. 특히 decode는 작은 query 때문에 prefill보다 matrix unit utilization이 낮고 memory·launch latency에 민감할 수 있다.

## 자주 생기는 오해

### `from_pretrained()`가 model을 compile한다

아니다. Module을 구성하고 parameter를 읽는다. Graph capture는 tensor를 target device에 놓고 `forward`를 실행할 때 시작한다.

### MLIR과 XLA는 같은 compiler다

아니다. MLIR은 extensible IR·pass infrastructure이고 XLA는 ML compiler다. StableHLO는 MLIR dialect이며, XLA 내부 HLO는 MLIR IR이 아니다.

### StableHLO는 accelerator binary다

아니다. StableHLO는 portable tensor program이다. XLA backend의 layout, fusion, scheduling, buffer assignment와 code generation이 더 남아 있다.

### PyTorch/XLA는 항상 StableHLO text를 거친다

아니다. 현재 source의 기본 live path는 XLA builder가 만든 HLO computation을 PJRT에 직접 넘긴다. StableHLO compile과 portable export는 구분해야 한다.

### `torch.compile()`이면 XLA를 사용한다

아니다. 기본 backend는 TorchInductor다. PyTorch/XLA 경로에는 XLA tensor와 `backend="openxla"` 같은 명시적인 선택이 필요하다.

### Llama 전체가 executable 하나가 된다

항상 그렇지 않다. Graph break, CPU fallback, prefill·decode shape 차이, cache 전략, sampling과 host control flow가 여러 executable과 synchronization을 만들 수 있다.

### Fusion은 많을수록 항상 좋다

아니다. 지나친 fusion은 compile 시간, register pressure, code size와 중복 계산을 늘릴 수 있다. Backend cost model과 profiler 결과로 판단해야 한다.

### 여러 TPU를 보이면 model이 자동 분산된다

아니다. Replication, data parallelism과 model sharding은 서로 다르다. Mesh와 sharding annotation 또는 이를 제공하는 distributed API가 필요하다.

## 제약과 가정

- 이 글은 PyTorch/XLA source snapshot의 LazyTensor와 `openxla` bridge를 기준으로 한다. PyTorch/XLA, TorchTPU, torchax와 OpenXLA integration은 빠르게 변하므로 다른 revision에서는 graph capture와 StableHLO 경계를 다시 확인해야 한다.
- Hugging Face Llama는 `_attn_implementation`, cache class와 Transformers version에 따라 다른 operation graph를 만든다.
- TPU backend의 최종 code generation 세부는 공개된 XLA:GPU pipeline만큼 모두 드러나지 않는다. 공개 source에서 확인할 수 없는 instruction 수준 동작을 추정하지 않았다.
- 예제는 compile 경계를 설명하기 위한 구조다. 특정 Llama checkpoint가 해당 PyTorch/XLA revision에서 모든 operation을 fallback 없이 실행한다고 보장하지 않는다.
- Shape, dtype, device topology, compiler flag와 sharding이 달라지면 별도 executable이 생길 수 있다.
- Quantization, custom Pallas kernel, speculative decoding, continuous batching과 serving scheduler는 별도 주제다.

## 핵심 정리

- Hugging Face의 `from_pretrained()`는 Llama module과 가중치를 준비할 뿐 compile하지 않는다.
- PyTorch/XLA는 XLA tensor operation을 Lazy IR로 기록한다. `torch.compile(..., backend="openxla")`는 Dynamo·FX 앞단을 추가한 뒤 기존 LazyTensor lowering을 이용한다.
- MLIR은 multi-level IR infrastructure다. StableHLO는 MLIR 기반 portable operation set이고, HLO는 MLIR이 아닌 XLA 내부 optimizer IR이다.
- 현재 PyTorch/XLA live default는 `XlaComputation` HLO를 PJRT에 직접 compile하도록 넘길 수 있다. StableHLO compile과 StableHLO export는 별도 경로다.
- XLA는 HLO에서 graph simplification, fusion, sharding, layout, scheduling, buffer assignment와 target code generation을 수행한다.
- PJRT는 compiler와 runtime의 공통 device API다. `PjRtBuffer`와 `PjRtLoadedExecutable`을 사용해 transfer, load와 execute를 연결한다.
- TPU에서는 compile된 program이 HBM의 tensor를 읽어 MXU, vector unit과 scalar unit을 사용한다. StableHLO operation이 그대로 hardware instruction으로 실행되는 것은 아니다.
- Llama generation은 대개 prefill graph와 반복 decode graph로 나뉜다. KV cache와 input shape를 고정하거나 bucket으로 제한해야 compile cache를 안정적으로 재사용할 수 있다.
- 여러 가속기에서는 sharding annotation, SPMD partitioning과 collective가 pipeline에 추가된다. `.to("xla")`만으로 model parallelism이 완성되지는 않는다.

## 참고 자료

- LLVM Project, [MLIR Language Reference](https://github.com/llvm/llvm-project/blob/bd439d56eca9a2a95e6f3747fb25200dd39c7e6f/mlir/docs/LangRef.md).
- OpenXLA, [StableHLO Specification](https://github.com/openxla/stablehlo/blob/3c9301c116f97fbaf696959a0ede32712aec587b/docs/spec.md).
- OpenXLA, [StableHLO Compatibility](https://github.com/openxla/stablehlo/blob/3c9301c116f97fbaf696959a0ede32712aec587b/docs/compatibility.md).
- OpenXLA, [XLA Terminology](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/terminology.md).
- OpenXLA, [XLA Architecture](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/architecture.md).
- OpenXLA, [XLA:GPU Architecture](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/gpu_architecture.md).
- OpenXLA, [From HLO to Thunks](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/hlo_to_thunks.md).
- OpenXLA, [PJRT C++ Device API Overview](https://github.com/openxla/xla/blob/3abee12b628f5ba4fcb8d8f849e3dda68b801ecf/docs/pjrt/cpp_api_overview.md).
- PyTorch, [`openxla` TorchDynamo backend](https://github.com/pytorch/pytorch/blob/98402d6df1bb92ff02cf41bb4d239cb48faf78f9/torch/_dynamo/backends/torchxla.py).
- PyTorch/XLA, [Overview](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/learn/xla-overview.md).
- PyTorch/XLA, [TorchDynamo Integration](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/perf/dynamo.md).
- PyTorch/XLA, [`LoweringContext`](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/lowering_context.cpp).
- PyTorch/XLA, [PJRT computation client](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/runtime/pjrt_computation_client.cpp).
- PyTorch/XLA, [SPMD User Guide](https://docs.pytorch.org/xla/master/spmd.html).
- PyTorch/XLA, [`scan` and `scan_layers`](https://docs.pytorch.org/xla/master/features/scan.html).
- Hugging Face Transformers, [Llama implementation](https://github.com/huggingface/transformers/blob/945dac9117cb54196888c0e6c08035792a98c485/src/transformers/models/llama/modeling_llama.py).
- Hugging Face Transformers, [Cache strategies](https://huggingface.co/docs/transformers/kv_cache).
- Google Cloud, [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm).
