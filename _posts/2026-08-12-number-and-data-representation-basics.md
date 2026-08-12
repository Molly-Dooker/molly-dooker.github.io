---
title: "숫자와 데이터 표현 기초"
date: 2026-08-12 17:13:30 +0900
categories: [Tenstorrent, llk]
tags: [llk]
description: "llk 구현을 읽는 데 필요한 bit pattern, 정수 표현, carry, ties-to-even, FP32와 BF16 구조를 설명합니다."
render_with_liquid: false
---

## 문서 범위

- 기준 구현: `tt-metal` commit [`69096826694c`](https://github.com/tenstorrent/tt-metal/tree/69096826694cac0e8bbde0050e38a3e411a6d91e)
- 기준 ISA: `tt-isa-documentation` commit [`b7738d9ac14a`](https://github.com/tenstorrent/tt-isa-documentation/tree/b7738d9ac14a34a4033d60dde9463466b23082e1)
- 관련 hardware 기초: [Tenstorrent LLK 기초](/posts/tenstorrent-llk-basics/)
- 확인일: 2026-08-12

이 글은 Tenstorrent llk 구현을 읽는 데 필요한 숫자와 bit 표현의 기초를 설명한다. 숫자와 bit pattern의 차이에서 시작해 bit 연산, carry, ties-to-even 반올림, signed integer 표현, FP32와 BF16 구조를 차례로 살펴본다.

## 1. 숫자와 bit pattern은 같은 개념이 아니다

컴퓨터가 register와 memory에 저장하는 것은 `0`과 `1`의 나열인 **bit pattern**이다. 같은 pattern도 어떤 data type으로 해석하느냐에 따라 전혀 다른 숫자가 된다.

예를 들어 같은 32bit pattern `0x80000000`은 다음처럼 해석될 수 있다.

| 해석 datatype | 의미 |
| --- | --- |
| `u32` | `2147483648` |
| two's-complement `i32` | `-2147483648`, 즉 `INT_MIN` |
| `fp32` | `-0.0` |
| sign-magnitude `i32` | sign bit만 1인 negative zero |

### Numeric cast와 bitcast

두 연산은 목적이 다르다.

- **Numeric cast**: 숫자 값을 가능한 한 유지하면서 destination format에 맞는 새 bit pattern을 만든다.
- **Bitcast**: bit pattern은 건드리지 않고 해석할 data type만 바꾼다.

예를 들어 정수 `1`을 `fp32`로 numeric cast하면 `1.0`이 되고 bit pattern도 바뀐다.

```text
int32 value     = 1
int32 bits      = 0x00000001

fp32 value      = 1.0
fp32 bits       = 0x3f800000
```

반면 `0x3f800000`을 `u32`와 `fp32`로 각각 bitcast해 읽으면 bit는 같지만 값은 다르다.

```text
same bits       = 0x3f800000
as u32          = 1065353216
as fp32         = 1.0
```

## 2. bit, byte, bit width

- **bit**: `0` 또는 `1` 하나
- **4bit**: 16진수 한 자리
- **8bit**: 1 byte
- **16bit**: 2 byte
- **32bit**: 4 byte

32bit 값의 bit 위치는 오른쪽부터 `0`에서 시작해 왼쪽 끝이 `31`이다.

```text
bit index     31                              0
              +-------------------------------+
32-bit value  | b31 b30 ... b2 b1 b0          |
              +-------------------------------+
```

bit width가 고정돼 있으면 범위도 정해진다.

| datatype | 일반적인 범위 |
| --- | ---: |
| `u16` | `0`부터 `2^16 - 1 = 65535` |
| `u32` | `0`부터 `2^32 - 1 = 4294967295` |
| two's-complement `i32` | `-2^31`부터 `2^31 - 1` |

## 3. `0x`는 16진수 표기다

`0x` prefix는 뒤의 숫자가 16진수임을 나타낸다. `x`는 곱하기 기호가 아니라 prefix의 일부다. 16진수 한 자리는 4bit와 정확히 대응하므로 긴 bit pattern을 간결하게 적을 수 있다.

```text
hex     binary
0x0     0000
0x1     0001
0x8     1000
0xf     1111
```

예를 들면 다음과 같다.

```text
0x8000      = 1000 0000 0000 0000
0xffff      = 1111 1111 1111 1111
0xffffffff  = 32 one-bits
```

`0x7fff_ffff`처럼 중간에 넣는 underscore는 사람이 자릿수를 읽기 쉽게 구분할 뿐 값에는 영향을 주지 않는다. 즉 `0x7fff_ffff`와 `0x7fffffff`는 같다.

## 4. `H`와 `L`은 상위·하위 절반을 부르는 이름이다

`H`와 `L`은 instruction이나 operator가 아니라, 값의 두 부분을 가리키려고 붙인 이름이다.

- `H`: High, 32bit 값의 상위 16bit
- `L`: Low, 32bit 값의 하위 16bit

```text
bit index      31             16 15               0
               +----------------+------------------+
32-bit value   | H: high 16    | L: low 16        |
               +----------------+------------------+
```

예를 들어 `0x3f818000`은 다음처럼 나뉜다.

```text
bits = 0x3f818000
H    = 0x3f81
L    = 0x8000
```

식으로 추출하면 다음과 같다.

```text
H = (bits >> 16) & 0xffff
L = bits & 0xffff
```

## 5. `>>`와 `<<`는 bit shift다

`x >> n`은 `x`의 bit를 오른쪽으로 `n`칸, `x << n`은 왼쪽으로 `n`칸 옮긴다.

```text
10110000 >> 4 = 00001011
00000001 << 4 = 00010000
```

32bit pattern을 16칸 오른쪽으로 옮기면 하위 16bit `L`이 사라지고 `H`가 오른쪽에 남는다.

```text
0x3f818000 >> 16 = 0x00003f81
```

Overflow가 없고 값을 unsigned로 해석한다면, 왼쪽 shift는 대략 `2^n`을 곱하는 연산이고 오른쪽 shift는 `2^n`으로 나눈 뒤 나머지를 버리는 연산이다.

다만 signed 값을 오른쪽으로 shift할 때 빈 왼쪽 bit를 `0`으로 채울지 sign bit로 채울지는 language와 instruction에 따라 다르다.

## 6. `&`와 `|`는 bitwise AND와 OR다

`&`와 `|`는 각 bit 위치를 서로 독립적으로 계산한다.

| `A` | `B` | `A & B` | `A \| B` |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 |

### `& 1`: 마지막 bit 하나 꺼내기

`1`의 bit pattern에서는 가장 오른쪽 bit만 `1`이다. 따라서 `x & 1`을 계산하면 `x`의 마지막 bit만 남는다.

```text
0100 & 0001 = 0000
0101 & 0001 = 0001
```

즉 결과는 항상 `0` 또는 `1`이다.

### `& mask`: 필요한 bit만 남기기

mask에서 `1`인 위치만 원래 bit를 보존한다.

```text
x                  = 0x92345678
x & 0x7fffffff     = 0x12345678
```

`0x7fffffff`는 bit 31만 `0`이므로 `x & 0x7fffffff`는 최상위 sign bit를 지운다.

### `| mask`: 특정 bit를 1로 설정하기

`OR`는 mask에서 `1`인 위치를 강제로 `1`로 바꾼다.

```text
x                  = 0x00000005
x | 0x80000000     = 0x80000005
```

이 식은 magnitude `5`를 유지하면서 bit 31을 sign bit로 설정한다.

`&`/`|`는 bitwise operator이고, C/C++의 logical operator `&&`/`||`와 다르다. 또한 여기서 `&`는 식의 두 operand 사이에 있으므로 C/C++ declaration의 reference나 단항 address-of operator가 아니다.

## 7. `LSB`와 `MSB`

- **LSB**(least significant bit): 가장 오른쪽 bit, bit 0
- **MSB**(most significant bit): 가장 왼쪽 bit
- 32bit 값에서는 bit 31이 MSB다.

```text
MSB                                      LSB
 |                                        |
 v                                        v
+------------------------------------------+
| b31 b30 ... b2 b1 b0                    |
+------------------------------------------+
```

ties-to-even에서 “even”은 보존되는 정수 또는 significand의 LSB가 `0`이라는 뜻이다.

```text
retained bits = ...0100  -> LSB=0 -> even
retained bits = ...0101  -> LSB=1 -> odd
```

따라서 `(bits >> 16) & 1`은 BF16으로 보존할 `H`의 LSB가 even인지 odd인지 검사한다.

## 8. ties-to-even의 의미와 bit 구현

### 먼저 어떤 반올림을 구현하려는가

정밀도가 높은 값을 더 적은 자릿수로 줄이면 일부 자릿수를 버려야 한다. 이때 버려지는 부분을 보고 가장 가까운 표현값을 선택하는 것이 round-to-nearest다.

십진수 정수로 반올림하는 예를 들면 다음과 같다.

| 입력 | 가까운 정수 | 결과 |
| ---: | --- | ---: |
| `3.4` | `3`이 더 가까움 | `3` |
| `3.7` | `4`가 더 가까움 | `4` |
| `6.2` | `6`이 더 가까움 | `6` |
| `6.8` | `7`이 더 가까움 | `7` |

이 값들은 어느 후보가 더 가까운지 명확하므로 별도 규칙이 필요하지 않다.

문제는 버려지는 부분이 정확히 `0.5`인 halfway다.

```text
3.5 is equally close to 3 and 4
6.5 is equally close to 6 and 7
```

halfway에서는 두 후보까지의 거리가 같으므로 “가장 가까운 값”만으로 결과를 결정할 수 없다. 이 동률을 처리하는 규칙에는 ties-away-from-zero, ties-toward-zero, ties-to-even 등 여러 방식이 있다.

| 입력 | 두 후보 | ties-to-even 결과 | 이유 |
| ---: | --- | ---: | --- |
| `2.5` | `2`, `3` | `2` | `2`가 even |
| `3.5` | `3`, `4` | `4` | `4`가 even |
| `6.5` | `6`, `7` | `6` | `6`이 even |
| `7.5` | `7`, `8` | `8` | `8`이 even |

따라서 ties-to-even은 “halfway이면 항상 올린다” 또는 “항상 내린다”는 규칙이 아니다.

```text
halfway + lower candidate even -> keep lower candidate
halfway + lower candidate odd  -> choose upper candidate
```

이 규칙은 halfway가 반복될 때 반올림 결과가 한쪽으로만 쏠리는 편향을 줄인다.

### 십진수 예제를 `H`와 `L`에 대응하기

이해를 위한 비유에서는 남길 정수 부분을 `H`, 버릴 소수 부분을 `L`처럼 생각할 수 있다.

```text
3.4 = [ 3 ][ .4 ]
        H     L

3.7 = [ 3 ][ .7 ]
        H     L

3.5 = [ 3 ][ .5 ]
        H     L
```

이는 실제 IEEE floating-point encoding이 아니라 각 부분의 역할을 보여 주기 위한 비유다.

- `H`: 최종 결과에 남길 부분이자 아래쪽 후보
- `H+1`: 위쪽 후보
- `L`: 버릴 부분이지만 어느 후보가 더 가까운지 알려 주는 나머지

따라서 `L` 자체를 반올림하는 것이 아니다. **`L`을 살펴본 뒤 최종 결과를 `H`로 둘지 `H+1`로 올릴지 정한다.**

```text
L < halfway  -> keep H
L > halfway  -> choose H + 1
L = halfway  -> inspect H parity
```

`3.5`와 `6.5`는 버릴 부분이 둘 다 `.5`다. `L`만 보면 두 경우를 구분할 수 없으므로, halfway에서는 남길 부분 `H`가 even인지 odd인지 추가로 확인해야 한다.

```text
3.5 -> H=3 odd  -> H+1=4
6.5 -> H=6 even -> keep H=6
```

그래서 버릴 부분 `L`을 검사하면서 `H`도 함께 봐야 한다. `L`은 두 후보까지의 거리와 halfway 여부를 알려 준다. 정확히 halfway라면 최종 `H`를 even으로 만들기 위해 `H`의 parity까지 확인한다.

### BF16에서 “even”이 뜻하는 것

위의 십진수 예제에서 even은 `2`, `4`, `6`처럼 정수의 짝수를 뜻한다. BF16 bit 연산에서는 같은 원리를 **최종적으로 보존되는 bit pattern의 LSB가 `0`인가**로 판단한다.

```text
H = ...0100 -> H_lsb=0 -> even
H = ...0101 -> H_lsb=1 -> odd
```

따라서 BF16의 ties-to-even은 다음처럼 동작한다.

```text
L < 0x8000 -> keep H
L > 0x8000 -> choose H + 1
L = 0x8000 -> H_lsb=0: keep H
              H_lsb=1: choose H + 1
```

여기까지가 구현하려는 반올림 규칙이다. 아래부터는 이 네 조건을 branch 없이 add, mask, shift만으로 구현하는 방법을 설명한다.

### 이제 carry로 반올림을 구현한다

**Carry는 한 자리의 표현 범위를 넘은 값을 바로 왼쪽 자리로 넘기는 동작**이다. 우리말로는 받아올림이다.

#### 십진수의 받아올림

십진수 한 자리는 `0`부터 `9`까지만 표현할 수 있다. `9+1=10`에서는 일의 자리에 `10`을 그대로 쓸 수 없으므로 `0`만 남기고 `1`을 십의 자리로 넘긴다.

```text
  9
+ 1
---
 10

ones result = 0
carry       = 1
```

여러 자릿수에서도 같은 일이 일어난다.

```text
  1299
+ 0001
------
  1300
```

두 자리씩 `H`와 `L`로 나누어 보면 `L=99`에 `1`을 더한 결과가 `100`이 되어, 앞의 `1`이 `H`로 넘어간다.

```text
  [ 12 ][ 99 ]
+ [ 00 ][ 01 ]
----------------
  [ 13 ][ 00 ]
     H      L
```

즉 `L`의 범위를 넘긴 부분이 사라지는 것이 아니라 `H`에 `1`로 더해진다.

#### 이진수의 carry

이진수 한 자리는 `0`과 `1`만 표현할 수 있다. 따라서 `1+1`의 결과 `2`는 한 bit에 들어가지 않고 이진수 `10`으로 표현된다.

```text
1 + 1 = 10

result bit = 0
carry      = 1
```

이전 자리에서 들어오는 carry-in이 없을 때 한 bit 덧셈의 기본 경우는 다음과 같다.

| `A` | `B` | 현재 자리 결과 | 왼쪽으로 보내는 carry |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 1 |

예를 들어 4bit가 모두 `1`인 값에 `1`을 더하면 carry가 연쇄적으로 왼쪽으로 전달된다.

```text
  1111
+ 0001
------
1 0000
^
carry-out
```

원래 4bit 공간에는 `0000`만 남고, 범위를 벗어난 왼쪽 `1`이 carry-out이다.

#### `0xffff + 1 = 0x1_0000`의 의미

`0xffff`는 16bit가 모두 `1`인 가장 큰 값이다.

```text
0xffff = 1111 1111 1111 1111
```

여기에 `1`을 더하면 모든 bit에서 carry가 이어져 17번째 bit가 `1`이 된다.

```text
  0xffff
+ 0x0001
--------
0x1_0000
```

underscore를 경계처럼 보면 `0x1_0000`의 왼쪽 `1`은 기존 16bit에 들어가지 못한 carry-out이고, 오른쪽 `0000`은 16bit 결과다.

```text
carry-out     = 0x1
low-16 result = 0x0000
```

#### 왜 `L`의 carry가 `H`를 바꾸는가

`H`와 `L`은 별개의 두 값이 아니다. 하나의 연속된 32bit pattern을 설명하기 편하도록 절반으로 나눈 것이다.

```text
bit index      31             16 15               0
               +----------------+------------------+
32-bit value   | H: high 16     | L: low 16        |
               +----------------+------------------+
```

`L`의 bit 15 바로 왼쪽인 bit 16은 `H`의 가장 오른쪽 bit다. 따라서 `L`에서 발생한 carry-out은 곧 `H`로 들어가는 carry-in이 된다.

```text
+----------------+------------------+
| H: high 16     | L: low 16        |
+----------------+------------------+
                 ^                  |
                 |    carry-out     |
                 +------------------+
```

예를 들어 다음 32bit 덧셈을 `H`와 `L`로 분리해 생각할 수 있다.

```text
  0x3f81 | 0x8000
+ 0x0000 | 0x8000
-------------------
  0x3f82 | 0x0000
```

먼저 `L`에서 덧셈이 일어난다.

```text
L:
0x8000 + 0x8000 = 0x1_0000

low-16 result = 0x0000
carry-out     = 1
```

그 carry-out이 `H`의 carry-in이 된다.

```text
H:
0x3f81 + carry-in 1 = 0x3f82
```

결과적으로 전체 값은 `0x3f82_0000`이 된다.

#### Carry와 overflow의 차이

두 용어는 관련이 있지만 기준으로 삼는 bit width가 다르다.

- **carry**: 한 자리나 하위 bit group에서 넘친 값을 바로 왼쪽 자리로 전달하는 동작
- **overflow**: 정해진 전체 destination bit width로 결과를 표현할 수 없는 상태

위 예제에서 `L`만 16bit 값으로 보면 `0x8000+0x8000`은 표현 범위를 넘는다. 하지만 실제 계산 대상은 `[H][L]`을 합친 32bit 값이다. 따라서 carry를 버리지 않고 `H`로 넘길 수 있다. `L`에서 생긴 overflow가 전체 32bit 덧셈에서는 carry가 되는 셈이다.

`fp32 -> bf16` 식은 `L`에서 이 carry를 의도적으로 만들거나 만들지 않음으로써 `H`를 유지할지 `H+1`로 올릴지를 결정한다.

### 여기서 `bias`가 뜻하는 것

이 절에서 말하는 `bias`는 floating-point exponent의 exponent bias와 다르다. 버릴 하위 16bit를 반올림하려고 전체 32bit pattern에 더하는 **rounding offset**이다.

`H_lsb`는 `H`의 가장 오른쪽 bit이므로 값이 `0` 또는 `1`뿐이다.

```text
H even -> H_lsb = 0
H odd  -> H_lsb = 1
```

따라서 다음 식에서 `bias`도 두 값 중 하나만 선택된다.

```text
bias = 0x7fff + H_lsb

H even -> bias = 0x7fff + 0 = 0x7fff
H odd  -> bias = 0x7fff + 1 = 0x8000
```

`0x7fff`에 `1`을 더하면 `0x8000`이 되는 과정은 다음과 같다.

```text
  0111 1111 1111 1111   = 0x7fff
+ 0000 0000 0000 0001   = 0x0001
----------------------
  1000 0000 0000 0000   = 0x8000
```

### 왜 halfway가 `L=0x8000`인가

버려지는 `L`의 범위는 `0x0000`부터 `0xffff`다. 인접한 두 BF16 bit pattern 후보 `H`와 `H+1` 사이의 정확한 절반 지점이 `L=0x8000`이다.

| `L`의 범위 | 위치 | 반올림 결과 |
| --- | --- | --- |
| `0x0000 <= L < 0x8000` | `H` 후보에 더 가까움 | `H` 유지 |
| `L = 0x8000` | 정확한 halfway | `H`의 even/odd에 따라 선택 |
| `0x8000 < L <= 0xffff` | `H+1` 후보에 더 가까움 | `H + 1` |

rounding offset을 더한 뒤 `L`의 결과가 `0xffff`를 넘으면 carry가 `H`로 넘어간다.

```text
+----------------+----------------+
| H: high 16     | L: low 16      |
+----------------+----------------+
        ^                 |
        |      carry      | L + bias > 0xffff
        +-----------------+
```

### Halfway에서 `H`가 even인 경우

`H=0x3f80`의 마지막 bit는 `0`이므로 `H_lsb=0`이다. 이때 `bias=0x7fff`다.

```text
bits                       = 0x3f80_8000
H                          = 0x3f80
L                          = 0x8000
H_lsb                      = 0
bias = 0x7fff + H_lsb      = 0x7fff

L + bias                   = 0x8000 + 0x7fff
                           = 0xffff
```

`0xffff`는 16bit `L` 안에 들어가므로 carry가 없다. 전체 32bit 덧셈에서도 `H`는 바뀌지 않는다.

```text
  0x3f80_8000
+ 0x0000_7fff
---------------
  0x3f80_ffff

carry                      = 0
result H                   = 0x3f80
```

마지막에 오른쪽으로 16bit shift하면 `L=0xffff`는 버려지고 even인 `H=0x3f80`이 그대로 남는다.

### Halfway에서 `H`가 odd인 경우

`H=0x3f81`의 마지막 bit는 `1`이므로 `H_lsb=1`이다. 이번에는 `bias=0x8000`이다.

```text
bits                       = 0x3f81_8000
H                          = 0x3f81
L                          = 0x8000
H_lsb                      = 1
bias = 0x7fff + H_lsb      = 0x8000

L + bias                   = 0x8000 + 0x8000
                           = 0x1_0000
```

`0x1_0000`은 17bit가 필요하므로 하위 16bit `L`에 들어가지 않는다. 맨 앞의 `1`이 carry가 되어 `H`에 더해진다.

```text
  0x3f81_8000
+ 0x0000_8000
---------------
  0x3f82_0000

carry                      = 1
result H                   = 0x3f81 + 1
                           = 0x3f82
```

odd였던 `0x3f81`에 1을 더하면 even인 `0x3f82`가 된다.

두 halfway 경로를 비교하면 다음과 같다.

| 기존 `H` | `H_lsb` | `bias` | `L + bias` | carry | 최종 `H` |
| --- | ---: | ---: | ---: | ---: | ---: |
| even `0x3f80` | 0 | `0x7fff` | `0xffff` | 0 | `0x3f80` 유지 |
| odd `0x3f81` | 1 | `0x8000` | `0x1_0000` | 1 | `0x3f82` |

### Halfway가 아닌 값도 같은 식으로 처리되는 이유

`H_lsb`에 따라 bias가 1만큼 달라진다. 하지만 이 차이가 최종 결과를 바꾸는 경우는 정확한 halfway뿐이다.

| `L` 조건 | 가장 경계에 있는 계산 | carry |
| --- | --- | ---: |
| `L < 0x8000` | `L_max 0x7fff + bias_max 0x8000 = 0xffff` | 항상 0 |
| `L = 0x8000`, `H` even | `0x8000 + 0x7fff = 0xffff` | 0 |
| `L = 0x8000`, `H` odd | `0x8000 + 0x8000 = 0x1_0000` | 1 |
| `L > 0x8000` | `L_min 0x8001 + bias_min 0x7fff = 0x1_0000` | 항상 1 |

첫 번째 행에서는 `L`이 가질 수 있는 가장 큰 값 `0x7fff`와 bias가 가질 수 있는 가장 큰 값 `0x8000`을 더해도 carry가 없다. 마지막 행에서는 `L`이 halfway보다 큰 가장 작은 값 `0x8001`에 가장 작은 bias `0x7fff`를 더해도 carry가 생긴다.

따라서 다음 한 식이 nearest rounding과 halfway의 ties-to-even을 모두 구현한다.

```text
bits += 0x7fff + ((bits >> 16) & 1)
bf16  = bits >> 16
```

이를 순서대로 풀면 다음과 같다.

```text
H_lsb = (bits >> 16) & 1
bias  = 0x7fff + H_lsb
bits  = bits + bias
bf16  = bits >> 16
```

1. `bits >> 16`으로 `H`를 오른쪽에 가져온다.
2. `& 1`로 `H`의 LSB만 추출한다.
3. `H`가 even이면 `0x7fff`, odd이면 `0x8000`을 더한다.
4. `L`의 덧셈에서 생긴 carry가 필요할 때만 `H`를 1 증가시킨다.
5. 마지막 `>> 16`으로 `L`을 버리고 반올림된 `H`만 남긴다.

## 9. two's-complement와 sign-magnitude

### Two's-complement

Two's-complement는 널리 쓰이는 signed integer 표현이다. 음수 `-x`는 양수 `x`의 모든 bit를 뒤집고 1을 더해 만든다.

```text
+5 in 32 bits              = 0x00000005
invert all bits            = 0xfffffffa
add 1                      = 0xfffffffb
-5 in two's-complement     = 0xfffffffb
```

32bit two's-complement의 범위는 `-2^31`부터 `2^31 - 1`이다. 음수 쪽에 값이 하나 더 있어서 `INT_MIN=-2^31`에 대응하는 양수 `+2^31`은 `i32`로 표현할 수 없다.

### Sign-magnitude

Sign-magnitude는 MSB 하나를 sign으로, 나머지 bit를 magnitude로 사용한다.

```text
+----------------+-------------------------------+
| sign: 1 bit    | magnitude: 31 bits            |
+----------------+-------------------------------+
```

```text
+5 in sign-magnitude       = 0x00000005
-5 in sign-magnitude       = 0x80000005
```

sign-magnitude에는 `+0=0x00000000`과 `-0=0x80000000`이 따로 존재한다. 32bit 범위는 `-(2^31-1)`부터 `+(2^31-1)`이므로 two's-complement의 `INT_MIN`과 정확히 대응하는 magnitude가 없다.

여기서 문제가 되는 부분은 `SFPCAST`가 integer source를 sign-magnitude로 읽는다는 점이다. Two's-complement의 `-5=0xfffffffb`를 그대로 넣으면 “sign=1, magnitude=0x7ffffffb”로 해석해 전혀 다른 값이 된다. 이를 피하려면 `SFPABS`, sign 복사, `INT_MIN` 보정이 필요하다.

## 10. `fp32`와 BF16 bit 구조

일반적인 normalized `fp32`는 sign 1bit, exponent 8bit, fraction 23bit로 구성된다.

```text
fp32
+------+----------+-------------------------+
| sign | exponent | fraction                |
| 1bit | 8bit     | 23bit                   |
+------+----------+-------------------------+
```

BF16은 sign과 exponent 폭은 같고 fraction을 7bit만 보존한다.

```text
bf16
+------+----------+---------+
| sign | exponent | fraction|
| 1bit | 8bit     | 7bit    |
+------+----------+---------+
```

BF16에 필요한 정보는 `fp32` bit pattern의 상위 16bit `H`에 들어 있다. 하위 16bit `L`을 그대로 버리면 truncation이다. `L`을 살펴보고 필요할 때 `H`를 1 늘리면 rounding이 된다.

- `Sign`: 부호 bit
- `Exp`: exponent field에서 bias를 제거한 exponent
- `Man`: 계산을 위해 implicit leading bit까지 포함한 mantissa

normalized 값은 개념적으로 다음 형태다.

```text
value = (-1)^Sign * (1.fraction) * 2^Exp
```

`1.fraction`의 앞쪽 `1`은 normalized floating-point encoding에 직접 저장하지 않아 implicit bit라고 부른다. zero, subnormal, infinity, NaN은 이 단순식의 예외 구현이며 여기서는 설명하지 않는다.

## 11. 자주 나오는 결과 처리 용어

| 용어 | 의미 | 예 |
| --- | --- | --- |
| truncate | 버려지는 자릿수를 반올림하지 않고 제거 | `3.9 -> 3`, `-3.9 -> -3` |
| round | 버려지는 부분을 보고 가까운 표현값을 선택 | `3.6 -> 4` |
| clamp | 값을 지정한 하한과 상한 안으로 제한 | `max(x, 0)` |
| saturate | 범위를 넘으면 destination datatype의 최솟값/최댓값에 고정 | `70000 -> u16 65535` |
| overflow | destination datatype으로 값을 표현할 수 없는 상태 | `2^32`를 `u32`로 변환 |
| sentinel | 특정 상태를 표시하기 위해 예약한 특별한 결과값 | overflow 시 `INT_MIN` |
| predication | lane별 조건에 따라 instruction 적용 여부를 제어 | 음수 lane에만 negate 적용 |