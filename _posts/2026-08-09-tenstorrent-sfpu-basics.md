---
title: "Tenstorrent SFPU 기초"
date: 2026-08-09 22:50:08 +0900
categories: [Tenstorrent, SFPU]
tags: [tenstorrent, sfpu, sfpi, blackhole, tensix, hardware]
description: "Tenstorrent Blackhole SFPU의 숫자 표현, register·tile mapping, SFPI, pipeline scheduling 기초를 정리합니다."
render_with_liquid: false
---

## 문서 범위

- 대상: SFPU 코드를 읽고 작성할 때 반복해서 필요한 숫자 표현, register, tile/vector mapping, SFPI, scheduling 기초
- hardware 설명 기준: Blackhole의 일반적인 full-tile elementwise 경로
- 관련 분석: Tenstorrent Typecast 최적화 정리(별도 문서)
- Typecast 원문: Jason Davies, [Typecast on Tenstorrent](https://www.jasondavies.com/2025/tenstorrent-typecast/), 2025-12-23
- 확인일: 2026-08-07

이 문서는 Typecast에만 한정되지 않는 SFPU 공통 선수 지식을 정리한다. 숫자와 bit pattern의 차이부터 SFPU의 Dst-to-LReg dataflow, 32-lane mapping, SFPI programming interface, pipeline scheduling까지 다룬다. 아래에서 “원문”은 별도 표시가 없는 한 위 Typecast 글을 가리키며, 구체적인 변환 알고리즘과 성능 분석은 별도 Typecast 분석 문서에 둔다.

## 숫자와 데이터 표현 기초

### 1. 숫자와 bit pattern은 같은 개념이 아니다

컴퓨터는 register와 memory에 `0`과 `1`의 나열인 **bit pattern**을 저장한다. 그 pattern을 어떤 datatype으로 해석하느냐에 따라 숫자의 의미가 달라진다.

예를 들어 같은 32bit pattern `0x80000000`은 다음처럼 해석될 수 있다.

| 해석 datatype | 의미 |
|---|---|
| `u32` | `2147483648` |
| two's-complement `i32` | `-2147483648`, 즉 `INT_MIN` |
| `fp32` | `-0.0` |
| sign-magnitude `i32` | sign bit만 1인 negative zero |

이 차이 때문에 typecast 구현에서는 “어떤 bit가 들어 있는가”와 “hardware가 그 bit를 어떤 datatype으로 해석하는가”를 항상 함께 확인해야 한다.

#### Numeric cast와 bitcast

두 연산은 목적이 다르다.

- **Numeric cast**: 숫자 값을 가능한 한 보존하면서 destination format에 맞는 새 bit pattern을 만든다.
- **Bitcast**: bit pattern은 그대로 두고 해석하는 datatype만 바꾼다.

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

원문의 `fp32 -> bf16` 알고리즘은 `fp32` 값을 `u32` bit pattern처럼 보고 정수 add와 shift를 수행한 뒤, 결과를 BF16 pattern으로 사용한다.

### 2. bit, byte, bit width

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
|---|---:|
| `u16` | `0`부터 `2^16 - 1 = 65535` |
| `u32` | `0`부터 `2^32 - 1 = 4294967295` |
| two's-complement `i32` | `-2^31`부터 `2^31 - 1` |

### 3. `0x`는 16진수 표기다

`0x` prefix는 뒤의 숫자가 16진수임을 나타낸다. 여기서 `x`는 곱하기 기호가 아니라 prefix의 일부다. 16진수 한 자리는 4bit와 정확히 대응하므로 긴 bit pattern을 짧게 쓰기 좋다.

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

### 4. `H`와 `L`은 상위·하위 절반을 부르는 이름이다

`H`와 `L`은 instruction이나 operator가 아니라 설명을 위한 약자다.

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

### 5. `>>`와 `<<`는 bit shift다

`x >> n`은 `x`의 bit를 오른쪽으로 `n`칸 옮기고, `x << n`은 왼쪽으로 `n`칸 옮긴다.

```text
10110000 >> 4 = 00001011
00000001 << 4 = 00010000
```

32bit pattern을 16칸 오른쪽으로 옮기면 하위 16bit `L`이 사라지고 `H`가 오른쪽에 남는다.

```text
0x3f818000 >> 16 = 0x00003f81
```

overflow가 없고 unsigned 값으로 해석할 때, 왼쪽 shift는 대략 `2^n`을 곱하고 오른쪽 shift는 `2^n`으로 나누어 소수 부분을 버리는 것과 같다.

다만 signed 값의 오른쪽 shift가 빈 왼쪽 bit를 `0`으로 채우는지 sign bit로 채우는지는 language와 instruction에 따라 다를 수 있다. 이 문서의 식은 일반 C++ signed shift를 뜻한다고 가정하지 않고, 원문에 명시된 SFPU instruction semantics를 기준으로 읽어야 한다.

원문의 `Man << (Exp - 23)`도 고정된 왼쪽 shift만을 뜻하지 않는다. `Exp - 23`이 양수이면 왼쪽으로, 음수이면 오른쪽으로 움직이는 hardware variable shift를 간단히 표현한 것이다.

### 6. `&`와 `|`는 bitwise AND와 OR다

`&`와 `|`는 각 bit 위치를 서로 독립적으로 계산한다.

| `A` | `B` | `A & B` | `A \| B` |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 |

#### `& 1`: 마지막 bit 하나 꺼내기

`1`의 bit pattern은 가장 오른쪽 bit만 `1`이다. 따라서 `x & 1`은 `x`의 가장 오른쪽 bit만 남긴다.

```text
0100 & 0001 = 0000
0101 & 0001 = 0001
```

즉 결과는 항상 `0` 또는 `1`이다.

#### `& mask`: 필요한 bit만 남기기

mask에서 `1`인 위치만 원래 bit를 보존한다.

```text
x                  = 0x92345678
x & 0x7fffffff     = 0x12345678
```

`0x7fffffff`는 bit 31만 `0`이므로 `x & 0x7fffffff`는 최상위 sign bit를 지운다.

#### `| mask`: 특정 bit를 1로 설정하기

`OR`는 mask가 `1`인 위치를 강제로 `1`로 만든다.

```text
x                  = 0x00000005
x | 0x80000000     = 0x80000005
```

이 식은 magnitude `5`를 유지하면서 bit 31을 sign bit로 설정한다.

`&`/`|`는 bitwise operator이고, C/C++의 logical operator `&&`/`||`와 다르다. 또한 여기서 `&`는 식의 두 operand 사이에 있으므로 C/C++ declaration의 reference나 단항 address-of operator가 아니다.

### 7. `LSB`와 `MSB`

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

### 8. ties-to-even의 의미와 bit 구현

#### 먼저 어떤 반올림을 구현하려는가

정밀도가 높은 값을 더 적은 자릿수로 줄이면 일부 자릿수를 버려야 한다. 이때 버려지는 부분을 보고 가장 가까운 표현값을 선택하는 것이 round-to-nearest다.

십진수 정수로 반올림하는 예를 들면 다음과 같다.

| 입력 | 가까운 정수 | 결과 |
|---:|---|---:|
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

원문의 `fp32 -> bf16` 구현은 **round-to-nearest, ties-to-even**을 사용한다. 일반적인 값은 가장 가까운 쪽으로 보내고, 정확한 halfway에서만 두 후보 중 even인 쪽을 선택한다.

| 입력 | 두 후보 | ties-to-even 결과 | 이유 |
|---:|---|---:|---|
| `2.5` | `2`, `3` | `2` | `2`가 even |
| `3.5` | `3`, `4` | `4` | `4`가 even |
| `6.5` | `6`, `7` | `6` | `6`이 even |
| `7.5` | `7`, `8` | `8` | `8`이 even |

따라서 ties-to-even은 “halfway이면 항상 올린다” 또는 “항상 내린다”는 규칙이 아니다.

```text
halfway + lower candidate even -> keep lower candidate
halfway + lower candidate odd  -> choose upper candidate
```

이 방식은 halfway가 반복해서 나타날 때 결과가 한 방향으로만 이동하는 편향을 줄이기 위해 사용된다.

#### 십진수 예제를 `H`와 `L`에 대응하기

이해를 위한 비유에서는 남길 정수 부분을 `H`, 버릴 소수 부분을 `L`처럼 생각할 수 있다.

```text
3.4 = [ 3 ][ .4 ]
        H     L

3.7 = [ 3 ][ .7 ]
        H     L

3.5 = [ 3 ][ .5 ]
        H     L
```

이 표기는 실제 IEEE floating-point encoding이 아니라 역할을 이해하기 위한 비유다.

- `H`: 최종 결과에 남길 부분이자 아래쪽 후보
- `H+1`: 위쪽 후보
- `L`: 버릴 부분이지만 어느 후보가 더 가까운지 알려 주는 나머지

따라서 반올림하는 대상은 `L` 자체가 아니다. **`L`을 보고 최종적으로 남을 `H`를 유지할지 `H+1`로 올릴지 결정한다.**

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

이것이 버릴 부분 `L`을 검사하는 과정에서 `H`의 정보도 가져오는 이유다. 반올림 대상은 `L`이 아니라 최종적으로 남을 `H`다. `L`은 거리와 halfway 여부를 알려 주고, halfway에서는 최종 `H`를 even으로 만들기 위해 `H`의 parity, 즉 even/odd 여부가 추가로 필요하다.

#### BF16에서 “even”이 뜻하는 것

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

#### 이제 carry로 반올림을 구현한다

**carry는 한 자리에서 표현할 수 있는 범위를 넘은 값을 바로 왼쪽 자리로 넘기는 것**이다. 한국어로는 받아올림이라고 한다.

##### 십진수의 받아올림

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

##### 이진수의 carry

이진수 한 자리는 `0`과 `1`만 표현할 수 있다. 따라서 `1+1`의 결과 `2`는 한 bit에 들어가지 않고 이진수 `10`으로 표현된다.

```text
1 + 1 = 10

result bit = 0
carry      = 1
```

이전 자리에서 들어오는 carry-in이 없을 때 한 bit 덧셈의 기본 경우는 다음과 같다.

| `A` | `B` | 현재 자리 결과 | 왼쪽으로 보내는 carry |
|---:|---:|---:|---:|
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

##### `0xffff + 1 = 0x1_0000`의 의미

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

##### 왜 `L`의 carry가 `H`를 바꾸는가

`H`와 `L`은 서로 떨어진 두 값이 아니라 하나의 연속된 32bit pattern을 설명하기 위해 절반으로 나눈 것이다.

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

##### Carry와 overflow의 차이

두 용어는 관련 있지만 관찰하는 bit width가 다르다.

- **carry**: 한 자리나 하위 bit group에서 넘친 값을 바로 왼쪽 자리로 전달하는 동작
- **overflow**: 정해진 전체 destination bit width로 결과를 표현할 수 없는 상태

위 예제에서 `L`만 16bit 값으로 보면 `0x8000+0x8000`은 16bit 범위를 넘는다. 하지만 실제 계산 대상은 `[H][L]`을 합친 32bit 값이므로 carry를 버리지 않고 `H`로 전달할 수 있다. 즉 `L`에서의 overflow를 전체 32bit 계산의 carry로 흡수하는 것이다.

`fp32 -> bf16` 식은 `L`에서 이 carry를 의도적으로 만들거나 만들지 않음으로써 `H`를 유지할지 `H+1`로 올릴지를 결정한다.

#### 여기서 `bias`가 뜻하는 것

이 절의 `bias`는 floating-point exponent에 더하는 exponent bias가 아니다. 버릴 하위 16bit를 반올림하기 위해 전체 32bit pattern에 더하는 **rounding offset**이다.

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

#### 왜 halfway가 `L=0x8000`인가

버려지는 `L`의 범위는 `0x0000`부터 `0xffff`다. 인접한 두 BF16 bit pattern 후보 `H`와 `H+1` 사이의 정확한 절반 지점이 `L=0x8000`이다.

| `L`의 범위 | 위치 | 반올림 결과 |
|---|---|---|
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

#### Halfway에서 `H`가 even인 경우

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

#### Halfway에서 `H`가 odd인 경우

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
|---|---:|---:|---:|---:|---:|
| even `0x3f80` | 0 | `0x7fff` | `0xffff` | 0 | `0x3f80` 유지 |
| odd `0x3f81` | 1 | `0x8000` | `0x1_0000` | 1 | `0x3f82` |

#### Halfway가 아닌 값도 같은 식으로 처리되는 이유

`H_lsb`에 따라 bias가 1만큼 달라지지만, 그 차이가 결과에 영향을 주는 것은 정확한 halfway뿐이다.

| `L` 조건 | 가장 경계에 있는 계산 | carry |
|---|---|---:|
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

### 9. two's-complement와 sign-magnitude

#### Two's-complement

일반적인 signed integer 표현이다. 음수 `-x`는 양수 `x`의 모든 bit를 뒤집고 1을 더해 만든다.

```text
+5 in 32 bits              = 0x00000005
invert all bits            = 0xfffffffa
add 1                      = 0xfffffffb
-5 in two's-complement     = 0xfffffffb
```

32bit two's-complement의 범위는 `-2^31`부터 `2^31 - 1`이다. 음수 쪽에 값이 하나 더 있기 때문에 `INT_MIN=-2^31`의 양수 counterpart인 `+2^31`은 `i32`로 표현할 수 없다.

원문이 다루는 고정폭 `SFPABS` semantics에서는 `INT_MIN`을 negate해도 같은 `INT_MIN` bit pattern이 남는다. `+2^31`을 `i32`로 표현할 수 없기 때문이다. 원문의 `i32 -> fp32`가 이 경우를 별도로 보정하는 이유다.

일반 C/C++ 코드에서는 `std::abs(INT_MIN)` 같은 연산이 표현 범위를 벗어나므로 이 hardware 결과를 그대로 가정하면 안 된다. 이 문서의 설명은 원문에 명시된 SFPU 동작을 가리킨다.

#### Sign-magnitude

MSB 하나는 sign, 나머지 bit는 절댓값 magnitude로 사용한다.

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

원문의 핵심 문제는 `SFPCAST`가 integer source를 sign-magnitude로 해석한다는 점이다. two's-complement `-5=0xfffffffb`를 그대로 sign-magnitude로 읽으면 “sign=1, magnitude=0x7ffffffb”가 되어 `-5`와 전혀 다른 값이 된다. 따라서 `SFPABS`, sign 복사, `INT_MIN` 보정이 필요하다.

### 10. `fp32`와 BF16 bit 구조

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

따라서 BF16에 필요한 정보는 `fp32` bit pattern의 상위 16bit `H`에 들어 있다. 단순히 하위 16bit `L`을 버리면 truncation이고, `L`을 검사해 `H`를 필요할 때 1 증가시키면 rounding이 된다.

원문에서 사용하는 기호는 다음 의미다.

- `Sign`: 부호 bit
- `Exp`: exponent field에서 bias를 제거한 exponent
- `Man`: 계산을 위해 implicit leading bit까지 포함한 mantissa

normalized 값은 개념적으로 다음 형태다.

```text
value = (-1)^Sign * (1.fraction) * 2^Exp
```

`1.fraction`의 앞쪽 `1`은 normalized floating-point encoding에 직접 저장하지 않아 implicit bit라고 부른다. zero, subnormal, infinity, NaN은 이 단순식의 예외이며 원문은 이들의 전체 동작을 설명하지 않는다.

### 11. 자주 나오는 결과 처리 용어

| 용어 | 의미 | 예 |
|---|---|---|
| truncate | 버려지는 자릿수를 반올림하지 않고 제거 | `3.9 -> 3`, `-3.9 -> -3` |
| round | 버려지는 부분을 보고 가까운 표현값을 선택 | `3.6 -> 4` |
| clamp | 값을 지정한 하한과 상한 안으로 제한 | `max(x, 0)` |
| saturate | 범위를 넘으면 destination datatype의 최솟값/최댓값에 고정 | `70000 -> u16 65535` |
| overflow | destination datatype으로 값을 표현할 수 없는 상태 | `2^32`를 `u32`로 변환 |
| sentinel | 특정 상태를 표시하기 위해 예약한 특별한 결과값 | overflow 시 `INT_MIN` |
| predication | lane별 조건에 따라 instruction 적용 여부를 제어 | 음수 lane에만 negate 적용 |

## SFPU hardware와 programming 기초

### Register, lane, row와 instruction

| 용어 | 이 문서에서의 의미 |
|---|---|
| register | hardware 내부에서 instruction이 직접 읽고 쓰는 작은 저장 공간 |
| `SrcA`, `SrcB` register | unpacker가 값을 놓고 matrix engine(FPU)이 입력 operand로 읽는 register. SFPU의 `LReg`와는 별개의 물리적 register다. |
| `Dst` register | unpacker와 MATH/SFPU가 값을 놓고 packer가 읽는 destination register 영역 |
| `Dst` tile slot | `Dst` 안에서 한 data tile이 차지하는 영역. 일반적인 full tile은 logical element 1024개를 네 face로 나누어 보관한다. |
| `LReg`, local register | `LReg[0]`, `LReg[1]`처럼 SFPU가 중간값을 보관하고 실제 vector instruction을 수행하는 내부 register. 일반적인 LReg 하나는 32 lanes × 32 bits다. |
| SFPU | Special Function Processing Unit. 32-lane vector instruction을 실제로 실행하는 hardware engine |
| SFPI | SFPU Interface. SFPU program을 C++ 형태로 작성하기 위한 library와 compiler |
| 32×32 data tile | logical element `32×32=1024`개로 이루어진 일반적인 Tensix data tile |
| face | 32×32 data tile을 나눈 16×16 영역. 일반적인 tile은 좌상, 우상, 좌하, 우하의 네 face로 구성된다. |
| logical tile row | 32×32 data tile에서 가로로 연속된 `1×32` element |
| `Dst` row | 일반적인 face 배치에서 16개 column을 갖는 hardware row |
| SFPU input row | 원문이 throughput 단위로 부르는 32-element SFPU vector. logical tile의 `1×32` row와 다르며, 이 문서에서는 필요할 때 vector group이라고도 부른다. |
| lane | 한 vector instruction 안에서 같은 연산을 독립적으로 수행하는 element 위치. Blackhole SFPU vector에는 32개 lane이 있다. |
| instruction | `SFPCAST`, `SFPIADD`처럼 hardware가 수행하는 한 명령 |
| sub-unit | Load, Simple, MAD, Round, Store처럼 서로 다른 종류의 instruction을 담당하는 실행 자원 |
| cycle | hardware clock 한 번의 진행 단위 |

#### `SrcA`, `SrcB`, `Dst`, `LReg`의 관계

`SrcA`와 `SrcB`, `LReg`는 모두 Tensix 내부 register라는 점만 같고, 소속된 compute engine과 역할이 다르다. `SrcA`와 `SrcB`는 matrix engine의 입력 register이고, `LReg`는 SFPU 내부의 vector register다. 같은 저장 공간을 이름만 다르게 부르는 것이 아니다.

```text
Matrix engine 경로

L1 SRAM
 |
 | Unpacker
 v
SrcA + SrcB             matrix engine의 입력 register
      |
      | Matrix/FPU 연산
      v
     Dst                matrix 결과가 놓이는 register set
      |
      | Packer
      v
   L1 SRAM
```

SFPU는 일반적으로 `SrcA`나 `SrcB`를 local operand register로 사용하지 않는다. SFPU가 elementwise 연산할 tile은 unpacker가 `Dst`에 직접 놓을 수 있고, matrix 연산 결과라면 이미 `Dst`에 있다. SFPU는 `Dst`의 32개 element를 `SFPLOAD`로 LReg에 읽고, LReg에서 하나 이상의 vector instruction을 수행한 뒤 `SFPSTORE`로 `Dst`에 기록한다.

```text
SFPU 경로

L1 SRAM
   |
   | Unpacker
   v
Dst의 input vector group
   |
   | SFPLOAD                         SFPU Load sub-unit
   v
input LReg
   |
   | SFPCAST / SFPIADD / SFPMAD / ...
   | 하나 이상의 SFPU instruction   LReg를 읽고 LReg에 결과 기록
   v
result LReg                          input과 같거나 다른 LReg
   |
   | SFPSTORE                        SFPU Store sub-unit
   v
Dst의 result vector group
   |
   | Packer
   v
L1 SRAM
```

이 그림은 instruction 사이의 **논리적인 data dependency**를 나타낸다. 실제 `SFPLOADMACRO` schedule에서는 서로 다른 input vector group에 대한 Load, Simple, MAD, Round, Store가 pipeline으로 겹칠 수 있다.

각 단계를 풀어 쓰면 다음과 같다.

1. `SFPLOAD`가 `Dst`의 32개 element를 한 LReg로 옮긴다.
2. `SFPCAST`, `SFPIADD`, `SFPMAD` 같은 SFPU instruction은 LReg를 operand로 읽는다.
3. 연산 결과는 같은 LReg를 갱신하거나 다른 LReg에 기록된다. 연산할 때마다 `Dst`로 중간 결과를 돌려보낼 필요는 없다.
4. 필요한 SFPU 연산이 끝나면 `SFPSTORE`가 결과 LReg의 32개 element를 `Dst`에 기록한다.
5. packer가 `Dst`의 tile을 format에 맞게 변환해 L1 SRAM으로 내보낸다.

여기서 Simple, MAD, Round 같은 연산 sub-unit은 모두 SFPU 내부의 실행 자원이다. 값을 별도의 외부 저장 공간에 옮겼다가 LReg로 되돌리는 것이 아니라, 해당 sub-unit이 LReg operand를 읽고 계산 결과를 LReg에 쓴다.

따라서 `Dst`는 matrix engine의 결과 공간인 동시에 SFPU와 packer가 접근하는 공유 지점이고, LReg는 그중 32개 element를 가져와 SFPU가 실제 계산하는 내부 작업 공간이다. 한 LReg에는 tile 전체가 아니라 한 32-element SFPU vector만 들어간다. **일반적인 지원 경로에는 LReg에서 L1 SRAM으로 직접 쓰는 경로가 없다.** LReg의 결과를 L1 SRAM으로 보내려면 반드시 다음 경로를 거친다.

```text
LReg --SFPSTORE--> Dst --Packer--> L1 SRAM
```

여기서 `L1 SRAM`과 SFPU local register를 가리키는 `LReg[1]`은 완전히 다른 저장 공간이다. 일부 코드나 설명이 `LReg[1]`을 짧게 `L1`이라고 부를 수 있으므로, 이 문서에서는 memory를 뜻할 때 `L1 SRAM`, SFPU register를 뜻할 때 `LReg[1]`로 구분해 표기한다.

#### SFPI란?

`SFPI`는 **SFPU Interface**의 약자다. SFPU가 실제 vector hardware라면 SFPI는 그 hardware를 C++ 형태로 프로그래밍하기 위한 software interface와 compiler다. `sfpi::`는 SFPI가 제공하는 C++ namespace다.

```text
SFPI C++ 코드
     |
     | SFPI compiler
     v
SFPLOAD, SFPCAST, SFPADD, SFPSTORE 등의 SFPU instruction
     |
     v
SFPU hardware에서 실행
```

예를 들어 `sfpi::vFloat`와 `sfpi::vInt`는 32-lane vector type이고, `sfpi::abs()` 같은 함수와 operator는 대응하는 SFPU 연산을 표현한다. SFPI compiler는 이를 실제 SFPU instruction으로 변환하고 LReg 할당과 instruction scheduling을 처리한다. 따라서 SFPI C++ 한 줄과 hardware instruction 하나가 항상 1:1로 대응하지는 않는다.

`sfpi::dst_reg[i]`도 LReg 자체를 뜻하지 않는다. 이는 `Dst` 안의 i번째 32-element vector group을 가리키는 SFPI view다. 다음 코드는 개념적으로 `SFPLOAD`와 `SFPSTORE`를 포함한다.

```cpp
sfpi::vFloat v = sfpi::dst_reg[i];  // Dst vector i --SFPLOAD--> LReg
v = v + 1.0f;                       // LReg의 32개 lane에서 연산
sfpi::dst_reg[i] = v;               // LReg --SFPSTORE--> Dst vector i
```

정리하면 register 사이의 관계는 다음과 같다.

| Register | 주 사용자 | 역할 |
|---|---|---|
| `SrcA`, `SrcB` | matrix engine | matrix 연산의 입력 operand |
| `Dst` | matrix engine, SFPU, packer | matrix 결과와 SFPU 작업 tile이 놓이는 공유 영역 |
| `LReg` | SFPU | `Dst`에서 읽은 32개 element에 vector 연산을 수행하는 내부 register |

#### 32×32 tile에서 32-lane SFPU vector까지

이 절은 Blackhole의 일반적인 full-tile elementwise 경로를 기준으로 한다. transpose, `Dst` address remap, `DEST_RD_COL_EXCHANGE` 같은 특수 설정은 사용하지 않는 경우다. 이 범위의 공간적 배치는 Wormhole도 같다.

unpacker는 L1 SRAM의 32×32 data tile을 한 `Dst` tile slot에 네 개의 16×16 face 형태로 배치한다. 이 시점에는 tile의 logical element 1024개가 `Dst`에 있지만, SFPU local register 하나에 1024개가 동시에 들어 있는 것은 아니다. SFPU는 `SFPLOAD`를 통해 `Dst`의 특정 32개 위치만 한 local register의 32개 lane으로 읽는다.

```text
L1 SRAM의 32×32 data tile
          |
          | Unpacker
          v
Dst의 한 tile slot
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
          | SFPLOAD: 32개 위치 선택
          v
LReg[k][0..31]
          |
          | SFPCAST, SFPIADD 등의 lanewise instruction
          v
32개 lane이 같은 연산을 독립적으로 수행
```

여기서 lane과 element는 같은 개념은 아니지만 일반적인 lanewise instruction에서는 1:1로 대응한다.

- **lane**은 같은 vector instruction을 독립적으로 수행하는 hardware 실행 위치다.
- **element**는 해당 lane에 들어가 처리되는 data 하나다.

따라서 모든 lane이 활성화된 일반적인 경우 한 SFPU vector instruction은 다음과 같이 32개 element를 처리한다.

```text
32 lanes × lane당 1 element = 최대 32 elements

LReg[k][ 0] -> lane  0이 element  0 처리
LReg[k][ 1] -> lane  1이 element  1 처리
...
LReg[k][31] -> lane 31이 element 31 처리
```

predication으로 일부 lane을 비활성화하면 실제로 연산되는 element 수는 32개보다 적을 수 있다. 또한 BF16이나 `u16`처럼 원래 element 폭이 32bit보다 작아도 LReg에서는 lane마다 하나의 32-bit 저장 공간을 사용하며, `SFPLOAD` 과정에서 필요한 변환이나 확장이 적용된다.

한 번의 일반적인 `SFPLOAD`는 16×16 face에서 **연속된 네 `Dst` row × 짝수 또는 홀수 여덟 column**을 읽는다. 즉 먼저 4×16 영역을 잡은 뒤 column을 한 칸씩 건너뛴 4×8 영역을 선택하며, `4×8=32`개 element가 한 SFPU vector를 구성한다.

face의 row 0..3을 예로 들면 다음과 같다.

```text
              face column
           0  1  2  3  4  5  ... 14 15
row 0      E  O  E  O  E  O  ...  E  O    // E : Even; O : Odd
row 1      E  O  E  O  E  O  ...  E  O
row 2      E  O  E  O  E  O  ...  E  O
row 3      E  O  E  O  E  O  ...  E  O

첫 vector, 짝수 column:
  lane  0..7  <- row 0, col 0,2,4,...,14
  lane  8..15 <- row 1, col 0,2,4,...,14
  lane 16..23 <- row 2, col 0,2,4,...,14
  lane 24..31 <- row 3, col 0,2,4,...,14

다음 vector, 홀수 column:
  lane  0..7  <- row 0, col 1,3,5,...,15
  lane  8..15 <- row 1, col 1,3,5,...,15
  lane 16..23 <- row 2, col 1,3,5,...,15
  lane 24..31 <- row 3, col 1,3,5,...,15
```

따라서 한 LReg의 lane layout은 logical tile row와 같은 `1×32`보다는 다음 `4×8`로 보는 편이 정확하다.

```text
lane  0  1  2  3  4  5  6  7
lane  8  9 10 11 12 13 14 15
lane 16 17 18 19 20 21 22 23
lane 24 25 26 27 28 29 30 31
```

한 face 안에서는 이 pattern을 다음 순서로 반복한다.

```text
face-local vector 0/1 : rows  0..3, 짝수/홀수 column
face-local vector 2/3 : rows  4..7, 짝수/홀수 column
face-local vector 4/5 : rows  8..11, 짝수/홀수 column
face-local vector 6/7 : rows 12..15, 짝수/홀수 column
```

즉 한 16×16 face에는 8개 SFPU vector가 있고, 한 32×32 tile에는 다음과 같이 32개 vector가 있다.

```text
Face 0 -> sfpi::dst_reg[ 0.. 7]
Face 1 -> sfpi::dst_reg[ 8..15]
Face 2 -> sfpi::dst_reg[16..23]
Face 3 -> sfpi::dst_reg[24..31]

4 faces × 8 vectors/face × 32 lanes/vector = 1024 elements
```

##### 32-lane 병렬 처리가 tile 전체 병렬 처리를 뜻하지는 않는다

32개 lane이 병렬로 동작한다는 말은 **한 vector instruction이 현재 선택된 vector group의 32개 element를 동시에 처리한다**는 뜻이다. 32×32 tile의 1024개 element 전체를 한 번에 처리한다는 뜻은 아니다.

```text
32×32 tile = 1024 elements

vector group  0 : 32 lanes가 32 elements 처리
vector group  1 : 32 lanes가 다음 32 elements 처리
...
vector group 31 : 32 lanes가 마지막 32 elements 처리

1024 elements / 32 lanes = 32 vector groups
```

간단히 비유하면 같은 작업을 하는 작업자 32명이 한 round에 element 32개를 맡고, 32개 round에 걸쳐 tile 전체를 방문하는 형태다. 다만 lane은 독립적인 software thread가 아니다. 32개 lane은 같은 instruction을 lockstep으로 실행하고, lane마다 별도 program counter나 서로 다른 instruction stream을 갖지 않는다. predication은 이 중 일부 lane의 결과 기록을 비활성화할 수 있다.

또한 위의 32 vector groups는 처리 묶음의 개수이지 32 cycles를 뜻하지 않는다. vector group 하나의 연산이 `SFPLOAD`, 여러 arithmetic instruction, `SFPSTORE`로 구성될 수 있고, 뒤에서 설명하는 pipeline이 서로 다른 vector group의 단계를 겹칠 수 있다.

logical tile의 가로 `1×32` row 네 개를 모두 처리하려면 좌측 face의 짝수/홀수 vector와 우측 face의 짝수/홀수 vector, 총 네 vector가 필요하다. 따라서 SFPU가 vector engine이라는 말은 가로나 세로 방향을 한 번에 처리한다는 뜻이 아니다. **하나의 instruction이 32개 lane에 같은 연산을 적용하는 SIMD engine**이라는 뜻이며, `Dst`에서 그 32개 lane으로 연결되는 기본 모양이 4×8인 것이다.

#### `row`와 throughput을 읽는 방법

하나의 SFPU input row에 대한 연산은 여러 instruction으로 구성될 수 있다. pipeline이 채워지면 input row 하나의 모든 instruction이 끝나기를 기다리지 않고 다음 input row의 load를 시작할 수 있다. 따라서 “한 input row가 완료되는 총 latency”와 “연속된 input row를 몇 cycle 간격으로 시작하거나 완료할 수 있는가”는 다르다.

원문의 `N cycles per input row`에서 `N`은 보통 **steady-state initiation interval**, 즉 pipeline이 채워진 뒤 다음 32-element vector group을 시작할 수 있는 간격이다. 한 vector group의 load부터 마지막 store까지 걸리는 latency가 `N` cycle이라는 뜻은 아니다.

| 구분 | 묻는 것 |
|---|---|
| latency | vector group 하나가 첫 load를 시작한 뒤 마지막 store를 마칠 때까지 얼마나 걸리는가? |
| throughput 또는 initiation interval | 연속된 두 vector group을 몇 cycle 간격으로 시작하고, steady state에서 몇 cycle마다 하나씩 끝낼 수 있는가? |

예를 들어 `fp32 -> bf16` schedule에서는 첫 group이 cycle 0에 시작해 마지막 store를 cycle 5에 실행하지만, 다음 group은 그 완료를 기다리지 않고 cycle 3에 시작한다.

```text
absolute cycle       0 1 2 3 4 5 6 7 8 ...
group 0              S-----work----F
group 1                    S-----work----F
group 2                          S-----work----F

S : 첫 SFPLOADMACRO issue
F : 마지막 store

group 시작 간격 = 3 cycles
=> throughput = 3 cycles/input row
```

따라서 일반적인 full tile의 32개 SFPU vector group에 대해 `3 cycles/input row`라면 반복 본체의 steady-state 비용은 개념적으로 `32 × 3 = 96 cycles`다. 다만 실제 tile 처리 시간에는 상수와 macro 설정 같은 prologue, 마지막 예약 instruction을 끝내는 drain, loop 주변 instruction과 다른 hardware stall이 추가될 수 있으므로 곧바로 “tile latency가 정확히 96 cycles”라고 해석하면 안 된다.

#### `VD`: Vector Destination

`VD`는 관례적으로 **Vector Destination**으로 읽으며, SFPU instruction encoding에서 vector register `D`를 선택하는 operand다. 일반적인 arithmetic, cast, load instruction에서는 결과를 기록할 `LReg[VD]`를 지정한다. 여기서 `VD`의 `D`는 operand 이름이므로 Tensix의 `Dst` register와 혼동하면 안 된다.

```text
VA, VB, VC : instruction에 따라 사용하는 vector source operand
VD         : 일반적으로 결과를 기록하는 vector destination operand

예: SFPCAST VC=2, VD=3
    LReg[2] --SFPCAST--> LReg[3]
```

다만 `VD`가 항상 write-only인 것은 아니다. `SFPIADD`처럼 기존 `LReg[VD]` 값도 operand로 읽는 read-modify-write instruction이 있고, `SFPSTORE`에서는 `VD`가 `Dst`에 저장할 source LReg를 선택한다. 따라서 정확한 read/write 방향은 각 instruction의 functional model을 확인해야 한다.

일반적으로 쓸 수 있는 arithmetic destination은 `LReg[0]`부터 `LReg[7]`까지다. `SFPLOADMACRO`에는 `VD=16`으로 선택하는 특별한 `LReg[16]`도 있다. 이 register는 일반 LReg처럼 자유롭게 접근하는 공간이 아니라 `SFPLOADMACRO`가 schedule한 instruction을 통해 제한적으로 읽고 쓸 수 있는 추가 register다.

```text
VD != 16 : 대개 LReg[0..7] 중 하나인 일반 destination
VD == 16 : SFPLOADMACRO 전용 추가 register인 LReg[16]
```

원문의 schedule에서 `VD=16`과 `VD!=16`의 구분은 Simple과 Round sub-unit을 같은 cycle에 배치하기 위한 hardware scheduling 제약으로 등장한다. 두 instruction 중 하나는 `VD=16`, 다른 하나는 `VD!=16`이어야 한다.

### SFPU 스케줄링을 읽기 위한 기본 개념

원문은 SFPU 실행 자원을 다음 다섯 열로 나누어 스케줄을 설명한다.

| 자원 | 대표 역할 |
|---|---|
| Load | `SFPLOAD`로 `Dst`의 한 32-element SFPU vector group을 local register로 읽는다. |
| Simple | bitwise, add, sign 설정, min/max 같은 단순 연산을 수행한다. |
| MAD | multiply-add와 indirect register 선택을 수행한다. |
| Round | shift나 `SFPSTOCHRND` 계열 변환을 수행한다. |
| Store | 결과를 `Dst`에 기록한다. |

#### 원문의 schedule 그림 읽는 법

원문의 그림은 위에서 아래로 시간이 흐르는 pipeline schedule이다. 왼쪽 다섯 열과 오른쪽 register 열은 서로 다른 정보를 보여 준다.

| 그림 부분 | 의미 |
|---|---|
| 가로 한 줄 | hardware의 한 execution cycle |
| `load`, `simple`, `mad`, `round`, `store` | 그 cycle에 각 sub-unit에서 실행되는 instruction |
| 오른쪽 `0..7`, `16` | cycle 번호가 아니라 `LReg[0..7]`, `LReg[16]`의 register 번호 |
| 오른쪽의 세로 색 막대 | 해당 값이 LReg에 존재하며 아직 뒤 instruction에서 필요로 되는 live range |
| `a0`, `v0`, `a1`, `v1`의 suffix | lane 번호가 아니라 예시 schedule의 input vector group 번호 |
| 같은 색 | `a0` 또는 `v0`처럼 같은 이름을 가진 하나의 값/instruction stream의 흐름 |
| instruction 왼쪽 위의 작은 숫자 | `SFPLOADMACRO`가 참조하는 사전 설정 instruction template 번호 |

즉 그림 오른쪽의 `0 1 2 3 4 5 6 7 16`을 시간축으로 읽으면 안 된다. 시간축은 그림의 **세로 방향**이고, 오른쪽 열은 같은 시간 동안 어느 LReg가 사용 중인지 나타낸다.

원문의 별도 text schedule에서 `[a]` 같은 표기는 “`VD=a`인 `SFPLOADMACRO`가 예약한 instruction”이라는 뜻이다. `...`도 새 instruction 이름이 아니라 앞서 issue한 macro의 예약 동작이 계속 pipeline에서 진행됨을 나타낸다.

#### 한 `SFPLOADMACRO`가 여러 칸을 채우는 이유

`SFPLOADMACRO` 한 개를 issue하면 현재 cycle에는 Load sub-unit에서 `Dst -> LReg` load를 수행하고, 사전에 설정한 instruction을 Simple, MAD, Round, Store sub-unit에 각각 최대 하나씩 미래 cycle용으로 예약할 수 있다. 따라서 그림에 색칠된 칸 수를 모두 더한 값은 issue cycle 수가 아니다.

##### MOP과 비슷한가

“하나의 대표 instruction으로 미리 설정한 여러 동작을 발동한다”는 직관에서는 MOP과 비슷하다. 하지만 두 hardware mechanism은 동작 위치와 규모가 다르므로 같은 것으로 보면 안 된다.

| 구분 | MOP | `SFPLOADMACRO` |
|---|---|---|
| hardware 위치 | Tensix frontend의 MOP Expander | SFPU의 Load sub-unit과 각 실행 sub-unit의 예약 queue |
| 한 번 실행했을 때 | 하나의 `MOP`을 반복을 포함할 수 있는 긴 일반 Tensix instruction stream으로 확장 | `SFPLOAD`를 즉시 실행하고 Simple, MAD, Round, Store instruction을 각각 최대 하나씩 미래 cycle에 예약 |
| 주 용도 | 반복되는 instruction 발행을 frontend에서 압축 | SFPU sub-unit을 겹쳐 사용해 vector group 처리 간격을 단축 |
| 실행 모양 | 확장된 instruction들이 backend로 차례로 전달됨 | 예약된 instruction들이 설정된 delay 뒤 서로 다른 sub-unit에서 실행됨 |

따라서 `SFPLOADMACRO`는 **작은 instruction 모음 자체**라기보다 다음과 같이 이해하는 편이 정확하다.

> `SFPCONFIG`로 미리 설정한 짧은 SFPU pipeline schedule을, `Dst -> LReg` load와 함께 발동시키는 instruction

```text
SFPCONFIG
  - instruction template 설정
  - Simple/MAD/Round/Store별 delay 설정
  - operand override와 store format 설정
          |
          v
SFPLOADMACRO(MacroIndex, VD, Dst address)
  - 현재 cycle: SFPLOAD 실행
  - 미래 cycle: 설정된 sub-unit instruction 예약
```

여기서 “한 번에 실행한다”는 말은 **모든 동작이 같은 cycle에 끝난다**는 뜻이 아니다. 하나의 foreground issue가 현재 load와 여러 미래 동작의 예약을 함께 발생시킨다는 뜻이다. 또한 hardware가 data 조건을 검사해 macro를 자동 선택하는 것도 아니다. 프로그램이 `MacroIndex`를 지정해 사전 설정된 sequence 중 하나를 호출하며, data-dependent 조건 처리는 그 안에 예약된 개별 instruction의 lane flag나 operand semantics가 담당한다.

```text
cycle k    : SFPLOADMACRO issue + Load 실행
cycle k+d0 : 예약된 Simple instruction 실행 가능
cycle k+d1 : 예약된 MAD instruction 실행 가능
cycle k+d2 : 예약된 Round instruction 실행 가능
cycle k+d3 : 예약된 Store instruction 실행 가능
```

Blackhole Vector Unit은 한 cycle에 Load 계열 하나와 Simple, MAD, Round, Store 각각 하나를 실행할 수 있다. 여러 sub-unit을 같은 cycle에 채우는 경로는 `SFPLOADMACRO`의 사전 예약을 사용한다. 그래서 cycle 수를 셀 때는 다음 순서로 읽는 편이 안전하다.

1. input group 하나를 시작하기 위해 foreground에서 실제로 issue하는 `SFPLOADMACRO`, 일반 SFPU instruction, 반복 본체의 `SFPNOP`을 찾는다.
2. 다음 group의 같은 시작점까지 몇 가로 행이 떨어져 있는지 센다. 이것이 steady-state cycles/input row다.
3. macro가 나중에 실행시키는 Simple/MAD/Round/Store 칸은 dependency와 자원 충돌을 검증하는 데 사용하되, 각각을 새 foreground issue cycle로 다시 더하지 않는다.
4. 반복이 끝난 뒤에만 나오는 `SFPNOP`은 아직 예약되어 있는 instruction을 완료시키는 pipeline drain이다. steady-state cycles/input row에는 포함되지 않지만 짧은 loop의 총 latency에는 포함된다.

반대로 반복 본체 안에서 매 group마다 의도적으로 issue하는 `SFPNOP`은 자원 제약을 지키기 위한 실제 issue slot이므로 cycles/input row에 포함된다.

개념적으로는 한 SFPU input row가 아래 단계를 지나지만, `SFPLOADMACRO`는 서로 다른 input row의 단계를 겹쳐 실행한다.

```text
time ------------------------------------------------------------>

input row n     Load ----> Simple/MAD/Round --------------------> Store
input row n+1             Load ----> Simple/MAD/Round ----------------> Store
input row n+2                       Load ----> Simple/MAD/Round ------------> Store

             +------------------------------------------+
             |      overlapped sub-unit scheduling      |
             +------------------------------------------+
```

이 schedule에는 서로 다른 두 종류의 병렬성이 함께 나타난다.

| 병렬성 | 동시에 진행되는 것 |
|---|---|
| lane-level SIMD | 한 vector group의 최대 32개 element가 32개 lane에서 같은 instruction을 수행 |
| sub-unit pipeline | 서로 다른 vector group의 Load, Simple, MAD, Round, Store 단계가 겹쳐 진행 |

위 그림의 `input row n`, `n+1`, `n+2`는 각각 32개 lane을 뜻하는 것이 아니라 **서로 다른 32-element vector group**을 뜻한다. 예를 들어 group `n`이 Round 단계에 있을 때 group `n+1`은 Simple 단계, group `n+2`는 Load 단계에 있을 수 있다. 따라서 한 tile의 32개 vector group을 순회하더라도 전체 시간은 단순히 `32 × 한 group의 latency`로 계산되지 않는다.

중요한 스케줄링 제약은 다음과 같다.

1. Simple과 Round sub-unit을 같은 시점에 쓰려면 원문이 다루는 경우 한 instruction의 `VD`가 `16`이고 다른 instruction의 `VD`가 `16`이 아니어야 한다.
2. 값의 live range가 initiation interval보다 길면 같은 local register를 매 input row 재사용할 수 없다. 이때 `LReg[0]`/`LReg[1]` 또는 `LReg[2]`/`LReg[3]`처럼 register를 번갈아 사용한다.
3. `SFPLOADMACRO`는 load에 미리 구성한 Simple, MAD, Round, Store 동작을 결합해 일반 instruction 발행과 자동 stall을 줄인다.
4. 표의 cycle/row에서 row는 SFPU input row를 뜻하며, 수치는 pipeline이 채워진 뒤의 처리 간격이다. 초기화와 drain에 필요한 cycle은 별도로 존재한다.

원문의 assembly 표기는 destination register를 마지막 operand에 둔다. 예를 들어 `sfpmad A, B, C, D`는 `D = A * B + C`를 뜻한다.

## 참고 자료

- [Typecast on Tenstorrent](https://www.jasondavies.com/2025/tenstorrent-typecast/)
- [Blackhole Vector Unit ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md)
- [Blackhole SFPLOAD ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md)
- [Blackhole LReg ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/LReg.md)
- [Blackhole Dst ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/Dst.md)
- [Blackhole SFPLOADMACRO ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md)
- [Blackhole MOP Expander ISA 문서](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/MOPExpander.md)
- [tt-metal compute engine과 register dataflow](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst)
- [tt-metal tile 내부 구조](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/advanced_topics/tiles.rst)
- [tt-metal SFPI 소개](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/examples/custom_sfpi.rst)
- [tt-metal custom SFPI add 예제](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/examples/custom_sfpi_add.rst)
