# 저장소 작업 지침

## 저장소 성격과 적용 범위

이 저장소는 Jekyll Chirpy 기반 개인 기술 블로그다. 이 `AGENTS.md`는 저장소 전체에 적용한다.

- 게시물 원문은 `_posts/`에 둔다.
- GitHub Pages 배포 workflow는 `.github/workflows/pages-deploy.yml`이다.
- `master` 또는 `main` push가 실제 사이트 배포를 시작한다.
- 현재 사이트 URL은 `https://molly-dooker.github.io`다.
- 참고할 대표적인 한국어 기술 문서 형식은 `_posts/2026-08-09-tenstorrent-llk-basics.md`다.

작업 전에는 항상 `git status -sb`로 기존 변경을 확인한다. 사용자의 변경을 덮어쓰거나, 요청과 무관한 파일을 수정·stage하지 않는다.

파일을 수정하기 전에 다음을 포함한 구체적인 계획을 먼저 제시한다.

1. 새로 만들거나 수정할 파일
2. 참고할 기존 문서와 출처
3. 작성 또는 변경할 내용의 구조
4. 수행할 검증
5. 요청된 경우에만 적용할 commit·push 범위

## 포스트 파일과 front matter

새 포스트의 파일명은 다음 형식을 사용한다.

```text
_posts/YYYY-MM-DD-lowercase-kebab-case-slug.md
```

날짜와 시간은 `_config.yml`의 `Asia/Seoul` timezone을 기준으로 한다. 필요하면 다음 명령으로 현재 시간을 확인한다.

```bash
date '+%Y-%m-%d %H:%M:%S %z'
```

기술 포스트는 기본적으로 다음 front matter를 사용한다.

```yaml
---
title: "문서 제목"
date: YYYY-MM-DD HH:MM:SS +0900
categories: [상위 분류, 하위 분류]
tags: [lowercase-tag, another-tag]
description: "검색 결과와 Open Graph에 사용할 한 문장 설명"
render_with_liquid: false
---
```

- 수식이 하나라도 있으면 `math: true`를 추가한다.
- `title`과 `description`은 반드시 따옴표로 감싼다.
- tag는 lowercase kebab-case를 우선한다.
- category와 tag를 새로 만들기 전에 기존 포스트의 분류와 중복되는지 확인한다.
- `render_with_liquid: false`는 수식의 중괄호와 예제 코드를 Liquid가 잘못 해석하는 일을 막으므로 기술 포스트에서 기본값으로 사용한다.

## 기술 문서 작성 방식

기본 본문은 한국어로 작성하되, code identifier와 널리 쓰이는 기술 용어는 억지로 번역하지 않는다. 처음 등장하는 약어는 원래 이름과 의미를 함께 설명한다.

### 문체와 문장 다듬기

새 글을 쓰거나 기존 글을 교정할 때는 원문의 문장 구조를 그대로 옮기지 말고, 같은 기술적 의미를 한국어 독자가 한 번에 이해할 수 있는 문장으로 다시 구성한다. 정확한 내용도 문장이 번역체이거나 지나치게 딱딱하면 읽기 어려우므로 기술 검토와 문체 검토를 별도 단계로 수행한다.

- Code identifier, API 이름, instruction, register, tensor shape처럼 표기가 의미의 일부인 용어는 그대로 유지한다.
- 일반 명사로 쓰인 `weight`, `row`, `output`, `error`, `update`, `matrix`, `inverse Hessian` 등은 문맥상 자연스러우면 `가중치`, `행`, `출력`, `오차`, `갱신` 또는 `보정`, `행렬`, `역헤시안`처럼 쓴다. 다만 code, 수식, 인용, 고유한 algorithm 용어까지 일괄 치환하지 않는다.
- 한 문장에 원인, 동작, 결과, 예외가 한꺼번에 들어가면 두세 문장으로 나눈다. 주어와 핵심 동작을 앞에 두고 조건이나 제약은 뒤에서 보충한다.
- 불필요한 피동형과 명사화를 줄인다. `수행된다`, `사용하는 것이 필요하다`, `확인하는 것이 가능하다`보다 `수행한다`, `사용해야 한다`, `확인할 수 있다`처럼 직접 쓴다.
- `뜻이다`, `것이다`, `하게 된다`, `할 수 있다`, `필요하다`가 가까운 문장에 반복되면 문장을 합치거나 서술어를 구체화한다. 이 표현들을 기계적인 금지어로 취급하지는 않는다.
- 영어 복수형과 한국어 복수 표현을 겹쳐 쓰지 않는다. 실제 identifier나 인용이 아니라면 `weights`, `rows`, `chunks`, `cores`보다 문맥에 맞는 한국어 단수형이나 수량 표현을 우선한다.
- 같은 개념은 한 문서 안에서 같은 용어로 부른다. `보상`, `보정`, `갱신`처럼 의미가 다른 표현을 섞어 쓸 때는 실제 연산의 역할에 맞춰 구분한다.
- 문체를 다듬는 과정에서 수식, 수치, tensor shape, algorithm 순서, API semantics, 조건부 표현과 출처의 의미를 바꾸지 않는다. `대체로`, `특정 조건에서`, `가능하다` 같은 한정어를 삭제해 주장을 더 강하게 만들지 않는다.
- 제목과 본문뿐 아니라 front matter의 `description`, 표의 설명, 목록, 그림 caption과 alt text, code block 전후 문장도 같은 기준으로 검토한다.

다음과 같은 직역형 문장은 기술 용어의 성격을 확인한 뒤 자연스럽게 고친다.

```text
Weight 하나를 제거한 뒤 inverse Hessian을 update한다.
-> 가중치 하나를 제거한 뒤 역헤시안을 갱신한다.

이는 다음 row가 같은 register를 overwrite하게 된다는 것을 의미한다.
-> 그러면 다음 행이 같은 register를 덮어쓴다.
```

문체 교정은 다음 순서로 진행한다.

1. 원문과 primary source에서 보존해야 할 수치, 수식, 용어, 조건을 먼저 확인한다.
2. 문단마다 핵심 주장 하나가 먼저 드러나도록 문장 순서와 길이를 다듬는다.
3. 영어 일반 명사, 피동형, 명사화, 반복 서술어를 문맥에 맞게 고친다.
4. 표, 목록, 그림 설명까지 포함해 용어와 어조의 일관성을 확인한다.
5. 최종 diff를 원문과 다시 대조해 문체 수정이 사실 변경으로 번지지 않았는지 확인한다.

권장 구성은 다음과 같다.

1. `## 문서 범위`: 다루는 범위, 기준 환경, 출처와 확인일
2. 문제의식 또는 배경
3. 핵심 개념과 용어
4. 단계별 유도·구현·분석
5. 실험 결과 또는 구체적인 예제
6. 제약, 가정
7. 핵심 정리
8. `## 참고 자료`

모든 글이 이 순서를 기계적으로 따를 필요는 없지만, 독자가 “왜 필요한가 → 어떻게 동작하는가 → 결과를 어떻게 해석하는가”를 따라갈 수 있어야 한다.

- 사실과 해석을 구분한다.
- 수치, API 동작, hardware semantics처럼 정확성이 중요한 내용은 원 출처와 대조한다.
- 논문·공식 문서·공식 source code 같은 primary source를 우선한다.
- 외부 자료는 검색 결과가 아니라 해당 내용을 직접 뒷받침하는 페이지에 link한다.
- 긴 원문을 그대로 옮기지 말고 요약·재구성한다.
- code block에는 `cpp`, `python`, `bash`, `text`처럼 language를 명시한다.
- 같은 개념을 본문, 표, 목록에서 불필요하게 반복하지 않는다.

## 수식, 표, 이미지

### 문자 기반 시각화

중요한 동작이나 알고리즘을 설명하는 문서에는 최소 하나 이상의 문자 기반 diagram을 포함한다.

- 구조선과 방향 표시는 주로 `+`, `-`, `|`, `<`, `>`, `v`, `^`를 사용한다.
- Diagram은 `text` code block 안에 작성하고, 고정폭 글꼴에서 열과 화살표가 정렬되도록 한다.
- Diagram 내부의 label, 단계명, 분기 문구와 설명은 ASCII 영문으로 작성한다. 한글 제목과 해설은 code block 밖에 두며, 수도 코드의 한글 주석에는 이 제한을 적용하지 않는다.
- 입력, 출력, 상태, memory 계층, producer/consumer 또는 단계 이름을 diagram 안에 직접 표시한다.
- 반복은 되돌아가는 화살표로, 분기와 병합은 갈라지고 합쳐지는 선으로 표현한다.
- 수식이나 표만으로 핵심 흐름을 대체하지 않는다. 수식과 표가 있더라도 실행 순서나 데이터 이동을 diagram으로 함께 보여준다.
- Mermaid, 이미지 또는 Unicode box-drawing 문자를 사용하더라도 ASCII diagram을 생략하지 않는다.
- 장식 목적의 복잡한 그림보다 실제 코드 동작과 상태 전이를 정확히 표현하는 간결한 그림을 우선한다.

예:

```text
+-------------+       +-------------+
| input tile  |------>| compute     |
+-------------+       +------+------+
                            |
                            v
                     +-------------+
                     | output tile |
                     +-------------+
```

### MathJax

수식이 있는 포스트에는 front matter에 `math: true`를 반드시 넣는다.

- inline 수식: 문장 안에서 single-dollar delimiter인 `$L_q$` 사용
- display 수식: 문장과 분리된 별도 줄에서 `$$` delimiter 사용
- display delimiter 앞뒤에는 빈 줄을 둔다.
- 작성 후 `$$` 개수가 짝수인지 확인한다.
- Liquid 충돌을 피하기 위해 `render_with_liquid: false`를 유지한다.

`$$...$$`는 inline 수식의 강조형이 아니라 별도의 display 수식이다. 문장 중간의 짧은 수식을 고치기 위해 `$...$`를 `$$...$$`로 바꾸지 않는다.

Jekyll의 Markdown parser는 MathJax보다 먼저 실행된다. 따라서 **표 안이 아니더라도** inline 수식에 raw `|`를 쓰면 table column separator로 오인해 문단 전체가 table HTML로 깨질 수 있다. 절댓값과 norm에는 다음처럼 `\lvert`, `\rvert`, `\lVert`, `\rVert`를 사용한다.

```text
$\lvert w_q\rvert$
```

다음 표기는 사용하지 않는다.

```text
$|w_q|$
```

표 separator는 markdownlint와 호환되는 다음 형식을 사용한다.

```markdown
| Name | Value |
| --- | ---: |
| A | 1 |
```

### 이미지

- 수식, 표, 짧은 code는 가능한 한 검색·복사가 가능한 text, LaTeX, Markdown으로 작성한다.
- 이미지는 text로 표현하기 어려운 diagram, screenshot, 측정 graph에만 사용한다.
- 새 image asset이 필요하면 `assets/img/posts/<post-slug>/` 아래에 둔다.
- 포스트에서는 `/assets/img/posts/<post-slug>/<file-name>` 형식으로 참조하고 실제 파일명과 확장자의 대소문자까지 일치하는지 확인한다. 로컬에만 존재하고 Git에 포함되지 않은 asset이 없는지도 확인한다.
- 의미 있는 alt text를 작성한다.
- 출처가 있는 이미지는 license와 인용 가능 여부를 확인하고 출처를 표시한다.
- 접근 권한이 필요한 Jira·Confluence URL을 그대로 image source로 사용하지 않는다.
- Diagram이 실제 timing, 측정값 또는 source 구조를 단순화한 개념도라면 caption이나 본문에서 단순화한 범위를 명시한다.
- Light와 dark theme에서 글자, 선, 배경의 대비가 충분한지 확인한다. 작은 화면에서 축소해도 label을 읽을 수 있어야 한다.

SVG에는 `viewBox`만 두지 말고 원본 좌표계와 같은 aspect ratio의 `width`와 `height`도 명시한다.

```xml
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="1440"
  height="900"
  viewBox="0 0 1440 900"
>
```

Chirpy는 post image를 lazy-loading `inline-flex` wrapper로 감싼다. SVG에 intrinsic width와 height가 없으면 VS Code Markdown preview에서는 보여도 실제 페이지에서는 wrapper와 image가 `0 × 0`으로 계산될 수 있다. `viewBox`를 함께 유지하면 명시한 원본 크기와 관계없이 본문 폭에 맞춰 반응형으로 축소된다.

이미지를 추가하거나 수정한 경우에는 다음을 확인한다.

1. `test -f` 또는 `git ls-files`로 참조 대상과 Git 포함 여부를 확인한다.
2. SVG는 `xmllint --noout <svg-path>`로 XML syntax를 검사한다.
3. VS Code preview만 보지 말고 production Jekyll HTML의 `<img>` 경로와 browser layout을 확인한다. 가능하면 headless browser나 실제 browser에서 light/dark theme와 mobile 폭을 함께 본다.
4. 게시 후 image URL이 HTTP 200과 올바른 `Content-Type`으로 응답하는지 확인하고, 실제 post에서 image가 0이 아닌 크기로 표시되는지 확인한다.

## 검증

포스트 변경에는 최소한 다음 검증을 수행한다.

### 1. Markdown lint

```bash
npx --yes markdownlint-cli2@0.20.0 '_posts/<post-file>.md'
```

### 2. Front matter와 수식 delimiter

Node 환경만 있을 때 front matter는 다음처럼 parse할 수 있다.

```bash
sed -n '2,/^---$/p' '_posts/<post-file>.md' | sed '$d' | npx --yes js-yaml >/dev/null
```

Display math delimiter는 indentation까지 허용해 센다.

```bash
post_path='_posts/<post-file>.md'
math_delimiters=$(rg -c '^[[:space:]]*\$\$$' "$post_path")
test "$((math_delimiters % 2))" -eq 0
```

수식이 없는 문서에서는 `rg`가 exit code 1을 반환할 수 있으므로 해당 검사는 생략하거나 0을 허용한다.

Inline 수식 안에 Markdown table로 오인될 raw `|`가 남아 있는지도 검사한다.

```bash
if rg -n '\$[^$]*\|[^$]*\$' "$post_path"; then
  echo 'raw | found inside inline math; use \lvert and \rvert' >&2
  exit 1
fi
```

### 3. Jekyll build와 HTML 검사

Ruby와 Bundler가 준비된 환경에서는 CI와 같은 검사를 실행한다.

```bash
JEKYLL_ENV=production bundle exec jekyll b
bundle exec htmlproofer _site \
  --disable-external \
  --ignore-urls '/^http:\/\/127\.0\.0\.1/,/^http:\/\/0\.0\.0\.0/,/^http:\/\/localhost/'
```

CI는 Ruby 3.4를 사용한다. 로컬에 Ruby, Bundler 또는 Docker가 없다면 build를 통과했다고 추정하지 않는다.

- 사용자가 publish를 요청하지 않았다면 검증을 위해 임의로 push하지 않는다.
- publish가 요청됐다면 로컬 lint를 먼저 통과시키고, push 후 GitHub Pages workflow의 Jekyll build와 html-proofer 결과를 확인한다.
- 검증 도구가 없으면 무엇을 실행하지 못했는지 최종 보고에 명시한다.

## Git, commit, push

Commit과 push는 사용자가 명시적으로 요청한 경우에만 수행한다.

1. `git status -sb`와 diff로 작업 범위를 확인한다.
2. `git add -A` 대신 요청된 파일만 explicit path로 stage한다.
3. `git diff --cached --check`, `git diff --cached --stat`, `git diff --cached --name-status`를 확인한다.
4. Conventional Commit 형식을 사용한다. 포스트 게시에는 `docs: publish <topic> guide`와 같은 짧은 message를 권장한다.
5. push 전에 현재 branch와 remote tracking 상태를 확인한다.
6. 사용자가 직접 게시를 요청한 경우에만 default branch에 push한다. PR이 요청된 경우 별도 branch를 만든다.

이 저장소의 commit hook는 `npx --no -- commitlint`를 사용한다. `node_modules`가 없어 hook를 실행할 수 없다면 hook를 바로 우회하지 말고 다음처럼 lockfile과 manifest를 변경하지 않는 방식으로 의존성을 준비한다.

```bash
npm install --no-save --package-lock=false
```

설치 후 `git status -sb`로 예상하지 않은 tracked file 변경이 없는지 다시 확인한다.

## Push 후 Pages 확인

이 저장소는 theme metadata 때문에 `gh`가 upstream인 `cotes2020/jekyll-theme-chirpy`를 현재 repository로 잘못 선택할 수 있다. GitHub CLI를 사용할 때는 항상 repository를 명시한다.

```bash
gh run list \
  -R Molly-Dooker/molly-dooker.github.io \
  --workflow pages-deploy.yml \
  --branch master \
  --limit 5
```

새 run ID를 확인한 뒤 build와 deploy가 끝날 때까지 기다린다.

```bash
gh run watch <run-id> \
  -R Molly-Dooker/molly-dooker.github.io \
  --exit-status
```

성공 조건은 다음과 같다.

- Jekyll `Build site` 성공
- `htmlproofer`를 실행하는 `Test site` 성공
- Pages artifact upload 성공
- `Deploy to GitHub Pages` 성공
- 실제 post URL이 HTTP로 열리고 새 title이 확인됨
- local `HEAD`와 remote branch SHA가 일치함
- 최종 `git status -sb`가 clean임

Workflow가 실패하면 log를 확인하고 원인을 수정해 다시 검증한다. 성공 여부를 확인하지 않은 채 “배포 완료”라고 보고하지 않는다.

## 최종 보고

작업 완료 시 다음 중 실제로 수행한 항목만 간결하게 보고한다.

- 생성·수정한 파일의 절대 경로 link
- 문서의 핵심 변경 내용
- 실행한 lint와 build 결과
- commit hash와 branch
- push 대상
- GitHub Actions run link와 결론
- 실제 게시 URL
- 수행하지 못한 검증 또는 남은 작업
