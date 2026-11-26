# 서버별 환경과 입력 파일 준비

## Python과 GPU

원래 환경은 Linux x86-64, Python 3.10.20, PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1, cuDNN 9.1, RTX 4090 24 GiB이다. 아래는 이 조합과 호환되는 NVIDIA 서버에서 저장소 루트를 기준으로 설치하는 예시다. `nvidia-smi`에서 대상 GPU와 드라이버를 먼저 확인한다. 시스템 Python이 3.10 계열이면 다음과 같이 가상환경을 만든다.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==2.2.6 setuptools==79.0.1 wheel
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python scripts/check_environment.py
```

`requirements.txt`는 학습과 복호에 쓰는 직접 의존성을 원 서버 버전으로 고정한다. pyldpc 0.7.9는 소스 패키지 빌드 중 NumPy를 읽으므로 먼저 NumPy와 빌드 도구를 설치한다. PyTorch와 torchvision의 CUDA 빌드는 두 패키지를 설치하는 명령에서 선택한다. GPU 세대나 드라이버가 이 빌드를 지원하지 않으면 대상 서버에 맞는 PyTorch 환경으로 별도 캠페인을 구성하고 버전을 기록한다. 실제 커널 연산과 검증 결과는 GPU와 라이브러리 버전에 따라 달라질 수 있다. 가상환경 경로는 저장소 루트의 `server.env`에 `LERRA_PYTHON=.venv/bin/python`으로 지정할 수 있다.

## 원 서버에서 입력 전송

입력 경로는 `LERRA_DATA_ROOT/lerra_parameter_watermark_protocol/` 아래의 상대 구조를 보존한다. 원 서버의 `LERRA_DATA_ROOT`는 `/data`다. 전송 목록은 106,865개 파일, 약 86.0 GiB다. 대상 디스크에는 이 입력과 별도로 학습 상태, 후보 가중치, 이미지 캐시를 저장할 공간이 필요하다.

| 입력 | 전송 범위 |
| --- | --- |
| 분류 | CIFAR-100 공식 Python train·test·meta, MobileNetV3-Large와 ResNet18의 C₀ |
| U-Net 및 SiT | ImageNet 학습 25,000장, Caltech256 학습 20,713장, 각 도메인 보정 목록과 이미지, 분할 기록, 네 C₀, SiT VAE |
| StyleSwin | FFHQ 원본 PNG 00000~59999, C₀, 보정에 쓰는 1,000장과 파일 해시 |
| 모델 구현과 근거 | Guided Diffusion, SiT, StyleSwin의 등록된 소스와 리비전, 이전 선정표 42개, 일곱 모델 계약 및 강건성 검증 기록 |

먼저 **새 서버**에서 쓸 디렉터리를 만든다. 예시 경로 `/scratch/lerra-inputs`를 실제 쓰기 가능한 경로로 바꾼다.

```bash
mkdir -p /scratch/lerra-inputs
```

**원 서버**의 이 저장소 루트에서 다음을 실행한다. `target`을 대상 SSH 계정과 호스트로 바꾼다. 파일 목록은 원본 등록 JSON과 해시에 따라 생성한다. `rsync`는 같은 명령으로 다시 실행해 중단된 전송을 마칠 수 있다.

```bash
target=user@new-server
target_data=/scratch/lerra-inputs
asset_list=$(mktemp)
python3 scripts/assets.py --root /data --files-from "$asset_list" &&
  rsync -aL --partial --protect-args --info=progress2 --files-from="$asset_list" /data/ "$target:$target_data/"
```

**새 서버**의 저장소 루트에서 검사한다. `--verify`는 등록된 SHA-256을 재계산하므로 약 86 GiB를 읽는다. 옵션을 생략하면 모든 파일의 존재 여부, 등록된 크기, 주요 목록 해시, Git 리비전을 빠르게 점검한다.

```bash
export LERRA_DATA_ROOT=/scratch/lerra-inputs
python3 scripts/assets.py --verify
python3 -B experiment_2/code/verify_design.py
```

이 경로를 저장소의 `server.env`에도 지정하면 실행기를 다시 호출할 때 같은 입력을 읽는다. 원본 JSON 안에 보이는 `/data/…`는 해시로 고정한 등록값이다. 프로그램이 읽는 시점에 새 서버의 경로에 대응시킨다. 실험 2의 진행 중인 산출물은 이 입력 목록에 포함되지 않는다. 출력 경로는 서버별로 지정한다.

## 선정 후 미세조정 자료

실험 2 설계는 분류 모델에 FGVC-Aircraft 2013b 원본 이미지와 공식 variant 레이블을 사용한다. `variants.txt`의 기종 순서와 `images_variant_train.txt`, `images_variant_val.txt`, `images_variant_test.txt`의 분할(3,334 / 3,333 / 3,333장)을 함께 보관한다. [FGVC-Aircraft 공식 배포](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

생성 모델에는 ArtBench-10의 공식 256×256 이미지를 사용한다. 훈련 50,000장 가운데 화풍별 500장씩을 검증에 할당하여 45,000 / 5,000장으로 나눈다. 공식 테스트 10,000장은 보류한다. 실제 분할 ID와 모델별 공통 FT 학습량은 원래 설계에 따라 FT 전에 한 번 확정해 두 서버에서 공유한다. 이 두 데이터셋은 현재 원 서버의 실험 입력 디렉터리에 없으므로 별도로 준비한다. [ArtBench-10 저자 배포](https://github.com/liaopeiyuan/artbench) · [실험 2 설계](../experiment_2/EXPERIMENT_DESIGN_KO.md)
