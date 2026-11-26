# 모델·데이터 준비와 서버 환경

`git clone`으로 받는 것은 실행 코드, 실험 설계, 설정, 입력의 목록·해시, 사전 점검 기록이다. **강도 선정에 필요한 모델 가중치와 원본 이미지는 원본 서버에서 새 서버로 한 번 복사한다.** 아래 절차는 GitHub에 데이터를 업로드하지 않는다.

| 자료 | 새 서버에서 확보하는 방법 |
| --- | --- |
| 실행 코드, 강도표, 학습 설정, 실험 설계 | GitHub에서 clone |
| 분류 입력 | 원본 서버에서 MobileNetV3-Large·ResNet18의 시작 가중치(C₀)와 CIFAR-100 공식 train·test·meta 전송 |
| 생성 입력 | 원본 서버에서 ImageNet·Caltech256의 U-Net 및 SiT 가중치 4개, StyleSwin 가중치 1개, SiT VAE, 등록된 모델 소스 전송 |
| 생성 학습·보정 이미지 | ImageNet 훈련 목록 25,000건, Caltech256 훈련 목록 20,713건, 도메인별 보정 자료, FFHQ 훈련 PNG 60,000장 전송. 분할·모델 등록·검증 기록도 같은 목록으로 전송 |
| 이 서버에서 만들 결과 | 훈련 중간 상태, C_T·W_T, 디코더, 검증 캐시와 선정표는 실행하면서 서버별 `LERRA_OUTPUT` 아래에 생성 |
| 후속 미세조정(FT) | FGVC-Aircraft와 ArtBench-10은 해당 단계 전에 별도 준비. 현재 실행 범위는 강도 선정까지 |

## 1. 새 서버에서 환경과 경로 준비

원본 환경은 **Ubuntu 22.04.5 LTS x86-64, Python 3.10.20, RTX 4090 24 GiB × 4, NVIDIA 드라이버 535.309.01, PyTorch 2.5.1·torchvision 0.20.1의 CUDA 12.1 빌드, cuDNN 9.1**이다. 새 서버에서는 한 장 이상의 호환 NVIDIA GPU를 사용할 수 있다. 병렬 실행기는 할당된 물리 GPU마다 워커를 한 개 실행한다. 기본 시작 조건은 그 카드에 다른 계산 프로세스가 없고 여유 VRAM이 20 GiB 이상인 경우다. 이 값은 워커의 시작 기준이며 실제 사용량은 모델과 실행 단계에 따라 달라진다.

준비할 도구는 Bash, Git, Python 3.10과 venv, NVIDIA 드라이버와 `nvidia-smi`다. 서버 간 전송에는 양쪽의 `rsync`와 원본 서버에서 새 서버로 접근할 수 있는 SSH 계정이 필요하다. Slurm에서는 GPU 점검과 학습을 GPU가 할당된 작업 셸에서 실행한다.

새 서버에서 저장소를 처음 clone한 뒤 아래 명령은 모두 `LessMark/`에서 실행한다.

```bash
git clone https://github.com/leejwian/LessMark.git
cd LessMark
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==2.2.6 setuptools==79.0.1 wheel
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

위 조합은 원본의 직접 의존성을 재현한다. `pyldpc`가 빌드 중에 NumPy를 가져오므로 NumPy를 먼저 설치한다. 다른 GPU의 호환성은 설치 후 `./lessmark check-env`로 확인한다. GPU 이름과 라이브러리 버전은 실험 결과에서 조건별로 기록된다.

입력은 총 **106,865개 파일, 92,355,404,562바이트(86.0 GiB)**다. 훈련 결과·캐시와 가상환경은 추가 공간을 사용한다. 아래 예시의 `/scratch`는 SSH 계정에 쓰기 권한이 있는 볼륨으로 바꿔도 된다. GPU 작업에서 같은 경로를 읽고 쓸 수 있어야 한다.

```bash
mkdir -p /scratch/lerra-inputs /scratch/lerra-runs
df -h /scratch/lerra-inputs /scratch/lerra-runs
cp server.env.example server.env
```

`server.env`는 이 서버에만 적용하는 Bash 설정이다. 예제의 세 경로를 실제 환경에 맞춘다.

| 설정 | 예시 | 용도 |
| --- | --- | --- |
| `LERRA_PYTHON` | `"$PWD/.venv/bin/python"` | clone 폴더 안에 설치한 실험용 인터프리터 |
| `LERRA_DATA_ROOT` | `/scratch/lerra-inputs` | `lerra_parameter_watermark_protocol/`을 담는 부모 폴더 |
| `LERRA_OUTPUT` | `/scratch/lerra-runs/experiment_2_strength_selection` | 이 서버에서 시작하는 실험의 가중치·캐시·결과 저장 위치 |

## 2. 원본 서버에서 등록 입력 전송

원본 서버의 이식본 폴더에서 실행한다. `target`은 실제 SSH 계정과 호스트로, `target_data`는 새 서버에서 정한 `LERRA_DATA_ROOT`로 바꾼다. 설정 파일이 없는 원본 서버에서 입력 루트 기본값은 `/data`다. 원본 서버에도 `server.env`를 만들었다면 `LERRA_DATA_ROOT=/data`와 원본 서버의 Python 경로를 지정한다. 입력은 실험 계정으로 읽을 수 있어야 한다.

```bash
cd ~/LERRA-experiment-2
target=user@new-server
target_data=/scratch/lerra-inputs
./lessmark copy-inputs --dry-run "$target:$target_data/"
./lessmark copy-inputs "$target:$target_data/"
```

첫 명령은 파일 복사 계획을 확인한다. 두 번째 명령은 등록 입력 목록에 따라 `rsync`로 복사하며 같은 명령으로 중단된 전송을 이어갈 수 있다. 새 서버에는 `/scratch/lerra-inputs/lerra_parameter_watermark_protocol/` 구조가 생긴다. 현 서버에서 진행 중인 실험의 중간 가중치와 선정 결과는 이 목록에 포함되지 않는다.

## 3. 새 서버에서 검증하고 시작

새 서버의 `LessMark/`에서 실행한다. `check-env`는 패키지 버전을 확인하고 보이는 GPU에서 짧은 CUDA 연산을 수행한다. `inputs --verify`는 등록된 파일의 SHA-256을 대조하므로 약 86 GiB를 읽는다. `verify`는 모델 가중치·데이터 분할·보정 사진과 210개 강도 후보가 실험 설계와 일치하는지 확인한다.

```bash
./lessmark check-env
./lessmark inputs --verify
./lessmark verify
./lessmark parallel --foreground --check
```

새 서버에서 일곱 조건 전체를 독립적으로 수행하려면 점검이 끝난 뒤 `./lessmark parallel --foreground`를 실행한다. 두 서버가 일을 분담한다면 양쪽에 서로 다른 조건을 지정한다. 재개할 때는 같은 코드·데이터 경로·수치 환경·결과 경로를 사용한다. [실행·재개와 결과 확인](../README.md)

## 후속 미세조정용 데이터

분류 FT에는 FGVC-Aircraft 2013b 원본 이미지와 공식 레이블을 준비한다. `variants.txt`의 기종 순서와 `images_variant_train.txt`, `images_variant_val.txt`, `images_variant_test.txt`의 공식 분할(3,334 / 3,333 / 3,333장)을 보관한다. [FGVC-Aircraft 공식 배포](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

생성 FT에는 ArtBench-10의 공식 256×256 이미지를 준비한다. 훈련 50,000장 가운데 화풍별 500장씩을 검증용으로 나누고 공식 테스트 10,000장은 보류한다. 실제 45,000 / 5,000장 분할 ID와 공통 FT 학습량은 FT 시작 전에 정한다. [ArtBench-10 공식 배포](https://github.com/liaopeiyuan/artbench) · [후속 평가 설계](../experiment_2/EXPERIMENT_DESIGN_KO.md#section-6)
