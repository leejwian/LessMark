# LessMark 실험 2

같은 256비트 메시지를 7개 모델·데이터셋 조건에 삽입하고, 6개 워터마크 방법마다 5개 강도를 비교한다. 현재 실행 범위는 **210개 후보의 학습·검증과 42개 조합별 강도 선정**이다. [실험 요약](experiment_2/EXPERIMENT_SUMMARY_KO.md) · [실험 설계](experiment_2/EXPERIMENT_DESIGN_KO.md)

| 무엇을 옮기는가 | 어디서 받는가 |
| --- | --- |
| 실행 코드, 강도표, 학습 설정, 데이터 분할·해시, 사전 점검 기록 | GitHub에서 `git clone` |
| 시작 가중치(C₀) 7개, SiT VAE, 등록된 사진·모델 소스·검증 자료 | 원본 서버에서 새 서버로 `./lessmark copy-inputs` 실행. **106,865개 파일, 86.0 GiB** |
| 진행 중인 학습의 중간 체크포인트, 후보 가중치(W_T), 공통 대조군(C_T) | 각 서버가 자신의 `LERRA_OUTPUT`에 생성. 입력 전송 대상에서 제외 |
| 선정 후 가지치기·양자화·미세조정 | 설계와 결과 인계 규격을 수록. 후속 실행기와 FGVC-Aircraft·ArtBench-10은 별도 준비 |

## 어느 서버에서 무엇을 실행하는가

원본 학습 환경은 **Ubuntu 22.04 x86-64, RTX 4090 24 GiB × 4, Python 3.10, PyTorch 2.5.1과 CUDA 12.1**이다. 새 서버는 호환되는 NVIDIA GPU 한 장부터 사용할 수 있다. 병렬 워커의 기본 시작 기준은 카드당 여유 VRAM 20 GiB다. 입력 86 GiB 외에 가중치와 캐시를 쓸 결과 볼륨도 필요하다.

1. **새 서버에서 준비한다.** 아래 명령으로 코드를 받고, 준비 안내에 따라 Python 환경과 입력·결과 디렉터리를 만든다. `server.env`에서 이 서버의 Python, 데이터 루트, 결과 경로, GPU를 지정한다. [설치 명령과 설정 예시](docs/assets.md#1-새-서버에서-환경과-경로-준비)

   ```bash
   git clone https://github.com/leejwian/LessMark.git
   cd LessMark
   ./lessmark help
   ```

2. **원본 서버에서 입력을 보낸다.** 원본 서버의 이식본 폴더(`~/LERRA-experiment-2`)에서 아래 명령을 실행한다. `user@new-server`와 목적지는 실제 SSH 계정·호스트와 새 서버의 `LERRA_DATA_ROOT`로 바꾼다. 첫 명령은 복사 예정 파일 수와 용량을 확인하고 두 번째 명령은 전송한다. [전송 절차](docs/assets.md#2-원본-서버에서-등록-입력-전송)

   ```bash
   ./lessmark copy-inputs --dry-run user@new-server:/scratch/lerra-inputs/
   ./lessmark copy-inputs user@new-server:/scratch/lerra-inputs/
   ```

3. **새 서버에서 점검하고 학습한다.** GPU를 할당받은 작업 셸에서 실행한다. `check-env`는 실제 CUDA 연산을, `inputs --verify`는 등록 입력의 해시를, `verify`는 실험 설계와 입력의 일치 여부를 검사한다.

   ```bash
   ./lessmark check-env
   ./lessmark inputs --verify
   ./lessmark verify
   ./lessmark parallel --foreground --check
   ./lessmark parallel --foreground
   ```

`parallel --foreground`는 이 서버에서 일곱 조건 전체를 실행한다. 사용자 systemd가 있는 서버에서는 `./lessmark parallel`로 백그라운드 서비스를 시작할 수 있다. 데이터 없이 후보표만 보려면 저장소 루트에서 `./lessmark plan`을 실행한다.

## 두 서버에서 나눠 실행하기

분담할 때는 **모델·데이터셋 조건 단위로** 두 서버의 담당을 정하고, 각각 `run --condition`을 사용한다. 기존 서버의 전체 `parallel`을 계속 두면 그 서버도 일곱 조건을 모두 가져가 중복 계산할 수 있다. 지정할 수 있는 조건 ID와 210개 강도값은 `./lessmark plan`이 출력한다.

```bash
# 새 서버가 ResNet18/CIFAR-100 한 조건(6개 방법 × 5개 강도)을 담당하는 예시
./lessmark run --condition resnet18_cifar100 --device cuda:0
```

`LERRA_GPU_IDS`는 병렬 워커가 쓸 물리 GPU를 고른다. 단일 `run`에서는 할당 환경의 `CUDA_VISIBLE_DEVICES`와 그 안의 논리 장치 번호(`--device`)를 사용한다. 서버마다 결과 폴더를 따로 둔다. 이식본은 원본 서버의 실행 코드와 해시가 달라 새 결과 폴더에서 시작한다. 이식본으로 시작한 조건은 같은 코드·입력 경로·수치 환경·출력 경로로 재실행하면 마지막 저장 지점에서 이어진다. 사용한 GPU와 라이브러리 버전은 조건별 `environment.json`에 기록한다.

## 결과 확인

```bash
./lessmark status --clean
./lessmark report --verify-artifacts
```

`LERRA_OUTPUT/candidates.csv`는 완료 후보의 BER와 품질을 담는다. `selected.csv`는 조합별 선정 강도와 상태를, `selection.json`은 후속 평가에 전달할 W_T·C_T·복호 자료의 경로와 해시를 담는다. 오류는 기본으로 로컬에 기록하며 이메일 알림은 `server.env.example`에서 설정할 수 있다. [산출물 안내](experiment_2/results/README.md)

## 폴더와 코드

| 경로 | 이곳에서 확인할 내용 |
| --- | --- |
| [lessmark](lessmark) · [server.env.example](server.env.example) · [requirements.txt](requirements.txt) | 공통 실행 명령, 서버별 경로·GPU 설정, Python 패키지 버전 |
| [docs/](docs/) · [scripts/](scripts/) | 서버 준비 안내와 그 절차에 쓰는 환경 검사·입력 목록·전송 코드 |
| [experiment_2/config/](experiment_2/config/) | 원본 설계 JSON, 후보 강도표, 옵티마이저·시드, 분할·검증 근거 |
| [experiment_2/code/](experiment_2/code/) | 강도 선정, 분류·생성 학습, 워터마크 삽입·복호, 설계 검증. [코드별 역할](experiment_2/README.md) |
| [experiment_2/parallel_strength/](experiment_2/parallel_strength/) | GPU 배정, 워커, 진행 상태 점검 |
| [experiment_2/results/](experiment_2/results/) | 원본 가중치 사전 점검 기록과 산출물 형식. 실제 학습 결과 위치는 서버별 `LERRA_OUTPUT` |
| [runtime/](runtime/) · [experiments/](experiments/) | 재사용하는 모델·삽입·품질 평가 코드와 SiT 내부 판독 구현 |
| [experiment_1/](experiment_1/) · [tools/](tools/) | 원본 설계가 경로로 참조하는 이전 실험의 검증 기록과 방법 이름표 |
| [tests/](tests/) | 강도 선정, 재개, 워커 배정, 입력 전송 등에 대한 테스트. 실험 환경에서 `python -m pytest tests` 실행 |
