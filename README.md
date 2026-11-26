# LessMark 실험 2

일곱 모델·데이터셋 조건에서 여섯 워터마크 방법을 다섯 강도로 비교한다. 원본과 같은 256비트 메시지(ID 292), 후보표, 훈련·검증 분할, 학습량과 선정 규칙을 사용한다. 현재 실행기는 210개 후보를 학습·평가하고 조합별 최저 BER의 강도를 고른다. 동률이면 가장 큰 강도를 선택한다. [실험 요약](experiment_2/EXPERIMENT_SUMMARY_KO.md) · [실험 설계](experiment_2/EXPERIMENT_DESIGN_KO.md)

## 다른 서버에서 시작하기

Linux와 NVIDIA CUDA GPU를 전제로 한다. 원본 환경은 Python 3.10, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, RTX 4090이다. GPU 수, Python 경로, 데이터 루트, 결과 볼륨은 서버별로 지정할 수 있다.

```bash
git clone https://github.com/leejwian/LessMark.git
cd LessMark
cp server.env.example server.env
```

1. [환경 설치와 입력 전송](docs/assets.md)에 따라 Python 환경을 만들고 원본 서버에서 등록 입력을 옮긴다. 체크포인트, 생성 모델 소스·VAE, 훈련 사진 등 약 86 GiB가 필요하다. 학습 결과와 캐시 공간은 별도로 확보한다.
2. `server.env`에서 `LERRA_PYTHON`을 설치한 인터프리터로 지정한다. `LERRA_DATA_ROOT` 아래에는 `lerra_parameter_watermark_protocol/`이 있어야 한다. `LERRA_OUTPUT`에는 이 서버가 쓸 결과 디렉터리를 지정한다.
3. 아래 명령으로 후보표, 모델·입력 해시, GPU 배정을 확인한다. `plan`은 데이터와 GPU 없이도 실행할 수 있다. `verify`는 등록된 체크포인트와 실사진을 실제로 읽는다.

```bash
./experiment_2/run_strength_selection.sh plan
./experiment_2/run_strength_selection.sh verify
./experiment_2/run_strength_selection.sh parallel --foreground --check
```

## 실행과 재개

GPU 워커를 현재 작업 세션에서 실행한다. Slurm 등에서 GPU를 할당받으면 해당 작업 셸에서 실행한다. 셸을 계속 유지하거나 작업 스케줄러로 실행 시간을 확보한다.

```bash
./experiment_2/run_strength_selection.sh parallel --foreground
```

사용자 systemd를 이용하는 호스트에서는 `parallel`로 백그라운드 서비스를 시작한다. 한 조건만 분담할 때는 다음처럼 `run`의 조건을 지정한다. 같은 서버에서 같은 입력 마운트·출력 폴더·코드·수치 환경으로 다시 실행하면 마지막 저장 지점에서 이어진다.

```bash
./experiment_2/run_strength_selection.sh run --condition resnet18_cifar100
./experiment_2/run_strength_selection.sh status --clean
./experiment_2/run_strength_selection.sh report --verify-artifacts
```

독립 재현에는 두 서버 모두 `parallel`을 사용할 수 있다. 일을 나눌 때는 양쪽 서버에 겹치지 않는 조건을 `run --condition`으로 지정한다. 위의 한 조건 실행 예시는 보이는 기본 CUDA 장치를 사용하며, 배치 시스템의 `CUDA_VISIBLE_DEVICES` 또는 `run --device cuda:0`으로 장치를 지정할 수 있다. 기존 서버의 `parallel`은 일곱 조건을 모두 가져가므로, 계속 실행해 두면 새 서버에서 같은 조건을 지정했을 때 중복 계산이 생길 수 있다.

서버마다 별도 결과 디렉터리를 쓴다. 이식본은 실행 코드 해시가 달라 기존 서버의 진행 중인 `plan.json` 및 중간 체크포인트와 별도 실행으로 관리한다. GPU와 라이브러리 버전은 조건별 `environment.json`에 기록된다. 서로 다른 하드웨어에서는 결과에 수치 차이가 생길 수 있다.

오류 알림의 기본값은 로컬 기록이다. Gmail 알림을 사용하려면 `server.env.example`의 SMTP 항목과 본인만 읽을 수 있는 앱 비밀번호 파일을 설정한다. 전송 기록은 결과 폴더 옆의 `<출력 폴더>.alerts/notifications/`에서 확인한다.

## 선정 결과와 후속 설계

`LERRA_OUTPUT`의 `candidates.csv`에는 완료 후보의 BER와 품질이 담긴다. `selected.csv`와 `selection.json`에는 42개 조합의 상태와 선정된 W_T, 공통 대조군 C_T, 복호 자료의 참조가 담긴다. 결과 경로와 파일별 의미는 [산출물 안내](experiment_2/results/README.md)에서 확인할 수 있다.

후속 단계는 선정된 쌍의 원래 도메인 테스트, 30%·70% 가지치기, 8·4비트 양자화, FGVC-Aircraft·ArtBench-10 미세조정이다. 이번 배포에는 이 단계의 설계와 입력 인계 규격이 들어 있다. 후속 실행기 구현과 FT 학습량·ArtBench 분할 확정은 이후 단계다. 설계서에서 인용하는 실험 1의 과거 보고서·보관 기록 일부는 원본 LERRA 작업공간에서 볼 수 있다. [후속 평가 설계](experiment_2/EXPERIMENT_DESIGN_KO.md#section-6) · [실행 설정 원본](experiment_2/config/NEXT_EXPERIMENT_DESIGN.json)
