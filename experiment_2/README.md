# 실험 2 · LessMark 비교

**7개 모델·데이터셋 조건 × 6개 삽입 방법 = 42개 조합**이다. 다섯 아키텍처 가운데 U-Net과 SiT-B/2를 ImageNet·Caltech256에서 각각 실험하므로 분류 2개와 생성 5개 조건이 된다. 조합마다 다섯 강도를 적용한 **210개 후보 모두에 같은 메시지**를 삽입한다. 수신자 ID 292의 BIBD 기반 256비트로, 0과 1이 각각 128개다. [실험 2 설계서](EXPERIMENT_DESIGN_KO.md)

**대표강도**는 다섯 후보를 만드는 사전 기준값이다. 비교 방법 다섯 개는 일곱 조건 모두에 원 논문 값을 쓴다(VanillaMark 0.01, GreedyMark 0.01, NeuralMark 1, DeepMark 0.1, TATOOED 0.0009). LessMark의 기준 α는 두 분류기에 0.2, 두 U-Net에 0.1, 두 SiT와 StyleSwin에 1을 지정했다. 별도 예비 강도 탐색 없이 **대표강도의 ¼·½·1·2·4배**를 본실험 후보로 확정했다. [원 논문 설정과 적용 범위](EXPERIMENT_DESIGN_KO.md#original-embedding-settings) · [42개 조합별 후보표](config/FIVE_STRENGTH_CANDIDATES_REVIEW.csv)

고정 학습량 T는 분류 **추가 200에포크(140,400업데이트)**, 생성 **2,048업데이트**다. 원본 C₀에서 원래 과제 손실만으로 T까지 학습한 모델이 미삽입 대조군 C_T다. 학습형 네 방법은 같은 C₀와 같은 훈련 조건에서 각 강도에 해당하는 워터마크 손실을 더해 T까지 학습한다. 직접 삽입하는 LessMark와 TATOOED는 C_T의 복사본에 신호를 더하고 추가 업데이트 없이 판독한다. 분류 LessMark는 헤드를 교체해도 남는 백본 합성곱에 삽입한다. MobileNetV3-Large는 마지막 역잔차 블록의 1×1 투영층, ResNet18은 마지막 잔차 블록의 두 번째 3×3 합성곱이다. [삽입 방법과 대조군](EXPERIMENT_DESIGN_KO.md#section-4)

각 조합에서 정상 반환된 256비트의 **최종 BER가 가장 낮은 강도**를 선정한다. 동률이면 가장 큰 강도를 택하고, 최저 BER가 0보다 커도 선정한다. 다섯 판독이 모두 비정상이면 해당 조합은 미선정이다. 품질은 210개 후보 모두에서 대응 C_T와 비교해 별도 보고한다. [선정 규칙](EXPERIMENT_DESIGN_KO.md#section-5)

목표는 **42개 선정 모델**이다. 확보한 W_T와 대응 C_T를 삽입 직후 별도 테스트 입력으로 비교한 뒤, 각 쌍의 복사본으로 **30%·70% 가지치기, 8·4비트 양자화·역양자화, 새 도메인 미세조정(FT)**을 각각 독립적으로 평가한다. FT는 두 모델 모두 워터마크 항을 끄고 분류는 FGVC-Aircraft, 생성은 ArtBench-10에서 학습한다. 각 변형 후 W에서 같은 메시지의 BER를 읽고, 같은 변형을 적용한 C와 W의 top-1 정확도 또는 생성 RGB 1,000쌍의 평균 MSE·SSIM을 비교한다. [강건성 평가](EXPERIMENT_DESIGN_KO.md#section-6) · [평가 지표](EXPERIMENT_DESIGN_KO.md#section-7)

| 찾는 자료 | 위치 |
| --- | --- |
| 설정 요약 | [비교 실험 요약표](EXPERIMENT_SUMMARY_KO.md) |
| 실행 설정 | [설계 JSON](config/NEXT_EXPERIMENT_DESIGN.json) · [옵티마이저·배치·난수 설정](config/STRENGTH_TRAINING.json) · [FFHQ 방향 계산용 사진 목록](config/LESSMARK_FFHQ_TRAIN_1000.csv) |
| 코드 | [공통 실행 명령](../lessmark) · [강도 선정 코드](code/run_strength_selection.py) · [분류 학습](code/classification_backend.py) · [생성 학습](code/generation_backend.py) · [삽입·복호](code/strength_watermarks.py) · [설계 검증기](code/verify_design.py) |
| 결과 | [본실험 산출물과 예비 점검의 구분](results/README.md) · [출력 파일 목록](EXPERIMENT_DESIGN_KO.md#section-8) |

## 강도 선정 실행

서버별 환경 설정, 입력 복사, 실행·재개 명령은 [저장소 README](../README.md)에 있다. `plan`으로 210개 후보를 확인하고 `report --verify-artifacts`로 완료된 가중치와 복호 자료의 해시를 대조할 수 있다.

조건별 `clean/`에 공통 대조군 C_T가 저장된다. `candidates.csv`에서 각 후보의 종료 BER와 품질을, `selected.csv`에서 42개 조합의 상태와 선정 강도를 확인한다. 후속 평가에는 `selection.json`에 기록된 W_T·C_T·복호 자료를 사용한다. 다섯 판독이 모일 때까지 상태는 `pending`이며, 다섯 판독이 모두 비정상이면 `selection_failure`다.
