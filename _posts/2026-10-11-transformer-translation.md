---
title: "Transformer로 한국어 → 영어 번역 모델 학습하기"
description: "사전 학습 없이 Encoder-Decoder Transformer를 학습하며 토크나이저, 빔 서치, 데이터 규모, warmup을 조정한 과정을 정리했습니다."
date: 2026-10-11
category: AI Engineering
permalink: /2026-10-transformer-translation/
image: /assets/images/transformer-translation/bleu-experiment-progress.svg
---

Transformer는 Encoder가 입력 문장을 읽고 Decoder가 번역문을 생성하는 구조로 기계 번역에 사용할 수 있습니다. 이번 실습에서는 한국어 문장을 영어로 옮기는 모델을 **사전 학습 없이 처음부터** 만들었습니다. SentencePiece 토크나이저부터 Transformer 모델까지 직접 학습하고, 데이터와 학습률 스케줄을 바꾸며 번역 품질이 어떻게 달라지는지 살펴봤습니다.

실습 코드는 [TF05-Translator.ipynb](https://github.com/mkmkchoi/ai-model-study/blob/main/ipynb/TF05-Translator.ipynb)에서 확인할 수 있습니다. 노트북은 Google Colab과 Google Drive 경로를 사용하므로, 실행할 때는 데이터 파일과 폴더 경로를 본인 환경에 맞게 설정해야 합니다.

## Encoder와 Decoder를 함께 쓰는 번역 모델

Encoder는 입력된 한국어 토큰 전체를 처리해 문맥 표현을 만듭니다. Decoder는 그 표현을 참고하면서 영어 토큰을 왼쪽부터 하나씩 생성합니다. 학습할 때는 정답 영어 문장을 한 칸씩 밀어 Decoder 입력과 정답으로 나누는 **teacher forcing**을 사용합니다.

모델은 PyTorch의 `torch.nn.Transformer`를 바탕으로 구성했습니다. 한국어와 영어 토크나이저는 별도로 학습한 SentencePiece BPE이며, 어휘 크기는 각각 12,000과 16,000입니다. 사전 학습된 번역 모델이나 언어 모델의 가중치는 사용하지 않았습니다. 전체 구조는 Transformer 논문의 Encoder-Decoder 방식을 따릅니다.

## 번역 품질은 BLEU와 chrF로 확인

BLEU는 모델 번역과 기준 번역 사이의 단어 단위 n-gram 겹침을 계산하고, chrF는 문자 단위 겹침을 봅니다. 두 지표 모두 높을수록 기준 문장과 더 많이 겹쳤다는 뜻이지만, 표현이 다른 올바른 번역이나 문장의 의미를 완전히 평가하지는 못합니다. 특히 BLEU는 한 문장보다 여러 문장을 모은 말뭉치 단위 평가에 맞는 지표입니다. 따라서 이 실습에서는 고정된 1,000개 문장을 번역해 점수를 계산했고, 다른 데이터셋의 점수와 직접 비교하지 않도록 주의해야 합니다.

Google Cloud 문서는 BLEU 30~40을 대략 “이해할 수 있는 수준에서 좋은 번역”으로 소개합니다. 이는 일반적인 참고 구간일 뿐이며, 실제 점수는 데이터셋, 토큰화 방식, 평가 도구와 설정에 따라 달라집니다.

## 초기 실험: 큰 모델보다 구조와 토크나이저

처음에는 Encoder와 Decoder를 각각 두 층으로 두고 `d_model=1024`로 넓게 구성했지만 BLEU는 2.17에 머물렀습니다. 학습 Epoch만 50에서 100으로 늘린 실험도 2.86에 그쳤습니다. 직접 만든 단순 토크나이저를 SentencePiece로 바꾸자 BLEU가 3.73으로 올랐고, 이후 모델을 더 깊고 좁게 바꾸며 개선됐습니다.

| 실험 | Encoder / Decoder | 주요 변경 | BLEU |
| --- | ---: | --- | ---: |
| 초기 구성 | 2 / 2 | `d_model=1024`, 단순 토크나이저 | 2.17 |
| 토크나이저 변경 | 2 / 2 | SentencePiece 적용 | 3.73 |
| 층을 늘리고 차원 조정 | 4 / 4 | `d_model=512`, 정규화 강도 낮춤 | 7.12 |
| Transformer base 구성 | 6 / 6 | `d_model=512`, head 8개, FFN 2048 | 11.75 |

첫 구성은 파라미터가 약 1억 2,300만 개였고, base 구성은 약 6,700만 개였습니다. 이 실험에서는 파라미터 수를 키우는 것보다 번역에 맞는 구조와 토크나이저를 선택하는 일이 더 중요했습니다. 다만 여러 설정을 함께 바꾼 단계가 있어, 각 변경의 효과를 완전히 분리한 비교는 아닙니다.

## Beam Search로 문장 전체를 고려하기

Greedy 디코딩은 매 단계에서 가장 확률이 높은 토큰 하나를 바로 선택합니다. Beam Search는 여러 후보 문장을 남겨두고 다음 토큰을 이어서 계산하므로, 첫 단어의 확률이 조금 낮더라도 문장 전체로 더 나은 후보를 찾을 수 있습니다. 후보가 길어질수록 누적 확률이 불리해지는 점은 길이 페널티로 보정합니다.

`beam_size=4`, 길이 페널티 계수 `alpha=0.4`에서 BLEU 14.08, chrF 41.88을 기록했습니다. Greedy 결과의 BLEU 11.75, chrF 41.00보다 높았습니다. Beam 크기를 6으로 늘린 실험은 이보다 개선되지 않아, 후보를 많이 유지하는 것만으로 성능이 계속 오르지는 않았습니다.

## 학습 데이터를 약 10배 늘리기

첫 데이터는 AI Hub의 학술 분야 한·영 병렬 말뭉치에서 가져온 244,032개 문장 쌍이었습니다. 여기에 일상생활·구어체 병렬 말뭉치를 추가하면서 한국어 문장 중복을 제거하고, 검증 문장이 학습 데이터에 들어가지 않도록 확인했습니다. 학습 데이터는 2,394,058개 문장 쌍으로 약 9.8배 늘었습니다.

같은 모델 구성에서 데이터 증량 전후를 비교했을 때 BLEU는 14.02에서 24.04로, chrF는 41.50에서 49.62로 상승했습니다. 모델 크기나 디코딩 설정을 조정했을 때보다 데이터 증가가 큰 폭의 개선과 함께 나타났습니다.

## 데이터가 바뀌면 Learning Rate Scheduler도 다시 봐야

데이터가 늘어난 뒤에는 Transformer 방식의 warmup 스케줄도 함께 점검했습니다. 이 스케줄은 Epoch이 아니라 학습 Step 수를 기준으로 동작합니다. 데이터가 약 10배가 되면서 1 Epoch의 Step도 약 3,800에서 37,400으로 증가했고, 기존 `warmup_steps=4,000` 설정에서는 Learning Rate가 Epoch 기준으로 훨씬 빠르게 낮아졌습니다.

그래서 warmup을 40,000 Step으로 늘려 Epoch 단위 학습률 감소 속도를 기존과 비슷하게 맞췄습니다. 노트북 기록에서 BLEU 최고점은 47 Epoch의 30.35였고, chrF 최고점은 49 Epoch의 56.28이었습니다. 두 최고점은 서로 다른 Epoch에서 나왔습니다. Google 문서의 대략적인 기준으로는 BLEU 30~40 구간에 해당하지만, 이 결과는 해당 실습의 1,000개 평가 문장과 설정에 한정됩니다.

<figure>
  <img src="{{ '/assets/images/transformer-translation/bleu-experiment-progress.svg' | relative_url }}" alt="번역 모델 실험 단계에 따라 BLEU 점수가 2.17에서 30.35까지 오른 결과">
  <figcaption>단계별 실험에서 기록한 BLEU 점수입니다. 모델 구성, 토크나이저, 데이터와 학습 스케줄이 차례로 달라진 결과이므로 각 막대를 단일 요인의 효과로 해석할 수는 없습니다.</figcaption>
</figure>

## 정리

이번 실습에서는 Encoder와 Decoder를 모두 사용하는 Transformer 번역 모델을 사전 학습 없이 만들었습니다. SentencePiece와 base 구조, Beam Search를 적용한 뒤에도 BLEU 14점대에 머물렀지만, 학습 데이터를 약 10배 늘리고 데이터 규모에 맞춰 warmup Step을 조정한 결과 BLEU 30.35까지 올랐습니다.

가장 크게 배운 점은 데이터 규모가 달라지면 모델만 보는 것으로 충분하지 않다는 것입니다. Step 기준 Learning Rate Scheduler는 데이터 양과 Epoch당 Step 수의 변화에 맞춰 다시 확인해야 합니다. 또한 BLEU와 chrF는 기준 번역과의 겹침을 보는 지표이므로, 숫자·약어 처리나 의미 전달을 확인하려면 실제 번역 사례도 함께 살펴봐야 합니다.

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Transformer 논문.
- Google Cloud, [The BLEU translation quality metric](https://docs.cloud.google.com/translate/docs/bleu-scores), BLEU 계산과 해석 및 한계.
- AI Hub, [분야별 한국어-영어 번역 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71749).
- AI Hub, [일상생활 및 구어체 한-영 번역 병렬 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71265).
- [실습 코드: TF05-Translator.ipynb](https://github.com/mkmkchoi/ai-model-study/blob/main/ipynb/TF05-Translator.ipynb).
