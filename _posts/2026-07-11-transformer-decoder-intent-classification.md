---
title: "Transformer Decoder로 문장 의도 분류 학습하기"
description: "Transformer의 Encoder·Decoder 차이에서 출발해 GPT의 Decoder-only 구조, Causal Mask 구현, 다음 토큰 예측을 더한 문장 의도 분류 실험을 정리했습니다."
date: 2026-07-11
category: AI Engineering
permalink: /2026-07-transformer-decoder-intent-classification/
image: /assets/images/transformer-decoder-intent-classification/transformer-decoder-gpt-architecture.png
---

문장 의도 분류는 문장을 읽고 사용자의 요청이나 대화 목적을 하나의 클래스로 예측하는 문제입니다. 이번 실습에서는 Transformer의 Encoder와 Decoder가 정보를 다루는 방식의 차이에서 출발해, GPT처럼 **Decoder만 사용하는 구조**로 문장의 의도를 분류해 보았습니다.

실제 구현은 아래 노트북에서 확인할 수 있습니다.

- [기본 Decoder 의도 분류 실습](https://github.com/mkmkchoi/ai-model-study/blob/main/ipynb/TF04-Intent-Classification-Decoder.ipynb)
- [다음 토큰 예측을 추가한 GPT-style 실습](https://github.com/mkmkchoi/ai-model-study/blob/main/ipynb/TF04-Intent-Classification-Decoder-gpt-style.ipynb)

## 1. Encoder, Decoder 그리고 GPT

Transformer에서 Encoder는 입력 토큰 전체를 분석해 문맥 표현을 만들고, Decoder는 이전에 생성한 토큰과 Encoder의 결과를 바탕으로 다음 출력을 순차적으로 만듭니다. 대표적으로 BERT는 Encoder 기반 모델이고, GPT는 Decoder 기반 모델로 알려져 있습니다.

대표 이미지의 왼쪽은 Encoder, 오른쪽은 Decoder입니다. Encoder는 입력 전체에 Self-Attention을 적용하고, Decoder는 이전 토큰만 참고하는 Masked Self-Attention을 먼저 적용한 뒤 Encoder의 결과를 받습니다. GPT는 입력 Encoder를 사용하지 않기 때문에, 그림의 빨간 박스로 표시한 Encoder-Decoder Attention 부분을 제거하고 Decoder만 사용합니다.

## 2. GPT 논문에서 본 분류 모델

GPT-1 논문은 단방향 Transformer Layer 위에 분류를 포함한 여러 Task용 헤드를 붙여 실험했습니다.

<figure>
  <img src="{{ '/assets/images/transformer-decoder-intent-classification/gpt-task-heads.png' | relative_url }}" alt="GPT 논문의 Transformer 기반 과제별 출력 헤드 구성">
  <figcaption>왼쪽은 GPT의 Transformer 구조, 오른쪽은 분류·문장 관계·유사도 등 과제별 출력 구성입니다. 출처: Radford et al., <a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf">Improving Language Understanding by Generative Pre-Training</a>.</figcaption>
</figure>

그중 첫 번째는 분류 모델입니다. SST-2(Stanford Sentiment Treebank-2) 영화 리뷰 문장을 활용해 긍정·부정의 감성을 분류했습니다. 논문에서는 사전 학습 모델을 활용해 91.3% 정확도를, 사전 학습 없이 학습한 조건에서는 84% 정확도를 보고했습니다.

학습 데이터와 분류 클래스 수는 이번 실습과 다르지만, Decoder-only 구조를 분류에 적용한다는 점에서 참고할 수 있는 사례입니다.


## 3. Causal Mask로 GPT 방식 만들기

PyTorch에서는 Encoder와 Decoder를 각각 `torch.nn.TransformerEncoderLayer`, `torch.nn.TransformerDecoderLayer`로 제공합니다. GPT는 Decoder 계열 모델이지만, 구현에서는 `TransformerEncoderLayer`에 Causal Mask를 전달하는 방법으로도 만들 수 있습니다. 두 블록의 핵심 구성은 비슷하고, GPT에서 중요한 차이는 미래 토큰을 가리는 **Masked Multi-Head Self-Attention**이기 때문입니다.

그래서 `MaskedTransformerClassifier` Class에 다음과 같이 마스크를 추가했습니다.

```python
def _generate_causal_mask(self, seq_len, device):
    return torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )

causal_mask = self._generate_causal_mask(seq_len, input_ids.device)
x = self.encoder(x, mask=causal_mask, src_key_padding_mask=(attention_mask == 0))
```

상삼각 영역의 `True`는 마스킹할 위치입니다. Attention을 계산할 때 뒤에 오는 토큰을 볼 수 없도록 만드는 장치입니다.

<figure>
  <img src="{{ '/assets/images/transformer-decoder-intent-classification/causal-mask-example.png' | relative_url }}" alt="3개 토큰 시퀀스에 적용된 Causal Mask 행렬">
  <figcaption>대각선 위쪽이 `True`인 Causal Mask 예시. 미래 위치의 Attention 점수는 계산에서 제외됩니다.</figcaption>
</figure>

예를 들어 `['내', '가', '가방', '에']`가 입력되면 각 위치는 다음처럼 계산됩니다.

- `내`: 자신만 보고 다음 토큰을 예측합니다.
- `가`: `내`, `가`까지만 봅니다.
- `가방`: `내`, `가`, `가방`까지만 봅니다.

문장을 통째로 입력하더라도 Causal Mask 덕분에 각 위치는 과거 토큰만 바탕으로 다음 토큰을 예측합니다. 따라서 토큰을 하나씩 덧붙이지 않고도 문장 전체를 한 번에 학습할 수 있습니다.

분류에는 패딩을 제외한 마지막 유효 토큰의 출력을 사용하고, LayerNorm과 Linear 분류 헤드에 통과시켰습니다.

## 4. Decoder 의도 분류 결과

Decoder로 문장 의도 분류 모델을 학습한 결과, 정확도 **0.8304**를 기록했습니다. Feed Forward Layer의 차원을 늘렸을 때 성능이 소폭 개선되는 것도 확인했습니다.

<figure>
  <img src="{{ '/assets/images/transformer-decoder-intent-classification/decoder-classification-result.png' | relative_url }}" alt="Transformer Decoder 의도 분류 모델의 정확도 0.8304 결과">
  <figcaption>Decoder 의도 분류 테스트 결과: 정확도 0.8304.</figcaption>
</figure>

GPT 논문에서 사전 학습 없이 SST-2 분류에 보고한 84%와 비슷한 수준의 수치이기도 합니다. 다만 데이터와 분류 클래스 수가 다르므로, 두 결과를 직접 비교할 수는 없습니다.

## 5. 다음 토큰 예측을 함께 학습하기

GPT 논문에서는 분류용 MLP Layer를 제거하고, 언어 생성만으로도 분류를 수행할 수 있는지 확인한 실험도 소개했습니다. SST-2 문장 끝에 `very`를 붙여 다음 토큰으로 `very positive` 또는 `very negative`를 출력하도록 한 방식입니다.

이 개념을 차용하여 분류용 MLP Layer는 유지한 채, 입력 문장 끝에 `문의 의도는:`을 붙였습니다. 그리고 다음 토큰으로 문장의 의도를 예측하도록 유도했습니다. 이렇게 구성하여 분류 Loss와 언어 모델 Loss를 합친 Total Loss가 최소가 되도록 학습했습니다.

```text
Total Loss = Classification Loss + Language Model Loss
```

두 손실의 합산 비율을 바꿔가며 테스트한 결과, 1:1로 더했을 때 정확도 **0.8313**을 기록했습니다. 기본 Decoder 분류 모델보다 아주 미세하게 높아진 값입니다.

<figure>
  <img src="{{ '/assets/images/transformer-decoder-intent-classification/gpt-style-auxiliary-loss-result.png' | relative_url }}" alt="다음 토큰 예측을 더한 GPT-style 의도 분류 모델의 정확도 0.8313 결과">
  <figcaption>분류 손실과 언어 모델 손실을 1:1로 더한 테스트 결과: 정확도 0.8313.</figcaption>
</figure>

## 6. 마무리

이번 실습에서는 Transformer Decoder만으로도 문장 의도 분류 모델을 만들 수 있음을 확인했습니다. 또한 GPT의 다음 토큰 예측 특성을 분류 학습에 함께 활용해 보았습니다.

다만 언어 예측 기능을 추가하면서 필요한 Hyper parameter 값이 커졌는데, 학습 가능한 파라미터 수는 8,393,840개에서 41,770,930개로 증가했습니다.
비슷한 성능을 내기위해 언어 모델로 학습할 때에는 훨씬 더 큰 규모가 필요하다는 얘기 입니다.

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), arXiv:1706.03762.
- Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- [TF04-Intent-Classification-Decoder.ipynb](https://github.com/mkmkchoi/ai-model-study/blob/main/ipynb/TF04-Intent-Classification-Decoder.ipynb).
- [TF04-Intent-Classification-Decoder-gpt-style.ipynb](https://github.com/mkmkchoi/ai-model-study/blob/main/ipynb/TF04-Intent-Classification-Decoder-gpt-style.ipynb).
