# Efficient LLM and Multimodal Foundation Model Survey

This repo contains the paper list and figures for [A Survey of Resource-efficient LLM and Multimodal Foundation Models](https://arxiv.org/pdf/2401.08092.pdf).

## Abstract

Large foundation models, including large language models (LLMs), vision transformers (ViTs), diffusion, and LLM-based multimodal models, are revolutionizing the entire machine learning lifecycle, from training to deployment. However, the substantial advancements in versatility and performance these models offer come at a significant cost in terms of hardware resources. To support the growth of these large models in a scalable and environmentally sustainable way, there has been a considerable focus on developing resource-efficient strategies. This survey delves into the critical importance of such research, examining both algorithmic and systemic aspects. It offers a comprehensive analysis and valuable insights gleaned from existing literature, encompassing a broad array of topics from cutting-edge model architectures and training/serving algorithms to practical system designs and implementations. The goal of this survey is to provide an overarching understanding of how current approaches are tackling the resource challenges posed by large foundation models and to potentially inspire future breakthroughs in this field.

## Scope and rationales

The scope of this survey is mainly defined by following aspects.
- We survey only algorithm and system innovations;
we exclude a huge body of work at hardware design,  which is out of our expertise.
- The definition of resource in this survey is limited to mainly physical ones, including computing, memory, storage, bandwidth, etc;
we exclude training data (labels) and privacy that can also be regarded as resources;
- We mainly survey papers published on top-tier CS conferences, i.e., those included in CSRankings.
We also manually pick related and potentially high-impact papers from arXiv.
- We mainly survey papers published after the year of 2020, since the innovation of AI is going fast with old knowledge and methods being overturned frequently.

![Screenshot](figs/example.png)


## Citation

```
@article{xu2024a,
    title = {A Survey of Resource-efficient LLM and Multimodal Foundation Models},
    author = {Xu, Mengwei and Yin, Wangsong and Cai, Dongqi and Yi, Rongjie
    and Xu, Daliang and Wang, Qipeng and Wu, Bingyang and Zhao, Yihao and Yang, Chen
    and Wang, Shihe and Zhang, Qiyang and Lu, Zhenyan and Zhang, Li and Wang, Shangguang
    and Li, Yuanchun, and Liu Yunxin and Jin, Xin and Liu, Xuanzhe},
    journal={arXiv preprint arXiv:2401.08092},
    year = {2024}
}
```

## Contribute

If we leave out any important papers, please let us know in the Issues and we will include them in the next version.

We will actively maintain the survey and the Github repo.

## Table of Contents

- [Foundation Model Overview](#foundation-model-overview)
    - [Language Foundation Models](#language-foundation-models)
    - [Vision Foundation Models](#vision-foundation-models)
    - [Multimodal Foundation Models](#multimodal-large-fms)
- [Resource-efficient Architectures](#resource-efficient-architectures)
    - [Efficient Attention](#efficient-attention)
    - [Dynamic Neural Network](#dynamic-neural-network)
    - [Diffusion-specific Optimization](#diffusion-specific-optimization)
    - [ViT-specific Optimizations](#vit-specific-optimizations)
- [Resource-efficient Algorithms](#resource-efficient-algorithms)
    - [Pre-training Algorithms](#pre-training-algorithms)
    - [Finetuning Algorithms](#finetuning-algorithms)
    - [Inference Algorithms](#inference-algorithms)
    - [Model Compression](#model-compression)
- [Resource-efficient Systems](#resource-efficient-systems)
    - [Distributed Training](#distributed-training)
    - [Federated Learning](#federated-learning)
    - [Serving on Cloud](#serving-on-cloud)
    - [Serving on Edge](#serving-on-edge)


## Foundation Model Overview


### Language Foundation Models

- Attention is all you need. *[arXiv'17]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/hyunwoongko/transformer)
- Bert: Pre-training of deep bidirectional transformers for language understanding. *[arXiv'18]* [[Paper]](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ) [[Code]](https://github.com/google-research/bert)
- DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *[arXiv'19]* [[Paper]](https://arxiv.org/pdf/1910.01108.pdf) [[Code]](https://huggingface.co/docs/transformers/model_doc/distilbert)
- Roberta: A robustly optimized bert pretraining approach. *[arXiv'19]* [[Paper]](https://arxiv.org/pdf/1907.11692.pdf) [[Code]](https://github.com/pytorch/fairseq)
- Sentence-bert: Sentence embeddings using siamese bert-networks. *[EMNLP'19]* [[Paper]](https://arxiv.org/pdf/1908.10084) [[Code]](https://github.com/UKPLab/sentence-transformers)
- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *[ACL'19]* [[Paper]](https://arxiv.org/pdf/1910.13461) [[Code]](https://paperswithcode.com/paper/bart-denoising-sequence-to-sequence-pre#code)
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *[arXiv'19]* [[Paper]](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf) [[Code]](https://github.com/google-research/text-to-text-transfer-transformer)
- Improving language understanding by generative pre-training. [[URL]](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf)
- Language Models are Unsupervised Multitask Learners. [[URL]](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)
- Language Models are Few-Shot Learners. *[NeurIPS'20]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) [[Code]](https://github.com/openai/gpt-3)
- GLM: General Language Model Pretraining with Autoregressive Blank Infilling. *[arXiv'21]* [[Paper]](https://arxiv.org/pdf/2103.10360) [[Code]](https://github.com/THUDM/GLM)
- Palm: Scaling language modeling with pathways. *[JMLR'22]* [[Paper]](https://www.jmlr.org/papers/volume24/22-1144/22-1144.pdf) [[Code]](https://github.com/lucidrains/PaLM-pytorch)
- Training language models to follow instructions with human feedback. *[NeurIPS'22]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)
- Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *[JMLR'22]* [[Paper]](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)
- Glam: Efficient scaling of language models with mixture-ofexperts. *[ICML'22]* [[Paper]](https://proceedings.mlr.press/v162/du22c.html)
- wav2vec 2.0: A framework for self-supervised learning of speech representations. *[NeurIPS'20]* [[Paper]](https://ar5iv.org/abs/2006.11477) [[Code]](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md)
- HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. *[IEEE/ACM Transactions on Audio,  Speech, and Language Processing'21]* [[Paper]](https://ar5iv.org/abs/2106.07447) [[Code]](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
- Robust Speech Recognition via Large-Scale Weak Supervision. *[ICML'23]* [[Paper]](https://proceedings.mlr.press/v202/radford23a.html)
- GPT-4 Technical Report. *[arXiv'23]* [[Paper]](https://ar5iv.org/abs/2303.08774)
- Palm 2 technical report. [[URL]](https://ar5iv.org/abs/2305.10403)
- Llama 2: Open foundation and fine-tuned chat models. *[arXiv'23]* [[Paper]](https://llama-2.ai/download/) [[Code]](https://github.com/facebookresearch/llama)

### Vision Foundation Models

- End-to-End Object Detection with Transformers. *[ECCV'20]* [[Paper]](https://arxiv.org/abs/2005.12872) [[Code]](https://github.com/facebookresearch/detr)
- Generative Pretraining from Pixels. *[ICML'20]* [[Paper]](https://proceedings.mlr.press/v119/chen20s.html) [[Code]](https://github.com/openai/image-gpt)
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *[ICLR'20]* [[Paper]](https://arxiv.org/abs/2010.11929) [[Code]](https://github.com/gupta-abhay/pytorch-vit)
- Training data-efficient image transformers & distillation through attention. *[ICML'21]* [[Paper]](https://arxiv.org/abs/2012.12877) [[Code]](https://github.com/facebookresearch/deit)
- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *[NeurIPS'21]* [[Paper]](https://arxiv.org/abs/2105.15203) [[Code]](https://github.com/NVlabs/SegFormer)
- You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection. *[NeurIPS'21]* [[Paper]](https://arxiv.org/abs/2105.15203) [[Code]](https://github.com/hustvl/YOLOS)
- Swin Transformer V2: Scaling Up Capacity and Resolution. *[CVPR'22]* [[Paper]](https://arxiv.org/abs/2111.09883) [[Code]](https://github.com/microsoft/Swin-Transformer)
- Masked Autoencoders Are Scalable Vision Learners. *[CVPR'22]* [[Paper]](https://arxiv.org/abs/2111.06377) [[Code]](https://github.com/IcarusWizard/MAE)
- Exploring Plain Vision Transformer Backbones for Object Detection. *[ECCV'22]* [[Paper]](https://arxiv.org/abs/2203.16527) [[Code]](https://github.com/ViTAE-Transformer/ViTDet)
- BEiT: BERT Pre-Training of Image Transformers. *[ICLR'22]* [[Paper]](https://arxiv.org/abs/2106.08254) [[Code]](https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/backbone/beit.py)
- DINOv2: Learning Robust Visual Features without Supervision. *[arXiv'20]* [[Paper]](https://arxiv.org/abs/2005.12872)
- Sequential Modeling Enables Scalable Learning for Large Vision Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2312.00785) [[Code]](https://github.com/ytongbai/LVM?tab=readme-ov-file)

### Multimodal Large FMs

- Learning transferable visual models from natural language supervision. *[ICML'21]* [[Paper]](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) [[Code]](https://github.com/OpenAI/CLIP.
)
- Align before fuse: Vision and language representation learning with momentum distillation. *[NeurIPS'21]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/505259756244493872b7709a8a01b536-Paper.pdf) [[Code]](https://github.com/salesforce/ALBEF)
- Scaling up visual and vision-language representation learning with noisy text supervision. *[ICML'21]* [[Paper]](https://proceedings.mlr.press/v139/jia21b/jia21b.pdf)
- Imagebind: One embedding space to bind them all. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.pdf) [[Code]](https://imagebind.metademolab.com/)
- Languagebind: Extending video-language pretraining to n-modality by language- based semantic alignment. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.01852.pdf) [[Code]](https://github.com/PKU-YuanGroup/LanguageBind)
- Pandagpt: One model to instruction-follow them all. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.16355.pdf) [[Code]](https://panda-gpt.github.io/)
- Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2301.12597.pdf) [[Code]](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
- Minigpt-4: Enhancing vision-language understanding with advanced large language models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2304.10592.pdf) [[Code]](https://minigpt-4.github.io/)
- mplug-owl: Modularization empowers large language models with multi-modality. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2304.14178.pdf?trk=public_post_comment-text) [[Code]](https://github.com/X-PLUG/mPLUG-Owl)
- Visual instruction tuning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2304.08485.pdf) [[Code]](https://llava-vl.github.io/)
- Flamingo: a visual language model for few-shot learning. *[NeurIPS'22]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf)
- Llama-adapter: Efficient fine-tuning of language models with zero-init attention. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2303.16199.pdf%20%C3%A2%E2%82%AC%C5%BE%3Emultimodalno%C3%85%E2%80%BA%C3%84%E2%80%A1%3C/a%3E,%C3%82%C2%A0%3Ca%20href=) [[Code]](https://github.com/OpenGVLab/LLaMA-Adapter)
- Palm-e: An embodied multimodal language model. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2303.03378.pdf?trk=public_post_comment-text) [[Code]](https://palm-e.github.io/)
- Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2303.17580.pdf) [[Code]](https://github.com/microsoft/JARVIS)
- Any-to-any generation via composable diffusion. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.11846.pdf) [[Code]](https://codi-gen.github.io/)
- Next-gpt: Any-to-any multimodal llm. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2309.05519.pdf) [[Code]](https://next-gpt.github.io/)
- Uniter: Universal image-text representation learning. *[ECCV'20]* [[Paper]](https://arxiv.org/pdf/1909.11740.pdf) [[Code]](https://github.com/ChenRocks/UNITER)
- Flava: A foundational language and vision alignment model. *[CVPR'22]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Singh_FLAVA_A_Foundational_Language_and_Vision_Alignment_Model_CVPR_2022_paper.pdf) [[Code]](https://flava-model.github.io/)
- Coca: Contrastive captioners are image-text foundation models. *[arXiv'22]* [[Paper]](https://arxiv.org/pdf/2205.01917.pdf)
- Grounded language-image pre-training. *[CVPR'22]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.pdf) [[Code]](https://github.com/microsoft/GLIP)
- Segment anything. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2304.02643.pdf) [[Code]](https://segment-anything.com/)
- Gemini: A Family of Highly Capable Multimodal Models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2312.11805.pdf)
- Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.11441.pdf) [[Code]](https://github.com/microsoft/SoM)
- Auto-encoding variational bayes. *[arXiv'13]* [[Paper]](https://arxiv.org/abs/1312.6114)
- Neural discrete representation learning. *[NeurIPS'17]* [[Paper]](https://arxiv.org/abs/1711.00937) [[Code]](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py)
- Denoising Diffusion Probabilistic Models. *[NeurIPS'20]* [[Paper]](https://arxiv.org/pdf/2006.11239.pdf) [[Code]](https://github.com/CompVis/latent-diffusion)
- Denoising diffusion implicit models. *[ICLR'21]* [[Paper]](https://arxiv.org/pdf/2010.02502.pdf) [[Code]](https://github.com/CompVis/latent-diffusion)
- Convolutional Networks for Biomedical Image Segmentation. *[MICCAI'15]* [[Paper]](https://arxiv.org/abs/1505.04597) [[Code]](https://github.com/milesial/Pytorch-UNet)
- High-Resolution Image Synthesis with Latent Diffusion Models. *[CVPR'22]* [[Paper]](https://arxiv.org/abs/2112.10752) [[Code]](https://github.com/CompVis/latent-diffusion)
- Consistency models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2303.01469) [[Code]](https://github.com/openai/consistency_models)
- Zero-shot text-to-image generation. *[ICML'21']* [[Paper]](https://arxiv.org/abs/2102.12092) [[Code]](https://github.com/openai/DALL-E)
- Any-to-Any Generation via Composable Diffusion. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.11846) [[Code]](https://codi-gen.github.io/)

## Resource-efficient Architectures


### Efficient Attention

- Longformer: The long-document transformer. *[arXiv'20]* [[Paper]](https://arxiv.org/pdf/2004.05150.pdf?forcedefault=true) [[Code]](https://github.com/allenai/longformer)
- ETC: Encoding Long and Structured Inputs in Transformers. *[ACL'20]* [[Paper]](https://arxiv.org/pdf/2004.08483) [[Code]](http://goo.gle/research-etc-model)
- Big bird: Transformers for longer sequences. *[NeurIPS'2]* [[Paper]](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf) [[Code]](https://github.com/google-research/bigbird)
- Efficient Attentions for Long Document Summarization. *[NAACL'21]* [[Paper]](https://arxiv.org/pdf/2104.02112) [[Code]](https://github.com/)
- MATE: Multi-view Attention for Table Transformer Efficiency. *[EMNLP'21]* [[Paper]](https://arxiv.org/pdf/2109.04312) [[Code]](https://github.com/google-research/tapas/blob/master/MATE.md)
- LittleBird: Efficient Faster & Longer Transformer for Question Answering. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2210.11870) [[Code]](https://github.com/jwnz/littlebird)
- Albert: A lite bert for self-supervised learning of language representations. *[arXiv'19]* [[Paper]](https://arxiv.org/pdf/1909.11942.pdf%3E,) [[Code]](https://github.com/google-research/ALBERT)
- An efficient encoder-decoder architecture with top-down attention for speech separation. *[ICLR'23]* [[Paper]](https://arxiv.org/pdf/2209.15200) [[Code]](https://github.com/JusperLee/TDANet)
- Reformer: The Efficient Transformer. *[ICLR'20]* [[Paper]](https://arxiv.org/pdf/2001.04451) [[Code]](https://github.com/lucidrains/reformer-pytorch)
- Transformers are rnns: Fast autoregressive transformers with linear attention. *[ICML'20]* [[Paper]](http://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf) [[Code]](https://github.com/idiap/fast-transformers)
- Linformer: Self-Attention with Linear Complexity. *[arXiv'20]* [[Paper]](https://arxiv.org/pdf/2006.04768) [[Code]](https://github.com/lucidrains/linformer)
- Luna: Linear unified nested attention. *[NeurIPS'21]* [[Paper]](https://proceedings.neurips.cc/paper/2021/file/14319d9cfc6123106878dc20b94fbaf3-Paper.pdf) [[Code]](https://github.com/sooftware/luna-transformer)
- Rethinking Attention with Performers. *[arXiv'20]* [[Paper]](https://proceedings.neurips.cc/paper/2021/file/14319d9cfc6123106878dc20b94fbaf3-Paper.pdf) [[Code]](https://github.com/calclavia/Performer-Pytorch)
- PolySketchFormer: Fast Transformers via Sketches for Polynomial Kernels. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.01655)
- Mega: Moving Average Equipped Gated Attention. *[ICLR'23]* [[Paper]](https://arxiv.org/pdf/2209.10655) [[Code]](https://github.com/lucidrains/Mega-pytorch)
- Vision Transformer with Deformable Attention. *[arXiv'22]* [[Paper]](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=Vision+Transformer+with+Deformable+Attention&btnG=#:~:text=%E5%B9%B4%E4%BB%BD-,%5BPDF%5D%20thecvf.com,-Vision%20transformer%20with) [[Code]](https://www.bing.com/search?q=github+Vision+Transformer+with+Deformable+Attention&qs=n&form=QBRE&sp=-1&lq=0&pq=github+&sc=10-7&sk=&cvid=A3CF875E050C48C09D4A92E63C1DDA94&ghsh=0&ghacc=0&ghpl=#:~:text=com/LeapLabTHU/DAT-,DAT,-Repository%20of%20Vision)
- CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. *[arXiv'21]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_CrossViT_Cross-Attention_Multi-Scale_Vision_Transformer_for_Image_Classification_ICCV_2021_paper.pdf) [[Code]](https://github.com/IBM/CrossViT)
- An attention free transformer. *[arXiv'21]* [[Paper]](https://arxiv.org/pdf/2105.14103) [[Code]](https://github.com/WD-Leong/NLP-Attention-Free-Transformer)
- Hyena hierarchy: Towards larger convolutional language models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2302.10866)
- Perceiver: General perception with iterative attention. *[ICML'21]* [[Paper]](http://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf) [[Code]](https://github.com/lucidrains/perceiver-pytorch)
- Scaling transformer to 1m tokens and beyond with rmt. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2304.11062.pdf??ref=eiai.info)
- Recurrent memory transformer. *[NeurIPS'22]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf) [[Code]](https://github.com/booydar/LM-RMT)
- RWKV: Reinventing RNNs for the Transformer Era. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.13048) [[Code]](https://github.com/dcarpintero/machine-learning/blob/main/llms/RWKV.md)
- Retentive Network: A Successor to Transformer for Large Language Model. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2307.08621) [[Code]](https://github.com/ShaderManager/RetNet)
- Efficiently modeling long sequences with structured state spaces. *[ICLR'22]* [[Paper]](https://arxiv.org/pdf/2111.00396) [[Code]](https://github.com/state-spaces/s4)
- Hungry hungry hippos: Towards language modeling with state space models. *[ICLR'23]* [[Paper]](https://arxiv.org/pdf/2212.14052) [[Code]](https://github.com/HazyResearch/H3)
- Resurrecting recurrent neural networks for long sequences. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2303.06349) [[Code]](https://github.com/TingdiRen/LRU_pytorch)
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2312.00752.pdf) [[Code]](https://github.com/state-spaces/mamba)

### Dynamic Neural Network

- Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *[JMLR'22]* [[Paper]](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf) [[Code]](https://github.com/VikParuchuri/marker/blob/master/data/examples/nougat/switch_transformers.md)
- Scaling vision with sparse mixture of experts. *[NeruIPS'21]* [[Paper]](https://proceedings.neurips.cc/paper/2021/file/48237d9f2dea8c74c2a72126cf63d933-Paper.pdf) [[Code]](https://github.com/google-research/vmoe)
- Glam: Efficient scaling of language models with mixture-of-experts. *[ICML'22]* [[Paper]](https://proceedings.mlr.press/v162/du22c/du22c.pdf) [[Code]](https://github.com/lucidrains/mixture-of-experts)
- Multimodal contrastive learning with limoe: the language-image mixture of experts. *[NeruIPS'22]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/3e67e84abf900bb2c7cbd5759bfce62d-Paper-Conference.pdf) [[Code]](https://github.com/YeonwooSung/LIMoE-pytorch)
- Mistral 7B. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.06825.pdf]]%3E) [[Code]](https://github.com/mistralai/mistral-src)
- Fast Feedforward Networks. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2308.14711) [[Code]](https://github.com/pbelcak/fastfeedforward)
- MoEfication: Transformer Feed-forward Layers are Mixtures of Experts. *[ACL'22]* [[Paper]](https://arxiv.org/pdf/2110.01786) [[Code]](https://github.com/thunlp/MoEfication)
- Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2212.05055) [[Code]](https://github.com/google-research/t5x/tree/main/t5x/contrib/moe)
- Simplifying Transformer Blocks. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2311.01906) [[Code]](https://github.com/kyegomez/SimplifiedTransformers)
- Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.05424) [[Code]](https://github.com/raymin0223/fast_robust_early_exit)
- Bert loses patience: Fast and robust inference with early exit. *[NeruIPS'20]* [[Paper]](https://proceedings.neurips.cc/paper/2020/file/d4dd111a4fd973394238aca5c05bebe3-Paper.pdf) [[Code]](https://github.com/JetRunner/PABEE)
- DeeBERT: Dynamic early exiting for accelerating BERT inference. *[arXiv'20]* [[Paper]](https://arxiv.org/pdf/2004.12993) [[Code]](https://github.com/huggingface/transformers/blob/main/examples/research_projects/deebert/README.md)
- LGViT: Dynamic Early Exiting for Accelerating Vision Transformer. *[MM'23]* [[Paper]](https://arxiv.org/pdf/2308.00255) [[Code]](https://github.com/falcon-xu/LGViT)
- Multi-Exit Vision Transformer for Dynamic Inference. *[arXiv'21]* [[Paper]](https://arxiv.org/pdf/2106.15183)
- Skipdecode: Autoregressive skip decoding with batching and caching for efficient llm inference. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2307.02628)

### Diffusion-specific Optimization

- Improved denoising diffusion probabilistic models. *[arXiv'21]* [[Paper]](https://arxiv.org/abs/2102.09672) [[Code]](https://github.com/openai/improved-diffusion)
- Accelerating diffusion models via early stop of the diffusion process. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2205.12524) [[Code]](https://github.com/ZhaoyangLyu/Early_Stopped_DDPM)
- Denoising diffusion implicit models. *[ICLR'21]* [[Paper]](https://arxiv.org/pdf/2010.02502.pdf) [[Code]](https://github.com/CompVis/latent-diffusion)
- gDDIM: Generalized denoising diffusion implicit models. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2206.05564) [[Code]](https://github.com/qsh-zh/gDDIM)
- Pseudo numerical methods for diffusion models on manifolds. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2202.09778) [[Code]](https://github.com/luping-liu/PNDM)
- Elucidating the design space of diffusion-based generative models. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2206.00364) [[Code]](https://github.com/NVlabs/edm)
- Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. *[NeurIPS'22]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf) [[Code]](https://github.com/LuChengTHU/dpm-solver)
- Progressive distillation for fast sampling of diffusion models. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2202.00512) [[Code]](https://github.com/Hramchenko/diffusion_distiller)
- Fast sampling of diffusion models with exponential integrator. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2204.13902) [[Code]](https://github.com/qsh-zh/deis)
- Score-based generative modeling through stochastic differential equations. *[arXiv'20]* [[Paper]](https://arxiv.org/abs/2011.13456) [[Code]](https://github.com/yang-song/score_sde)
- Learning fast samplers for diffusion models by differentiating through sample quality. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2202.05830)
- Salad: Part-level latent diffusion for 3d shape generation and manipulation. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Koo_SALAD_Part-Level_Latent_Diffusion_for_3D_Shape_Generation_and_Manipulation_ICCV_2023_paper.pdf) [[Code]](https://salad3d.github.io/)
- Binary Latent Diffusion. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Binary_Latent_Diffusion_CVPR_2023_paper.pdf) [[Code]](https://github.com/ZeWang95/BinaryLatentDiffusion)
- LD-ZNet: A latent diffusion approach for text-based image segmentation. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/PNVR_LD-ZNet_A_Latent_Diffusion_Approach_for_Text-Based_Image_Segmentation_ICCV_2023_paper.pdf) [[Code]](https://koutilya-pnvr.github.io/LD-ZNet/)
- Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Schramowski_Safe_Latent_Diffusion_Mitigating_Inappropriate_Degeneration_in_Diffusion_Models_CVPR_2023_paper.pdf) [[Code]](https://github.com/ml-research/safe-latent-diffusion)
- High-resolution image reconstruction with latent diffusion models from human brain activity. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf) [[Code]](https://github.com/yu-takagi/StableDiffusionReconstruction)
- Belfusion: Latent diffusion for behavior-driven human motion prediction. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Barquero_BeLFusion_Latent_Diffusion_for_Behavior-Driven_Human_Motion_Prediction_ICCV_2023_paper.pdf) [[Code]](https://github.com/BarqueroGerman/BeLFusion)
- Unified multi-modal latent diffusion for joint subject and text conditional image generation. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2303.09319)
- SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2306.00980) [[Code]](https://github.com/snap-research/SnapFusion)
- ERNIE-ViLG 2.0: Improving text-to-image diffusion model with knowledge-enhanced mixture-of-denoising-experts. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_ERNIE-ViLG_2.0_Improving_Text-to-Image_Diffusion_Model_With_Knowledge-Enhanced_Mixture-of-Denoising-Experts_CVPR_2023_paper.pdf) [[Code]](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/text_to_image)
- ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2211.01324)
- ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.07702) [[Code]](https://yingqinghe.github.io/scalecrafter/)
- Image Super-resolution Via Latent Diffusion: A Sampling-space Mixture Of Experts And Frequency-augmented Decoder Approach. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.12004) [[Code]](https://github.com/tencent-ailab/Frequency_Aug_VAE_MoESR)

### ViT-specific Optimizations

- LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference. *[ICCV'21]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.pdf) [[Code]](https://github.com/facebookresearch/LeViT)
- MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. *[ICLR'22]* [[Paper]](https://arxiv.org/abs/2110.02178) [[Code]](https://github.com/apple/ml-cvnets.
)
- EfficientFormer: Vision Transformers at MobileNet Speed. *[NeurIPS'22]* [[Paper]](https://arxiv.org/abs/2206.01191) [[Code]](https://github.com/snap-research/EfficientFormer)
- EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention. *[CVPR'23]* [[Paper]](https://arxiv.org/abs/2305.07027) [[Code]](https://github.com/mit-han-lab/efficientvit)
- MetaFormer Is Actually What You Need for Vision. *[CVPR'22]* [[Paper]](https://arxiv.org/abs/2111.11418) [[Code]](https://github.com/sail-sg/poolformer)

## Resource-efficient Algorithms


### Pre-training Algorithms

- Deduplicating Training Data Makes Language Models Better. *[ACL'22]* [[Paper]](https://aclanthology.org/2022.acl-long.577.pdf) [[Code]](https://github.com/google-research/deduplicate-text-datasets)
- TRIPS: Efficient Vision-and-Language Pre-training with Text-Relevant Image Patch Selection. *[EMNLP'22]* [[Paper]](https://aclanthology.org/2022.emnlp-main.273.pdf)
- Masked autoencoders are scalable vision learners. *[CVPR'22]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) [[Code]](https://github.com/facebookresearch/mae)
- MixMAE: Mixed and masked autoencoder for efficient pretraining of hierarchical vision transformers. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_MixMAE_Mixed_and_Masked_Autoencoder_for_Efficient_Pretraining_of_Hierarchical_CVPR_2023_paper.pdf) [[Code]](https://github.com/Sense-X/MixMIM)
- COPA: Efficient Vision-Language Pre-training through Collaborative Object-and Patch-Text Alignment. *[MM'23]* [[Paper]](https://arxiv.org/pdf/2308.03475.pdf)
- Patchdropout: Economizing vision transformers using patch dropout. *[WACV'23]* [[Paper]](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_PatchDropout_Economizing_Vision_Transformers_Using_Patch_Dropout_WACV_2023_paper.pdf) [[Code]](https://github.com/yueliukth/PatchDropout)
- Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Joint_Token_Pruning_and_Squeezing_Towards_More_Aggressive_Compression_of_CVPR_2023_paper.pdf) [[Code]](https://github.com/megvii-research/TPS-CVPR2023)
- Zero-Cost Proxies for Lightweight NAS. *[ICLR'21]* [[Paper]](https://openreview.net/pdf?id=0cmMMy8J5q) [[Code]](https://github.com/SamsungLabs/zero-cost-nas)
- ZiCo: Zero-shot NAS via inverse Coefficient of Variation on Gradients. *[ICLR'23]* [[Paper]](https://openreview.net/pdf?id=rwo-ls5GqGn) [[Code]](https://github.com/SLDGroup/ZiCo)
- PASHA: Efficient HPO and NAS with Progressive Resource Allocation. *[ICLR'23]* [[Paper]](https://openreview.net/pdf?id=syfgJE6nFRW) [[Code]](https://github.com/ondrejbohdal/pasha)
- RankNAS: Efficient Neural Architecture Search by Pairwise Ranking. *[EMNLP'21]* [[Paper]](https://aclanthology.org/2021.emnlp-main.191.pdf)
- PreNAS: Preferred One-Shot Learning Towards Efficient Neural Architecture Search. *[ICML'23]* [[Paper]](https://proceedings.mlr.press/v202/wang23f/wang23f.pdf) [[Code]](https://github.com/tinyvision/PreNAS)
- ElasticViT: Conflict-aware Supernet Training for Deploying Fast Vision Transformer on Diverse Mobile Devices. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_ElasticViT_Conflict-aware_Supernet_Training_for_Deploying_Fast_Vision_Transformer_on_ICCV_2023_paper.pdf) [[Code]](https://github.com/microsoft/Moonlit/tree/main/ElasticViT)
- Efficient training of BERT by progressively stacking. *[ICML'19]* [[Paper]](https://proceedings.mlr.press/v97/gong19a/gong19a.pdf) [[Code]](https://github.com/gonglinyuan/StackingBERT)
- On the Transformer Growth for Progressive BERT Training. *[NAACL'21]* [[Paper]](https://aclanthology.org/2021.naacl-main.406.pdf) [[Code]](https://github.com/google-research/google-research/tree/master/grow_bert)
- Staged training for transformer language models. *[ICML'22]* [[Paper]](https://proceedings.mlr.press/v162/shen22f/shen22f.pdf) [[Code]](https://github.com/allenai/staged-training)
- Knowledge Inheritance for Pre-trained Language Models. *[NAACL'22]* [[Paper]](https://aclanthology.org/2022.naacl-main.288.pdf) [[Code]](https://github.com/thunlp/Knowledge-Inheritance)
- Learning to Grow Pretrained Models for Efficient Transformer Training. *[ICLR'23]* [[Paper]](https://openreview.net/pdf?id=cDYRS5iZ16f) [[Code]](https://github.com/VITA-Group/LiGO)
- Mesa: A memory-saving training framework for transformers. *[arXiv'21]* [[Paper]](https://arxiv.org/pdf/2111.11124.pdf) [[Code]](https://github.com/ziplab/Mesa)
- GACT: Activation compressed training for generic network architectures. *[ICML'22]* [[Paper]](https://proceedings.mlr.press/v162/liu22v/liu22v.pdf) [[Code]](https://github.com/LiuXiaoxuanPKU/GACT-ICML)

### Finetuning Algorithms

- Memory efficient continual learning with transformers. *[NeurIPS'22]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/4522de4178bddb36b49aa26efad537cf-Paper-Conference.pdf)
- Metatroll: Few-shot detection of state-sponsored trolls with transformer adapters. *[WWW'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3543507.3583417) [[Code]](https://github.com/ltian678/metatroll-code)
- St-adapter: Parameter-efficient image-to- video transfer learning. *[NeurIPS'22]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/a92e9165b22d4456fc6d87236e04c266-Paper-Conference.pdf) [[Code]](https://github.com/linziyi96/st-adapter)
- Parameter-efficient fine-tuning without introducing new latency. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.16742.pdf) [[Code]](https://github.com/baohaoliao/pafi_hiwi)
- Adamix: Mixture-of-adaptations for parameter-efficient model tuning. *[arXiv'22]* [[Paper]](https://arxiv.org/pdf/2210.17451v1.pdf) [[Code]](https://github.com/microsoft/AdaMix)
- Residual adapters for parameter-efficient asr adaptation to atypical and accented speech. *[arXiv'21]* [[Paper]](https://arxiv.org/pdf/2109.06952.pdf)
- Make your pre-trained model reversible: From parameter to memory efficient fine-tuning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2306.00477.pdf) [[Code]](https://github.com/BaohaoLiao/mefts)
- Pema: Plug-in external memory adaptation for language models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2311.08590.pdf)
- The power of scale for parameter-efficient prompt tuning. *[arXiv'21]* [[Paper]](https://arxiv.org/pdf/2104.08691.pdf)
- Attempt: Parameter-efficient multi-task tuning via attentional mixtures of soft prompts. *[EMNLP'22]* [[Paper]](https://aclanthology.org/2022.emnlp-main.446.pdf) [[Code]](https://github.com/AkariAsai/ATTEMPT)
- Mprompt: Exploring multi-level prompt tuning for machine reading comprehension. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.18167.pdf) [[Code]](https://github.com/Chen-GX/MPrompt)
- Bioinstruct: Instruction tuning of large language models for biomedical natural language processing. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.19975.pdf) [[Code]](https://github.com/hieutran81/BioInstruct)
- Decomposed prompt tuning via low-rank reparameterization. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.10094.pdf) [[Code]](https://github.com/XYaoooo/DPT)
- A dual prompt learning framework for few-shot dialogue state tracking. *[WWW'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3543507.3583238) [[Code]](https://github.com/YANG-Yuting/DPL)
- User-aware prefix-tuning is a good learner for personalized image captioning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2312.04793.pdf)
- Prefix-diffusion: A lightweight diffusion model for diverse image captioning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2309.04965.pdf)
- Domain aligned prefix averaging for domain generalization in abstractive summarization. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.16820.pdf) [[Code]](https://github.com/pranavajitnair/DAPA)
- Prefix propagation: Parameter-efficient tuning for long sequences. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.12086.pdf) [[Code]](https://github.com/MonliH/prefix-propagation)
- Pip: Parse-instructed prefix for syntactically controlled paraphrase generation. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.16701.pdf) [[Code]](https://github.com/uclanlp/PIP)
- Towards building the federated gpt: Federated instruction tuning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.05644.pdf?utm_referrer=https%3A%2F%2Fdzen.ru%2Fmedia%2Fid%2F5e048b1b2b616900b081f1d9%2F645f43ee7906327d8643ffbc) [[Code]](https://github.com/JayZhang42/FederatedGPT-Shepherd)
- Domain-oriented prefix-tuning: Towards efficient and generalizable fine-tuning for zero-shot dialogue summarization. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2204.04362.pdf)
- Unified low-resource sequence labeling by sample-aware dynamic sparse finetuning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2311.03748.pdf) [[Code]](https://github.com/psunlpgroup/FISH-DIP)
- On the effectiveness of parameter-efficient fine-tuning. *[AAAI'23]* [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/download/26505/26277) [[Code]](https://github.com/fuzihaofzh/AnalyzeParameterEfficientFinetune)
- Sensitivity-aware visual parameter-efficient fine-tuning. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/He_Sensitivity-Aware_Visual_Parameter-Efficient_Fine-Tuning_ICCV_2023_paper.pdf) [[Code]](https://github.com/)
- Vl-pet: Vision-and-language parameter-efficient tuning via granularity control. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_VL-PET_Vision-and-Language_Parameter-Efficient_Tuning_via_Granularity_Control_ICCV_2023_paper.pdf) [[Code]](https://github.com/HenryHZY/VL-PET)
- Smartfrz: An efficient training framework using attention-based layer freezing. *[ICLR'23]* [[Paper]](https://openreview.net/pdf?id=i9UlAr1T_xl)
- Token mixing: parameter-efficient transfer learning from image-language to video-language. *[AAAI'23]* [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25267/25039) [[Code]](https://github.com/yuqi657/video)
- One-for-all: Generalized lora for parameter-efficient fine-tuning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2306.07967.pdf) [[Code]](https://sites.google.com/view/generalized-lora)
- Dsee: Dually sparsity-embedded efficient tuning of pre-trained language models. *[arXiv'21]* [[Paper]](https://arxiv.org/pdf/2111.00160.pdf) [[Code]](https://github.com/VITA-Group/DSEE)
- Longlora: Efficient fine-tuning of long-context large language models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2309.12307.pdf?trk=public_post_comment-text) [[Code]](https://github.com/dvlab-research/LongLoRA)
- Qlora: Efficient finetuning of quantized llms. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.14314.pdf) [[Code]](https://github.com/artidoro/qlora)
- Pela: Learning parameter-efficient models with low-rank approximation. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.10700.pdf) [[Code]](https://github.com/guoyang9/PELA)
- Efficientdm: Efficient quantization-aware fine-tuning of low-bit diffusion models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.03270.pdf)
- Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.14152)
- Loftq: Lora-fine-tuning-aware quantization for large language models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.08659.pdf?trk=public_post_comment-text) [[Code]](https://github.com/yxli2123/LoftQ)
- Full parameter fine-tuning for large language models with limited resources. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2306.09782) [[Code]](https://github.com/OpenLMLab/LOMO)
- Fine-tuning language models with just forward passes. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.17333) [[Code]](https://github.com/princeton-nlp/MeZO)
- Efficient transformers with dynamic token pooling. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2211.09761) [[Code]](https://github.com/PiotrNawrot/dynamic-pooling)
- Qa-lora: Quantization-aware low-rank adaptation of large language models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2309.14717) [[Code]](https://github.com/yuhuixu1993/qa-lora)
- Efficient low-rank backpropagation for vision transformer adaptation. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2309.15275)

### Inference Algorithms

- Fast inference from transformers via speculative decoding. *[ICML'23]* [[Paper]](https://arxiv.org/abs/2211.17192) [[Code]](https://github.com/feifeibear/LLMSpeculativeSampling)
- Accelerating Large Language Model Decoding with Speculative Sampling. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2302.01318) [[Code]](https://github.com/lucidrains/speculative-decoding)
- SpecTr: Fast Speculative Decoding via Optimal Transport. *[NeurIPS'23]* [[Paper]](https://arxiv.org/abs/2310.15141) [[Code]](https://github.com/lucidrains/speculative-decoding)
- ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training. *[EMNLP'20]* [[Paper]](https://arxiv.org/abs/2001.04063) [[Code]](https://github.com/microsoft/ProphetNet)
- Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2309.08168) [[Code]](https://github.com/dilab-zju/self-speculative-decoding)
- LLMCad: Fast and Scalable On-device Large Language Model Inference. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2309.04255)
- Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads. [[URL]](https://sites.google.com/view/medusa-llm)
- Break the Sequential Dependency of LLM Inference Using Lookahead Decoding. [[URL]](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
- SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.09781) [[Code]](https://github.com/flexflow/FlexFlow/tree/inference)
- Prompt Cache: Modular Attention Reuse for Low-Latency Inference. *[arXiv‘23]* [[Paper]](https://arxiv.org/abs/2311.04934)
- Inference with Reference: Lossless Acceleration of Large Language Models. *[arXiv‘23]* [[Paper]](https://arxiv.org/abs/2304.04487) [[Code]](https://github.com/microsoft/unilm)
- Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding. *[arXiv‘23]* [[Paper]](https://arxiv.org/abs/2307.15337) [[Code]](https://github.com/imagination-research/sot)
- LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. *[EMNLP’23]* [[Paper]](https://arxiv.org/abs/2310.05736) [[Code]](https://github.com/microsoft/LLMLingua)
- Prompt Compression and Contrastive Conditioning for Controllability and Toxicity Reduction in Language Models. *[EMNLP‘22]* [[Paper]](https://arxiv.org/abs/2210.03162) [[Code]](https://github.com/BYU-PCCL/prompt-compression-contrastive-coding)
- Etropyrank: Unsupervised keyphrase extraction via side-information optimization for language model-based text compression. *[ICML‘23]* [[Paper]](https://arxiv.org/abs/2308.13399)
- LLMZip: Lossless Text Compression using Large Language Models. *[arXiv‘23]* [[Paper]](https://arxiv.org/abs/2306.04050) [[Code]](https://github.com/vcskaushik/LLMzip)
- In-context Autoencoder for Context Compression in a Large Language Mode. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2307.06945) [[Code]](https://github.com/getao/icae.
)
- Nugget 2D: Dynamic Contextual Compression for Scaling Decoder-only Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.02409)
- Boosting LLM Reasoning: Push the Limits of Few-shot Learning with Reinforced In-Context Pruning. *[arXiv‘23]* [[Paper]](https://arxiv.org/abs/2312.08901)
- PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination. *[ICML‘20]* [[Paper]](https://arxiv.org/abs/2001.08950) [[Code]](https://github.com/IBM/PoWER-BERT)
- Length-Adaptive Transformer: Train Once with Length Drop, Use Anytime with Search. *[ACL‘21]* [[Paper]](https://arxiv.org/abs/2010.07003) [[Code]](https://github.com/clovaai/length-adaptive-transformer)
- TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference. *[NAACL‘21]* [[Paper]](https://arxiv.org/abs/2105.11618) [[Code]](https://github.com/thunlp/TR-BERT)
- DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. *[NeurIPS‘21]* [[Paper]](https://arxiv.org/abs/2106.02034) [[Code]](https://github.com/raoyongming/DynamicViT)
- AdaViT: Adaptive Vision Transformers for Efficient Image Recognition. *[CVPR‘21]* [[Paper]](https://arxiv.org/abs/2111.15668)
- AdaViT: Adaptive Tokens for Efficient Vision Transformer. *[CVPR‘22]* [[Paper]](https://arxiv.org/abs/2112.07658) [[Code]](https://a-vit.github.io/)
- SPViT: Enabling Faster Vision Transformers via Soft Token Pruning. *[ECCV‘22]* [[Paper]](https://arxiv.org/abs/2112.13890) [[Code]](https://github.com/PeiyanFlying/SPViT)
- PuMer: Pruning and Merging Tokens for Efficient Vision Language Models. *[ACL ‘23]* [[Paper]](https://arxiv.org/abs/2305.17530) [[Code]](https://github.com/csarron/PuMer)
- H2o: Heavy-hitter oracle for efficient generative inference of large language models. *[NeurIPS‘23]* [[Paper]](https://arxiv.org/abs/2306.14048) [[Code]](https://github.com/FMInference/H2O)
- Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers. *[arXiv ‘23]* [[Paper]](https://arxiv.org/abs/2305.15805) [[Code]](https://github.com/sanagno/adaptively_sparse_attention)
- Landmark Attention: Random-Access Infinite Context Length for Transformers. *[NeurIPS‘23]* [[Paper]](https://arxiv.org/abs/2305.16300) [[Code]](https://github.com/epfml/landmark-attention/)
- Train short, test long: Attention with linear biases enables input length extrapolation. *[ICLR‘22]* [[Paper]](https://arxiv.org/abs/2108.12409) [[Code]](https://github.com/ofirpress/attention_with_linear_biases)
- A Length-Extrapolatable Transformer. *[ACL‘22]* [[Paper]](https://arxiv.org/abs/2212.10554) [[Code]](https://github.com/sunyt32/torchscale)
- CLEX: Continuous Length Extrapolation for Large Language Models. *[arXiv ‘23]* [[Paper]](https://arxiv.org/abs/2310.16450v1) [[Code]](https://github.com/DAMO-NLP-SG/CLEX)
- Extending Context Window of Large Language Models via Positional Interpolation. *[arXiv ‘23]* [[Paper]](https://arxiv.org/abs/2306.15595)
- YaRN: Efficient Context Window Extension of Large Language Models. *[arXiv’23]* [[Paper]](https://arxiv.org/abs/2309.00071) [[Code]](https://github.com/jquesnelle/yarn)
- functional interpolation for relative positions improves long context transformers. *[arXiv '23]* [[Paper]](https://arxiv.org/abs/2310.04418)
- PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training. *[arXiv '23]* [[Paper]](https://ar5iv.labs.arxiv.org/html/2309.10400) [[Code]](https://github.com/dwzhu-pku/PoSE)
- Recurrent Memory Transformer. *[NeurIPS'22]* [[Paper]](https://arxiv.org/abs/2207.06881) [[Code]](https://github.com/booydar/LM-RMT)
- Block-Recurrent Transformers. *[NeurIPS'22]* [[Paper]](https://arxiv.org/abs/2203.07852) [[Code]](https://github.com/lucidrains/block-recurrent-transformer-pytorch)
- Memformer: A Memory-Augmented Transformer for Sequence Modeling. *[ACL'22]* [[Paper]](https://arxiv.org/abs/2010.06891) [[Code]](https://github.com/lucidrains/memformer)
- LM-Infinite: Simple On-the-Fly Length Generalization for Large Language Models. *[arXiv '23]* [[Paper]](https://arxiv.org/abs/2308.16137) [[Code]](https://github.com/Glaciohound/LM-Infinite)
- Efficient Streaming Language Models with Attention Sinks. *[arXiv '23]* [[Paper]](https://arxiv.org/abs/2309.17453) [[Code]](https://github.com/mit-han-lab/streaming-llm)
- Parallel context windows for large language models. *[ACL'23]* [[Paper]](https://arxiv.org/abs/2212.10947) [[Code]](https://github.com/ai21labs/parallel-context-windows)
- LongNet: Scaling Transformers to 1,000,000,000 Tokens. *[arXiv '23]* [[Paper]](https://arxiv.org/abs/2307.02486) [[Code]](https://aka.ms/LongNet)
- Efficient Long-Text Understanding with Short-Text Models. *[TACL'23]* [[Paper]](https://arxiv.org/abs/2208.00748) [[Code]](https://github.com/Mivg/SLED)

### Model Compression

- From Dense to Sparse: Contrastive Pruning for Better Pre-Trained Language Model Compression. *[AAAI'22]* [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21408)
- Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models. *[NeurIPS'22]* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html) [[Code]](https://github.com/lmxyy/sige)
- ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design. *[HPCA'23]* [[Paper]](https://ieeexplore.ieee.org/abstract/document/10071027) [[Code]](https://github.com/GATECH-EIC/ViTCoD)
- A Simple and Effective Pruning Approach for Large Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2306.11695) [[Code]](https://github.com/locuslab/wanda)
- Dynamic Sparse Training: Find Efficient Sparse Network From Scratch With Trainable Masked Layers. *[ICLR'20]* [[Paper]](https://arxiv.org/abs/2005.06870) [[Code]](https://github.com/junjieliu2910/DynamicSparseTraining)
- UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers. *[ICML'23]* [[Paper]](https://dl.acm.org/doi/10.5555/3618408.3619703) [[Code]](https://github.com/sdc17/UPop)
- Sparsegpt: Massive language models can be accurately pruned in one-shot. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2301.00774) [[Code]](https://github.com/ist-daslab/sparsegpt)
- One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models. *[ICASSP'24]* [[Paper]](https://arxiv.org/abs/2310.09499) [[Code]](https://github.com/talkking/MixGPT)
- BiT: Robustly Binarized Multi-distilled Transformer. *[NeurIPS'22]* [[Paper]](https://papers.nips.cc/paper_files/paper/2022/hash/5c1863f711c721648387ac2ef745facb-Abstract-Conference.html) [[Code]](https://github.com/facebookresearch/bit)
- DSEE: Dually Sparsity-embedded Efficient Tuning of Pre-trained Language Models. *[ACL'23]* [[Paper]](https://aclanthology.org/2023.acl-long.456/) [[Code]](https://github.com/VITA-Group/DSEE)
- Block-Skim: Efficient Question Answering for Transformer. *[AAAI'22]* [[Paper]](https://aaai.org/papers/10710-block-skim-efficient-question-answering-for-transformer/) [[Code]](https://github.com/ChandlerGuan/blockskim)
- Depgraph: Towards any structural pruning. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html) [[Code]](https://github.com/VainF/Torch-Pruning)
- PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance. *[ICML'22]* [[Paper]](https://proceedings.mlr.press/v162/zhang22ao.html) [[Code]](https://github.com/QingruZhang/PLATON)
- Differentiable joint pruning and quantization for hardware efficiency. *[ECCV'20]* [[Paper]](https://dl.acm.org/doi/abs/10.1007/978-3-030-58526-6_16)
- SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning. *[HPCA'21]* [[Paper]](https://ieeexplore.ieee.org/document/9407232) [[Code]](https://github.com/mit-han-lab/spatten-llm)
- Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.06694) [[Code]](https://github.com/princeton-nlp/LLM-Shearing)
- Accelerated sparse neural training: A provable and efficient method to find n: m transposable masks. *[NeurIPS'21]* [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/b0490b85e92b64dbb5db76bf8fca6a82-Abstract.html) [[Code]](https://github.com/itayhubara/AcceleratedSparseNeuralTraining)
- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.18403)
- What matters in the structured pruning of generative language models?. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2302.03773)
- LLM-Pruner: On the Structural Pruning of Large Language Models. *[NeurIPS'23]* [[Paper]](https://dev.neurips.cc/virtual/2023/poster/72074) [[Code]](https://github.com/horseee/LLM-Pruner)
- Deja vu: Contextual sparsity for efficient llms at inference time. *[ICML'23]* [[Paper]](https://arxiv.org/abs/2310.17157) [[Code]](https://github.com/FMInference/DejaVu)
- PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2312.12456) [[Code]](https://github.com/SJTU-IPADS/PowerInfer)
- Distilling Large Vision-Language Model with Out-of-Distribution Generalizability. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.html) [[Code]](https://github.com/xuanlinli17/large_vlm_distillation_ood)
- DIME-FM : DIstilling Multimodal and Efficient Foundation Models. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Sun_DIME-FM__DIstilling_Multimodal_and_Efficient_Foundation_Models_ICCV_2023_paper.html) [[Code]](https://github.com/sunxm2357/DIME-FM)
- MixKD: Towards Efficient Distillation of Large-scale Language Models. *[arXiv'20]* [[Paper]](https://arxiv.org/abs/2011.00593)
- Less is More: Task-aware Layer-wise Distillation for Language Model Compression. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2210.01351) [[Code]](https://github.com/oshindutta/task-aware-distillation)
- Propagating Knowledge Updates to LMs Through Distillation. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2306.09306)
- GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2306.13649) [[Code]](https://github.com/njulus/GKD)
- Knowledge Distillation of Large Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2306.08543)
- Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. *[ACL'23]* [[Paper]](https://aclanthology.org/2023.conll-babylm.24.pdf) [[Code]](https://github.com/timinar/BabyLlama)
- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. *[ACL'23]* [[Paper]](https://aclanthology.org/2023.findings-acl.507/) [[Code]](https://github.com/google-research/distilling-step-by-step)
- Teaching Small Language Models to Reason. *[ACL'22]* [[Paper]](https://aclanthology.org/2023.acl-short.151/)
- Explanations from Large Language Models Make Small Reasoners Better. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2210.06726)
- Lion: Adversarial distillation of closed-source large language model. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.12870) [[Code]](https://github.com/yjiangcm/lion)
- LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2304.14402) [[Code]](https://github.com/mbzuai-nlp/LaMini-LM)
- LLM. int8 (): 8-bit Matrix Multiplication for Transformers at Scale. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2208.07339) [[Code]](https://github.com/TimDettmers/bitsandbytes)
- LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2206.09557)
- Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning. *[NeurIPS'22]* [[Paper]](https://nips.cc/Conferences/2022/Schedule?showEvent=53412) [[Code]](https://github.com/ist-daslab/obc)
- GPTQ: accurate post-training quantization for generative pre-trained transformers. *[ICLR'23]* [[Paper]](https://arxiv.org/abs/2210.17323) [[Code]](https://github.com/IST-DASLab/gptq)
- Few-bit Backward: Quantized Gradients of Activation Functions for Memory Footprint Reduction. *[ICML'22]* [[Paper]](https://openreview.net/forum?id=m2S96Qf2R3) [[Code]](https://github.com/SkoltechAI/fewbit)
- SqueezeLLM: Dense-and-Sparse Quantization. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2306.07629) [[Code]](https://github.com/SqueezeAILab/SqueezeLLM)
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2306.03078) [[Code]](https://github.com/Vahe1994/SpQR)
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2306.00978) [[Code]](https://github.com/mit-han-lab/llm-awq)
- QuIP: 2-Bit Quantization of Large Language Models With Guarantees. *[NeurIPS'23]* [[Paper]](https://neurips.cc/virtual/2023/poster/69982) [[Code]](https://github.com/jerry-chee/QuIP)
- OWQ: Lessons learned from activation outliers for weight quantization in large language models. *[arXiv'23]* [[Paper]](http://arxiv.org/abs/2306.02272) [[Code]](https://github.com/xvyaward/owq)
- FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs. *[arXiv'23]* [[Paper]](http://arxiv.org/abs/2308.09723)
- BinaryBERT: Pushing the Limit of BERT Quantization. *[ACL'21]* [[Paper]](https://arxiv.org/abs/2012.15701) [[Code]](https://github.com/huawei-noah/Pretrained-Language-Model)
- I-BERT: Integer-only BERT Quantization. *[ICML'21]* [[Paper]](https://proceedings.mlr.press/v139/kim21d.html) [[Code]](https://github.com/kssteven418/I-BERT)
- Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models. *[NeurIPS'22]* [[Paper]](https://arxiv.org/abs/2209.13325) [[Code]](https://github.com/wimh966/outlier_suppression)
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. *[ICML'22]* [[Paper]](https://arxiv.org/abs/2211.10438) [[Code]](https://github.com/mit-han-lab/smoothquant)
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. *[NeurIPS'22]* [[Paper]](https://nips.cc/Conferences/2022/Schedule?showEvent=54407) [[Code]](https://github.com/microsoft/DeepSpeed)
- Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer. *[NeurIPS'22]* [[Paper]](https://openreview.net/forum?id=fU-m9kQe0ke) [[Code]](https://github.com/yanjingli0202/q-vit)
- RPTQ: Reorder-based Post-training Quantization for Large Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2304.01089) [[Code]](https://github.com/hahnyuan/rptq4llm)
- Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling. *[ACL'23]* [[Paper]](https://aclanthology.org/2023.emnlp-main.102/) [[Code]](https://github.com/ModelTC/Outlier_Suppression_Plus)
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats. *[arXiv'23]* [[Paper]](http://arxiv.org/abs/2307.09782)
- ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2303.08302)
- I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_I-ViT_Integer-only_Quantization_for_Efficient_Vision_Transformer_Inference_ICCV_2023_paper.pdf) [[Code]](https://github.com/zkkli/I-ViT)
- Q-Diffusion: Quantizing Diffusion Models. *[ICCV'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf) [[Code]](https://github.com/Xiuyu-Li/q-diffusion)
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization. *[ISCA'23]* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3579371.3589038)
- QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models. *[arXiv'23]* [[Paper]](http://arxiv.org/abs/2310.08041)
- Integer or floating point? new outlooks for low-bit quantization on large language models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.12356)
- Oscillation-free Quantization for Low-bit Vision Transformers. *[ICML'23]* [[Paper]](https://openreview.net/forum?id=DihXH24AdY) [[Code]](https://github.com/nbasyl/OFQ)
- FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization. *[ICML'23]* [[Paper]](https://openreview.net/forum?id=EPnzNJTYsb)
- OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models. *[arXiv'23]* [[Paper]](http://arxiv.org/abs/2308.13137) [[Code]](https://github.com/OpenGVLab/OmniQuant)
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.17888) [[Code]](https://github.com/facebookresearch/LLM-QAT)
- Compression of generative pre-trained language models via quantization. *[ACL'22]* [[Paper]](https://aclanthology.org/2022.acl-long.331/)
- BitNet: Scaling 1-bit Transformers for Large Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.11453) [[Code]](https://github.com/kyegomez/BitNet)
- QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models. *[arXiv'23]* [[Paper]](http://arxiv.org/abs/2310.08041)
- LLM-FP4: 4-Bit Floating-Point Quantized Transformers. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.16836)
- Atom: Low-bit Quantization for Efficient and Accurate LLM Serving. *[arXiv'23]* [[Paper]](http://arxiv.org/abs/2310.19102)
- Matrix Compression via Randomized Low Rank and Low Precision Factorization. *[NeurIPS'23]* [[Paper]](https://arxiv.org/abs/2310.11028) [[Code]](https://github.com/pilancilab/matrix-compressor)
- TensorGPT: Efficient Compression of the Embedding Layer in LLMs based on the Tensor-Train Decomposition. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2307.00526)
- LORD: Low Rank Decomposition Of Monolingual Code LLMs For One-Shot Compression. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2309.14021)
- ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with Linear Taylor Attention. *[HPCA'23]* [[Paper]](https://ieeexplore.ieee.org/document/10071081) [[Code]](https://github.com/GATECH-EIC/ViTaLiTy)

## Resource-efficient Systems


### Distributed Training

- Optimizing Dynamic Neural Networks with Brainstorm. *[OSDI'23]* [[Paper]](https://www.usenix.org/system/files/osdi23-cui.pdf) [[Code]](https://github.com/Raphael-Hao/brainstorm)
- GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints. *[SOSP'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3600006.3613145)
- Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates. *[SOSP'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3600006.3613152) [[Code]](https://github.com/SymbioticLab/Oobleck)
- Varuna: Scalable, Low-cost Training of Massive Deep Learning Models. *[EuroSys'22]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3492321.3519584) [[Code]](https://github.com/microsoft/varuna)
- HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism. *[ATC'20]* [[Paper]](https://www.usenix.org/system/files/atc20-park.pdf)
- ZeRO-Offload: Democratizing Billion-Scale Model Training. *[ATC'21]* [[Paper]](https://www.usenix.org/system/files/atc21-ren-jie.pdf) [[Code]](https://github.com/microsoft/DeepSpeed)
- Whale: Efficient Giant Model Training over Heterogeneous GPUs. *[ATC'22]* [[Paper]](https://www.usenix.org/system/files/atc22-jia-xianyan.pdf) [[Code]](https://github.com/alibaba/EasyParallelLibrary)
- SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization. *[ATC'23]* [[Paper]](https://www.usenix.org/system/files/atc23-zhai.pdf) [[Code]](https://github.com/zms1999/SmartMoE)
- Behemoth: A Flash-centric Training Accelerator for Extreme-scale DNNs. *[FAST'21]* [[Paper]](https://www.usenix.org/system/files/fast21-kim.pdf)
- FlashNeuron: SSD-Enabled Large-Batch Training of Very Deep Neural Networks. *[FAST'21]* [[Paper]](https://www.usenix.org/system/files/fast21-bae.pdf) [[Code]](https://github.com/SNU-ARC/flashneuron.git)
- Sequence Parallelism: Long Sequence Training from System Perspective. *[ACL'23]* [[Paper]](https://aclanthology.org/2023.acl-long.134.pdf) [[Code]](https://github.com/FrankLeeeee/Sequence-Parallelism)
- Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models. *[ASPLOS'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)
- Mobius: Fine Tuning Large-scale Models on Commodity GPU Servers. *[ASPLOS'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3575693.3575703)
- Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression. *[ASPLOS'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3575693.3575712) [[Code]](https://github.com/MachineLearningSystem/Optimus-CC)
- Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads. *[ASPLOS'22]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3503222.3507778) [[Code]](https://github.com/parasailteam/coconet)
- FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement. *[SIGMOD'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3588964) [[Code]](https://github.com/PKU-DAIR/Hetu)
- On Optimizing the Communication of Model Parallelism. *[MLSys'23]* [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2023/file/a42cbafcabb6dc7ce77bfe2e80f5c772-Paper-mlsys2023.pdf) [[Code]](https://github.com/alpa-projects/alpa)
- Reducing Activation Recomputation in Large Transformer Models. *[MLSys'23]* [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2023/file/e851ca7b43815718fbbac8afb2246bf8-Paper-mlsys2023.pdf) [[Code]](https://github.com/NVIDIA/Megatron-LM)
- PipeFisher: Efficient Training of Large Language Models Using Pipelining and Fisher Information Matrices. *[MLSys'23]* [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2023/file/29d8ab58bcd65e45a831feeaed051d23-Paper-mlsys2023.pdf) [[Code]](https://github.com/kazukiosawa/pipe-fisher)
- Breadth-First Pipeline Parallelism. *[MLSys'23]* [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2023/file/14bc46029b7ac590f56a203e0a3ef586-Paper-mlsys2023.pdf)
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts. *[MLSys'23]* [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2023/file/f9f4f0db4894f77240a95bde9df818e0-Paper-mlsys2023.pdf) [[Code]](https://github.com/stanford-futuredata/megablocks)
- Tutel: Adaptive Mixture-of-Experts at Scale. *[MLSys'23]* [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2023/file/9412531719be7ccf755c4ff98d0969dc-Paper-mlsys2023.pdf) [[Code]](https://github.com/microsoft/tutel)
- Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. *[ICLR'20]* [[Paper]](https://openreview.net/pdf?id=Syx4wnEtvH) [[Code]](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py)
- Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism. *[VLDB'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.14778/3570690.3570697) [[Code]](https://github.com/PKU-DAIR/Hetu-Galvatron)
- MiCS: Near-linear Scaling for Training Gigantic Model on Public Cloud. *[VLDB'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.14778/3561261.3561265)
- Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism. *[ATC'21]* [[Paper]](https://www.usenix.org/system/files/atc21-eliad.pdf) [[Code]](https://github.com/saareliad/FTPipe)
- Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs. *[NSDI'23]* [[Paper]](https://www.usenix.org/system/files/nsdi23-thorpe.pdf) [[Code]](https://github.com/uclasystem/bamboo)
- Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models. *[SIGCOMM'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3603269.3604869)
- MLaaS in the wild: Workload analysis and scheduling in Large-Scale heterogeneous GPU clusters. *[NSDI'22]* [[Paper]](https://www.usenix.org/system/files/nsdi22-paper-weng.pdf) [[Code]](https://github.com/alibaba/clusterdata)
- Zero: Memory optimizations toward training trillion parameter models. *[SC'20]* [[Paper]](https://dl.acm.org/doi/pdf/10.5555/3433701.3433727) [[Code]](https://github.com/microsoft/deepspeed)
- Efficient large-scale language model training on gpu clusters using megatron-lm. *[HPC'21]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3458817.3476209) [[Code]](https://github.com/nvidia/megatron-lm)
- Alpa: Automating inter-and Intra-Operator parallelism for distributed deep learning. *[OSDI'22]* [[Paper]](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf) [[Code]](https://github.com/alpa-projects/alpa)
- Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training. *[ICPC'23]* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3605573.3605613) [[Code]](https://github.com/hpcaitech/ColossalAI)
- Megatron-lm: Training multi-billion parameter language models using model parallelism. *[arXiv'19]* [[Paper]](https://arxiv.org/pdf/1909.08053.pdf) [[Code]](https://github.com/NVIDIA/Megatron-LM)
- Pytorch FSDP: experiences on scaling fully sharded data parallel. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2304.11277.pdf) [[Code]](https://github.com/pytorch/pytorch)
- DeepSpeed. [[URL]](https://www.microsoft.com/en-us/research/project/deepspeed/)
- Huggingface PEFT. [[URL]](https://github.com/huggingface/peft)
- FairScale. [[URL]](https://github.com/facebookresearch/fairscale)
- OpenLLM: Operating LLMs in production. [[URL]](https://github.com/bentoml/OpenLLM)

### Federated Learning

- Flower: A friendly federated learning research framework. *[arXiv'20]* [[Paper]](https://arxiv.org/pdf/2007.14390) [[Code]](https://github.com/adap/flower)
- Fedml: A research library and benchmark for federated machine learning. *[arXiv'20]* [[Paper]](https://arxiv.org/pdf/2007.13518) [[Code]](https://github.com/FedML-AI/FedML)
- FedNLP: Benchmarking Federated Learning Methods for Natural Language Processing Tasks. *[NAACL'22]* [[Paper]](https://arxiv.org/pdf/2104.08815) [[Code]](https://github.com/FedML-AI/FedNLP)
- FATE-LLM: A Industrial Grade Federated Learning Framework for Large Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.10049) [[Code]](https://github.com/FederatedAI/FATE)
- Federatedscope-llm: A comprehensive package for fine-tuning large language models in federated learning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2309.00363) [[Code]](https://federatedscope.io/docs/llm/)
- Federated Self-supervised Speech Representations: Are We There Yet?. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2204.02804)
- Towards Building the Federated GPT: Federated Instruction Tuning. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2305.05644.pdf?utm_referrer=https%3A%2F%2Fdzen.ru%2Fmedia%2Fid%2F5e048b1b2b616900b081f1d9%2F645f43ee7906327d8643ffbc) [[Code]](https://github.com/JayZhang42/FederatedGPT-Shepherd)
- Federated Fine-Tuning of LLMs on the Very Edge: The Good, the Bad, the Ugly. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.03150)
- Privacy-Preserving Fine-Tuning of Artificial Intelligence (AI) Foundation Models with Federated Learning, Differential Privacy, Offsite Tuning, and Parameter-Efficient Fine-Tuning (PEFT). *[TechRxiv'23]* [[Paper]](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.24191886.v1)
- Efficient federated learning for modern nlp. *[MobiCom'23]* [[Paper]](https://xumengwei.github.io/files/MobiCom23-AdaFL.pdf) [[Code]](https://github.com/UbiquitousLearning/FedAdapter)
- Federated few-shot learning for mobile NLP. *[MobiCom'23]* [[Paper]](https://www.caidongqi.com/pdf/MobiCom23-FeS.pdf) [[Code]](https://github.com/UbiquitousLearning/FeS)
- Low-parameter federated learning with large language models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2307.13896) [[Code]](https://github.com/llm-eff/FedPepTAO/blob/main/README.md)
- FedPrompt: Communication-Efficient and Privacy-Preserving Prompt Tuning in Federated Learning. *[ICASSP'23]* [[Paper]](https://ieeexplore.ieee.org/iel7/10094559/10094560/10095356.pdf)
- Reducing Communication Overhead in Federated Learning for Pre-trained Language Models Using Parameter-Efficient Finetuning. *[Conference on Lifelong Learning Agents'23]* [[Paper]](https://proceedings.mlr.press/v232/malaviya23a/malaviya23a.pdf)
- FEDBFPT: An efficient federated learning framework for BERT further pre-training. *[AAAI'23]* [[Paper]](https://dl.acm.org/doi/abs/10.24963/ijcai.2023/483) [[Code]](https://github.com/Hanzhouu/FedBFPT)
- FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning. *[arXiv'22]* [[Paper]](https://arxiv.org/pdf/2208.05174)
- FedBERT: When federated learning meets pre-training. *[TIST'22]* [[Paper]](https://www.researchgate.net/profile/Dezhong-Yao/publication/358359946_FedBERT_When_Federated_Learning_Meets_Pre-Training/links/62d8096345865722d77874e1/FedBERT-When-Federated-Learning-Meets-Pre-training.pdf) [[Code]](https://github.com/siabdullah4/FedBERT/blob/main/README.md)
- FedPerfix: Towards Partial Model Personalization of Vision Transformers in Federated Learning. *[CVPR'23]* [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_FedPerfix_Towards_Partial_Model_Personalization_of_Vision_Transformers_in_Federated_ICCV_2023_paper.pdf) [[Code]](https://github.com/imguangyu/FedPerfix)
- Federated fine-tuning of billion-sized language models across mobile devices. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2308.13894) [[Code]](https://github.com/wuyaozong99/ForwardFL)
- Efficient Federated Prompt Tuning for Black-box Large Pre-trained Models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2310.03123)
- Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2312.06353)

### Serving on Cloud

- Orca: A Distributed Serving System for Transformer-Based Generative Models. *[OSDI'22]* [[Paper]](https://www.usenix.org/conference/osdi22/presentation/yu)
- SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2308.16369)
- Fast Distributed Inference Serving for Large Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.05920)
- FlexGen: high-throughput generative inference of large language models with a single GPU. *[ICML'23]* [[Paper]](https://dl.acm.org/doi/10.5555/3618408.3619696) [[Code]](https://github.com/FMInference/FlexGen)
- DeepSpeed-FastGen. [[URL]](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)
- Splitwise: Efficient Generative LLM Inference Using Phase Splitting. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2311.18677)
- Efficiently Scaling Transformer Inference. *[MLSys'23]* [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2023/file/523f87e9d08e6071a3bbd150e6da40fb-Paper-mlsys2023.pdf)
- DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale. *[SC'22]* [[Paper]](https://ieeexplore.ieee.org/document/10046087)
- FlashDecoding++: Faster Large Language Model Inference on GPUs. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2311.01282)
- Flash-Decoding for long-context inference. [[URL]](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- A High-Performance Transformer Boosted for Variable-Length Inputs. *[IPDPS'23]* [[Paper]](https://www.computer.org/csdl/proceedings-article/ipdps/2023/376600a344/1OSI3YtxzTq) [[Code]](https://github.com/bytedance/ByteTransformer)
- SpotServe: Serving Generative Large Language Models on Preemptible Instances. *[ASPLOS'24]* [[Paper]](https://arxiv.org/abs/2311.15566) [[Code]](https://github.com/Hsword/SpotServe)
- HexGen: Generative Inference of Foundation Model over Heterogeneous Decentralized Environment. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2311.11514) [[Code]](https://github.com/Relaxed-System-Lab/HexGen)
- Punica: Multi-Tenant LoRA Serving. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.18547) [[Code]](https://github.com/punica-ai/punica)
- SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2311.03285) [[Code]](https://github.com/S-LoRA/S-LoRA)
- Efficient Memory Management for Large Language Model Serving with PagedAttention. *[SOSP'23]* [[Paper]](https://dl.acm.org/doi/10.1145/3600006.3613165) [[Code]](https://github.com/vllm-project/vllm)
- Efficiently Programming Large Language Models using SGLang. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2312.07104)

### Serving on Edge

- EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge. *[SenSys'23]* [[Paper]](https://yanzhenyu.com/assets/pdf/EdgeFM-SenSys23.pdf)
- EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2308.14352.pdf)
- Serving MoE Models on Resource-constrained Edge Devices via Dynamic Expert Swapping. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2308.15030.pdf)
- LLMCad: Fast and Scalable On-device Large Language Model Inference. *[arXiv'23]* [[Paper]](https://arxiv.org/pdf/2309.04255.pdf)
- STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining. *[ASPLOS'23]* [[Paper]](https://dl.acm.org/doi/10.1145/3575693.3575698)
- Practical Edge Kernels for Integer-Only Vision Transformers Under Post-training Quantization. *[MLSys'23]* [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2023/file/023560744aae353c03f7ae787f2998dd-Paper-mlsys2023.pdf)
- Powerinfer: Fast large language model serving with a consumer-grade gpu. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2312.12456) [[Code]](https://github.com/SJTU-IPADS/PowerInfer)
- Llm in a flash: Efficient large language model inference with limited memory. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2312.11514)
