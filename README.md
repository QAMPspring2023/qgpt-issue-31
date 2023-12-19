# qgpt-issue-31
This repository contains the code, machine learning models, and datasets used in the paper ["A Survey of Classical and Quantum Sequence Models"]()

This study is a review and comprehensive analysis of various classical and quantum sequence models. It also highlights and delves deep into some of the most recent developments in quantum sequence models. Particular attention has been placed on Quantum Self Attention Nueral Networks(QSANN) and Quantum Recurrent Nueral Networks(QRNN). We implement these models and parallely compare them with their classical counterparts. We also evaluate the performance of quantumm and classical self attention nueral networks on vision related tasks. The implementations are as follows:
+ Quantum Self Attention Nueral Networks (QSANN)
  - Source : [Li, Guangxi, Xuanqiang Zhao, and Xin Wang. "Quantum self-attention neural networks for text classification." arXiv preprint arXiv:2205.05625 (2022).](https://arxiv.org/abs/2205.05625)
+ Quantum Vision Transformer (QVT)
  - Source : [Li, Guangxi, Xuanqiang Zhao, and Xin Wang. "Quantum self-attention neural networks for text classification." arXiv preprint arXiv:2205.05625 (2022).](https://arxiv.org/abs/2205.05625)
+ Classical Vision Transformer (CVT)
+ Classical Transformer
  - Source : [https://github.com/rdisipio/qtransformer](https://github.com/rdisipio/qtransformer)
+ Quantum Recurrent Nueral Networks (QRNN)
  - Source : [Li, Yanan, et al. "Quantum Recurrent Neural Networks for Sequential Learning." arXiv preprint arXiv:2302.03244 (2023).](https://arxiv.org/abs/2302.03244)
+ Classical Recurrent Nueral Networks 

# Datasets used
+ Meaning Classification(MC)
  - Paper : [Lorenz, Robin, et al. ”QNLP in practice: Running compositional models of meaning on a quantum computer.” Journal of Artificial Intelligence Research 76 (2023): 1305-1342](https://www.jair.org/index.php/jair/article/view/14329)
  - Source : [https://github.com/CQCL/qnlp_lorenz_etal_2021_resources/tree/main/datasets](https://github.com/CQCL/qnlp_lorenz_etal_2021_resources/tree/main/datasets)
+ RELPRON
  - Paper : [Rimell, Laura, et al. ”RELPRON: A relative clause evaluation data set for compositional distributional semantics.” Computational Linguistics 42.4 (2016): 661-701](https://direct.mit.edu/coli/article-abstract/42/4/661/1555)
  - Source :  [https://www.repository.cam.ac.uk/items/31732619-285c-4f78-b1aa-c82f513e9484](https://www.repository.cam.ac.uk/items/31732619-285c-4f78-b1aa-c82f513e94840)
+ Sentiment Labelled Sentences
  - Paper : [Kotzias, Dimitrios, et al. "From group to individual labels using deep features." Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining. 2015.](https://dl.acm.org/doi/abs/10.1145/2783258.2783380)
  - Source : [Kotzias,Dimitrios. (2015). Sentiment Labelled Sentences. UCI Machine Learning Repository. https://doi.org/10.24432/C57604.](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
+ Optical Recognition of Handwritten Digits
  - Source : [https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
+ MNIST
  - Source : ["THE MNIST DATABASE of handwritten digits". Yann LeCun, Courant Institute, NYU Corinna Cortes, Google Labs, New York Christopher J.C. Burges, Microsoft Research, Redmond.](http://yann.lecun.com/exdb/mnist/)
+ Fashion MNIST
  - Paper : [Xiao, Han, Kashif Rasul, and Roland Vollgraf. "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms." arXiv preprint arXiv:1708.07747 (2017).](https://arxiv.org/abs/1708.07747)
  - Source : [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
 
# Folder hierarchy
```
.
├── Checkpoint presentations
│   └── QAMP_31_first_checkpoint_1_final.pptx
├── Classical_Transformer.ipynb
├── Datasets
│   ├── MC RP Dataset
│   │   ├── mc_dev_data.txt
│   │   ├── mc_test_data.txt
│   │   ├── mc_train_data.txt
│   │   ├── rp_test_data.txt
│   │   └── rp_train_data.txt
│   └── Sentiment Labelled Sentences Dataset
│       ├── amazon_cells_labelled.txt
│       ├── imdb_labelled.txt
│       ├── readme.txt
│       └── yelp_labelled.txt
├── Presentations shared
│   ├── Classical_attention_survey_Anu.pdf
│   ├── QML_Image_Encoding_paper summary.pptx
│   ├── QRL _35_ppt_QAMP.pptx
│   ├── QRNN_QAMP.pptx
│   ├── Transformers_Presentation_Anu.pdf
│   └── gpt models.pdf
├── QRNN
│   ├── Amp_encoding_QRNN.ipynb
│   ├── QRNN.ipynb
│   ├── QRNN_PENNY_TFIDF.ipynb
│   ├── QRNN_Pennylane.ipynb
│   ├── QRNN_QISKIT_TFIDF.ipynb
│   └── QRNN_qiskit.ipynb
├── QRNN Image Classification.ipynb
├── QSANN codes
│   ├── Modified_QSANN_pennylane_w_pred_trained_model.ipynb
│   ├── QSANN_pennylane.ipynb
│   ├── QSANN_qiskit.ipynb
│   ├── QSANN_qiskit_with_preprocessor.ipynb
│   └── Qsann_with_preprocessor.ipynb
├── QTT
│   ├── Feat_eco_model_222111_vocab_size20_mc
│   ├── Feat_eco_model_222121_vocab_size100_rp
│   └── QSANN_qiskit_experiment_pos_enco.ipynb
├── QVT
│   ├── Quantum Vision Transformer-PennyLane-Binary.ipynb
│   └── Quantum Vision Transformer-PennyLane-MutliClass.ipynb
├── README.md
├── RNN.ipynb
└── Survey_plot.ipynb
```

# Installation Instructions
Get the code :

```
git clone https://github.com/QAMPspring2023/qgpt-issue-31.git
```
