# Image-Retrieval
Image Retrieval through Unsupervised Hypergraph-Based Manifold Ranking as described in the paper [MultimediaRetrieval throughUnsupervised Hypergraph-basedManifoldRanking](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ic.unicamp.br/~dcarlos/papers/journals/IEEE-TIP_2019-AcceptedVersion.pdf). The algorithm is compared with the results that were retrieved from the extracted features by employing the FAISS (Facebook AI Similarity Search) library, which is also implemented. A subset of the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) was utilized to retrieve images before and after fine-tuning the backbone architectures (VGG-16,VGG-19,ResNet-50,ResNet-101).

<table>
<tr><th> </th><th> </th></tr>
<tr><td>

## Retrieval Accuracy
|Metric| Pre-trained (ImageNet) | Fine-Tuned (Animal Dataset)|
|--|--|--|
|Precision| 0.8534|0.8993 |
|Recall| 0.99|0.9833 |

</td><td>

|Metric| Paper | FAISS|
|--|--|--|
|Precision| 0.8993|0.9493 |
|Recall| 0.9833|1.0 |

</td></tr> </table>

## Retrieved Results

![results](https://drive.google.com/uc?export=view&id=1HYtcHHxEXIHh_pPTqSwkq8VkzOl9OqiI)
