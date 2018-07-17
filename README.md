# Predicting the subcellular location of eukaryotic proteins

Coursework of COMPGI10 Bioinformatics (MSc Machine Learning optional module)

## Motivation 
The subcellular location of eukaryotic proteins can provide insights into the function of proteins. Recurrent Neural Networks (RNNs) have shown to perform well on classification tasks when sequential data is observed. This paper explores the use of RNNs in classifying proteins as either cytosolic, secreted, nuclear or mitochondrial on the basis of only the sequence of underlying amino acids. The focus of this paper is on finding an RNN architecture that is both high performing and tractable using limited computational resources.

## Results
Using a special type of RNNs, a GRU (Gated Recurrent unit) with a single 64 unit layer, it is possible to achieve a cross-validated prediction accuracy of 69.58% with a 95% confidence interval of [68.36; 70.80]. As a by-product, the method also yields low dimensional vector representations of amino acids which allows to cluster similar amino acids together.


## Network architecture
[![network.png](https://s20.postimg.org/reiopnt5p/network.png)](https://postimg.org/image/uy4mfgvvd/)

