<p align="center">
	<img src="https://github.com/daiquocnguyen/SANNE/blob/master/sanne_logo.png">
</p>

## From Random Walks to Transformer for Node Embeddings<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FSANNE%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/SANNE"><a href="https://github.com/daiquocnguyen/SANNE/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/SANNE"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/SANNE">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/SANNE">
<a href="https://github.com/daiquocnguyen/SANNE/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/SANNE"></a>
<a href="https://github.com/daiquocnguyen/SANNE/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/SANNE"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/SANNE">

This program provides the implementation of our unsupervised node embedding model SANNE as described in [our paper]() whose central idea is to employ a transformer self-attention network to iteratively aggregate vector representations of nodes in sampled random walks.

<p align="center">
	<img src="https://github.com/daiquocnguyen/SANNE/blob/master/SANNE.png">
</p>

## Usage

### Requirements
- Python 3
- Tensorflow >= 1.6
- Tensor2tensor >= 1.9
- scikit-learn

### Training

To run the program in the transductive setting:

	$ python train_SANNE.py --embedding_dim 128 --name cora.16.8.trans.pickle --batch_size 64 --num_sampled 512 --num_epochs 50 --num_hidden_layers 2 --num_heads 4 --learning_rate 0.00005 --model_name cora_squash_trans_168_bs64_2_4_3

To run the program in the inductive setting:

	$ python train_SANNE_ind.py --embedding_dim 128 --name pubmed.128.8.ind5.pickle --idx_time 5 --nameTrans pubmed.128.8.trans.pickle --batch_size 64 --num_sampled 512 --num_epochs 50 --saveStep 2 --num_hidden_layers 8 --num_heads 4 --learning_rate 0.00005 --model_name pubmed_ind5_1288_bs64_8_4_3
	
### Notes

It is important to note that you should report the mean and standard deviation of accuracies over 10 different times of sampling training/validation/test sets when comparing models. File `utils.py` has a function `sampleUniformRand` to randomly sample 10 different data splits of training/validation/test sets. I also include my 10 different data splits in `dataset_name.10sampledtimes`. You can see command examples in `commands.txt`.

## Cite

Please cite the paper whenever SANNE is used to produce published results or incorporated into other software:

	@InProceedings{Nguyen2018SANNE,
		author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
		title={{A Self-Attention Network based Node Embedding Model}},
		booktitle={...},
		year={2018}
	}
## License

As a free open-source implementation, SANNE is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

SANNE is licensed under the Apache License 2.0.
