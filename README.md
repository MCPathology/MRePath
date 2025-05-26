MRePath
===========

<details>
<summary>
  <b>Multimodal Cancer Survival Analysis via Hypergraph Learning with Cross-Modality Rebalance</b>, IJCAI 2021.
  <a href="https://arxiv.org/abs/2505.11997" target="blank">[HTML]</a>
  <br><em>Mingcheng Qu, Guang Yang, Donglin Di, Tonghua Su, Yue Gao, Yang Song, Lei Fan*</em></br>
</summary>

```bash
@inproceedings{chen2021multimodal,
  title={Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images},
  author={Chen, Richard J and Lu, Ming Y and Weng, Wei-Hung and Chen, Tiffany Y and Williamson, Drew FK and Manz, Trevor and Shady, Maha and Mahmood, Faisal},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4015--4025},
  year={2021}
}
```
</details>

**Summary:** We propose a multimodal survival prediction framework that incorporates hypergraph learning to effectively capture both contextual and hierarchical details from pathology images. Moreover, it employs a modality rebalance mechanism and an interactive alignment fusion strategy to dynamically reweight the contributions of the two modalities, thereby mitigating the pathology-genomics imbalance.
<img src="model.jpg" width="1500px" align="center" />

### Downloading TCGA Data
To download diagnostic WSIs (formatted as .svs files), molecular feature data and other clinical metadata, please refer to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/) and the [cBioPortal](https://www.cbioportal.org/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/).

## Installation Guide for Linux (using anaconda)

### Pre-requisites:
* Linux (Tested on Ubuntu 22.04) 
* NVIDIA GPU (Tested on Nvidia GeForce RTX 4090 Ti) with CUDA 12.1
* Python (3.10.7)


## Processing Whole Slide Images
To process WSIs, first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, a pretrained truncated ResNet50 is used to encode raw image patches into 1024-dim feature vectors, which we then save as .pt files for each WSI. The extracted features then serve as input (in a .pt file) to the network. The following folder structure is assumed for the extracted features vectors:    
```bash
DATA_ROOT_DIR/
    └──TCGA_BLCA/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_BRCA/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_GBMLGG/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_LUAD/
        ├── slide_1.ptd
        ├── slide_2.pt
        └── ...
    └──TCGA_UCEC/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    ...
```
DATA_ROOT_DIR is the base directory of all datasets / cancer type(e.g. the directory to your SSD). Within DATA_ROOT_DIR, each folder contains a list of .pt files for that dataset / cancer type.


## Molecular Features and Genomic Signatures
Processed molecular profile features containing mutation status, copy number variation, and RNA-Seq abundance can be downloaded from the [cBioPortal](https://www.cbioportal.org/), which we include as CSV files in the [following directory](https://github.com/iccv2021anon/10972/tree/master/datasets_csv_sig). For ordering gene features into gene embeddings, we used the following categorization of gene families (categorized via common features such as homology or biochemical activity) from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/gene_families.jsp?ex=1). Gene sets for homeodomain proteins and translocated cancer genes were not used due to overlap with transcription factors and oncogenes respectively. The curation of "genomic signatures" can be modified to curate genomic embedding that reflect unique biological functions.
