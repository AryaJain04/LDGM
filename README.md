# Identifying gene regulatory network rewiring using Latent Differential Graphical Models (LDGM)

## Brief Description
LDGM is a novel approach for estimating tissue-specific gene regulatory networks directly, without the need to infer networks for individual tissues. It focuses on capturing network rewiring between different tissue types, offering efficiency and reliability advantages, particularly with smaller sample sizes.

## Reference Publication
This is the link to the publication by Dechao Tian, Quanquan Gu, Jian Ma https://doi.org/10.1093/nar/gkw581

## Installation Instructions
To run the code by author 'LDGM_reproduced ' use MATLAB Simuline Online. To run the other code, use Jupyter Notebook.  

## Execution Instructions
To execute LDGM with example data, follow these steps:

1. Download the example data from https://github.com/bionetslab/grn-benchmark/tree/main/reference_datasets.
2. Navigate to the directory where LDGM folder is prresent in the current repositoryy.
3. Run the LDGM code and modify as per the requirement and the data used.

Replace `[input_file]` with the path to the input data file and `[output_file]` with the desired path for the output file.

## Relevant Parameters
You might need to adjust the DGLoss function, differential_graph, L1GeneralProjectedSubGradient in code to shape the matrix and handle the large amount of data. 

## Input File/Output File Format Specification
The input data is in the .tsv and .txt format. Any input format can be used and code can modified accordingly.

## Explanation of Output
The output file, named network.tsv, is a tab-separated file containing all edges row-wise, with the following columns:

Target: Represents the target of the edge.

Regulator: Denotes the source of the edge.

Condition: Specifies the condition to which the edge belongs.

Weight: Indicates the weight of the edge.

## Other Necessary Information
Please make sure to check if your data follows Gaussian Distribution and if 'yes' then create the correlation matrix using Pearson Correlation matrix and if 'no' then create the correlation matrix using Latent Correlation matrix. 
