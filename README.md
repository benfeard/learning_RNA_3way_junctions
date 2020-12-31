# Can a Machine Learn RNA 3-way Junctions?
 
The goal of this project is to write a Python script that uses machine learning to take in RNA sequences and base pairing data for three-way junctions to predict the alignment of the stems A, B, and C. RNA structure prediction is a difficult but important field. RNA play a fundamental role in biology and from a biophysical perspective, their structure can teach us about their function in a variety of fields from developmental biology to cancer. Current approaches use experimental data from NMR, X-Ray crystallography, chemical modifications, or molecular dynamics simulations.

I want to use an RNA sequence and the base pair information to focus on a fundamental feature of RNA tertiary structures. Junctions play a large roll in defining tertiary RNA structure. Beyond primary (nucleotide sequence) and secondary (base pairing and helices) structures, the tertiary structure can help us understand how RNA can physical interact with its environment. The simplest junction contains two stems (helices) but lacks the interesting three dimensional alignment questions. A junction containing three stems falls into distinct structural conformatinos that can provide useful, functional information about an RNA. I will breakdown a three-way junction into features for machine learning that consist of the sequence of stems A, B, and C and the sequence of the junctions connecting the stems a, b, and c.

![RNA 3-Way Junctions described in Lescoute et al.](https://github.com/benfeard/learning_RNA_3way_junctions/blob/main/Figures/Figure%201.png "RNA 3-Way Junctions described in Lescoute et al.")
**RNA 3-Way Junctions described in Lescoute et al.[1]**

I generated a training set by hand as most online databases aren't created to specifically provide information on specific features such as RNA 3-way junctions. The training set will be small so this will be an exercise in how to use machine learning algorithms to predict RNA tertiary orientation rather than to solve the problem outright.

### Machine Learning Algorithms

I used Scikit-learn's library to implement a Neural Network with three hidden layers with 10 hidden units each and a Naive Bayes classifer.

To format the labelled RNA structures as data for the algorithms, I used an ordinal encoding approach where each nucleotide GUCA were encoded as numbers 0.25, 0.50, 0.75, and 1.00. I also included a new pseudo-nucleotide "Z" which represents empty sequences. Sometimes the junctions connecting stems have a length of zero but still need to be encoded. Z was converted to 0.00 and used in all the stems and junctions as padding to make them 64 characters each. This was then combined to make one large sequence for each RNA.

### Limitations

Running the code with the dataset, both machine learning algorithms performed poorly with the Naive Bayes classifier having 50% accuracy and the Neural Networking having 38% accuracy. This was expected as the dataset is small with only 32 RNA (9 in Family A, 6 in Family B, and 17 in Family C). Additionally, RNA sequences are variable in nucleotides and length. A much larger dataset would be needed to accurately predict how these sequences might share common information regarding their tertiary alignment as three-way junctions.

### Improvements

- The Neural Network layers and hidden units could be tweaked and optimized to increase accuracy.
- More testing would be needed to know if the ordinal encoding unintentially biases particular nucleotides due to them having unequal numerical values. This could be addressed by using a one-hot encoding solution such as "ACGUZ" --> [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]
- The lack of specific base pair information to distinguish stems (as opposed to the nucleotide sequence of stems in order) could be limiting the secondary structure information. Ex: Alu SRP[1] has two helices paired together and this is not capture in the dataset. However, including base pair information would require new decisions such as to use only canonical base pairing or beyond and would need to address how to capture that information as numerical values and not just patterns used in Dot-Bracket Notation ((((....))))

![Alu SRP from Lescoute et al.](https://github.com/benfeard/learning_RNA_3way_junctions/blob/main/Figures/Figure%202%20v2.png "Alu SRP from Lescoute et al.")
**Alu SRP from Lescoute et al.[1]**
**Connected stems highlighted by yellow circle.**

###References:
1. Lescoute A, Westhof E. Topology of three-way junctions in folded RNAs. RNA. 2006;12(1):83-93. doi:10.1261/rna.2208106
2. Byron K, Wang JT. A Computational Approach to Finding RNA Tertiary Motifs in Genomic Sequences. 2017;arXiv:1701.00435.
3. Bindewald E, Hayes R, Yingling YG, Kasprzak W, Shapiro BA. RNAJunction: a database of RNA junctions and kissing loops for three-dimensional structural analysis and nanodesign. Nucleic Acids Res. 2008;36(Database issue):D392-D397. doi:10.1093/nar/gkm842