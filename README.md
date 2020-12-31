# learning_RNA_3way_junctions
 
The goal of this project is to write a Python script that uses machine learning to take in RNA sequences and base pairing data for three-way junctions to try and predict the alignment of the stems A, B, and C.

The input would be an RNA sequence and the base pair information. The features would be the sequence of stems A, B, C and the sequence of the junctions connecting the stems, X, Y, Z.

I will need to generate a training set by hand as most online databases aren't create to specifically provide information on specific features such as RNA 3-way junctions.

The training set will be small so this will be an exercise in machine learning algorithms rather than a conclusive study.

### Machine Learning

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)


### Limitations
Both performed poorly with Bayes having 50% accuracy while neural had 38%. The dataset is small and the variability is high.

### Improvements
use basepairs to distinguish stems instead of only the sequences, see Alu SRP where to helices are paired together.

although I hope this doesn't weight actual nucleotides more than others from the ordinal encoding
"ZGUCA" --> [0.00, 0.25, 0.5, 0.75, 1.0] 

Optional one-hot encoding for future deep learning method
"ACGUZ" --> [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]


References:
1. Lescoute A, Westhof E. Topology of three-way junctions in folded RNAs. RNA. 2006;12(1):83-93. doi:10.1261/rna.2208106
2. Byron K, Wang JT. A Computational Approach to Finding RNA Tertiary Motifs in Genomic Sequences. 2017;arXiv:1701.00435.
3. Bindewald E, Hayes R, Yingling YG, Kasprzak W, Shapiro BA. RNAJunction: a database of RNA junctions and kissing loops for three-dimensional structural analysis and nanodesign. Nucleic Acids Res. 2008;36(Database issue):D392-D397. doi:10.1093/nar/gkm842