AAAI-2024

Data format: All data files are stored in pkl format, the ground truth is the [" id ", "community"] dictionary, and the input data is a series of time series graphs

cite: D. Kong, A. Zhang, and Y. Li, “Learning persistent community structures in dynamic networks via topological data analysis,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 8, 2024,
pp. 8617-8626.

Abstract: Dynamic community detection methods often lack effective mechanisms to ensure temporal consistency, hindering the analysis of network evolution. In this paper, we propose a novel deep graph clustering framework with temporal consistency regularization on inter-community structures, inspired by the concept of minimal network topological changes within short intervals. Specifically, to address
the representation collapse problem, we first introduce MFC, a matrix factorization-based deep graph clustering algorithm that preserves node embedding. Based on static clustering results, we construct probabilistic community networks and compute their persistence homology, a robust topological measure, to assess structural similarity between them. Moreover, a novel neural network regularization TopoReg is introduced to ensure the preservation of topological similarity between inter-community structures over time intervals.
Our approach enhances temporal consistency and clustering accuracy on real-world datasets with both fixed and varying numbers of communities. It is also a pioneer application of TDA in temporally persistent community detection, offering an insightful contribution to field of network analysis
