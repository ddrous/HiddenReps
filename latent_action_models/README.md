## Keeping Track of the Baselines

Our WARP has three components:
1) Context-Selection (to force the model to use the actions)
2) A/B Forward Dynamics Split (to disambiguate the dynamics )
2) Encoding (to embed things in a lower compressed space)

We trained specifically trained both our models and baseline WM to never use the actions (context-ratio=0 during training). The baselines also had A/B splitting and Encoding. Additionnaly, the model had NO stop-gradient operator on the encoding. (all future experiments have that)
- WARP-MNIST: 260228-032228-MNIST-WARP-Great*
- WARP-Weather: 260301-024354-Weather-WARP*
- WM-MNIST: 260228-101851-MNIST-WM*
- WM-Weather: 260301-094748-Weather-WM*

We trained baselines with maximum context ratio 1 (as current WM are trained). These baselines have no A/B spliting, but some Encoding. 
- WM-MNIST: 260301-230842-WM-MNIST
- WM-Weather: 260301-204843-WM-Weather

We trained our model and a baseline with a random context ratio (the goal standard). These baselines have no A/B splitting, but some Encoding
- WARP-MNIST:
- WARP-Weather:
- WM-MNIST:
- WM-Weather:

We must train the baselines with no A/B splitting Encoding, and NO Encoding.
- WM-MNIST:
- WM-Weather:
