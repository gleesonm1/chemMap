# chemMap
Tools to extract quantitative chemical information from chemical map data.

This tool is designed to work with the outputs produced by AZtec (Oxford Instruments), but should be easily adaptable for other data formats.

At the moment chemMap remains in early development. More information will be placed here in the near future.

Currently, chemMap can be used to 

1. Load in quatitative (wt%) or qualitative (raw count) chemcial map data
2. Calculate common ratios used in geological samples from either wt% or count data (e.g., Mg#, An).
3. Perform cluster analysis on the map data.
4. Plot chemical maps, specifying a particular element or ratio. With cluster analysis performed first it is possible to produce these chemical maps for a single phase.
5. Extract a transect from a chemical map.
6. Separate crystals into subpopulations based on their size (cluster analysis must be performed first).

2 examples are currently provided in this repository. One focuses on a lava sample from Floreana in the southern Galapagos and, using raw count data, we identify different olivine populations (based on size) and calculate their forsterite content. The second provides quantitative map data from a gabbroic xenolith collected from Hualali volcano, Hawaii. Using this we can identify crystal populations based on composition and extract a quantitative chemical transect across a crystal.
