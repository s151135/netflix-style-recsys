# Data and media credits

## MovieLens

This educational project uses MovieLens latest small from the GroupLens Research
Group at the University of Minnesota. The dataset snapshot contains 100,836 ratings
from 610 anonymized users. The original dataset's complete README, usage conditions
and citation are preserved in [notices/MOVIELENS_README.txt](notices/MOVIELENS_README.txt).
They also apply to redistributed dataset transformations, including demo histories.
No endorsement by GroupLens or the University of Minnesota is implied.

Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets:
History and Context*. ACM Transactions on Interactive Intelligent Systems 5, 4.
https://doi.org/10.1145/2827872

Source: https://grouplens.org/datasets/movielens/

The upstream dataset conditions limit use to research and require permission for
commercial or revenue-bearing purposes. This release is prepared as an educational
portfolio demonstration. The data are not relicensed by this project.

## Optional TMDB artwork

The local catalogue inherited a small collection of TMDB image-path references.
No poster or backdrop image files are bundled, and the release UI does not load
remote artwork by default. It uses local text cards instead. Catalogue descriptions
and credits are a manually maintained demonstration snapshot, not a live feed.

Setting `RECSYS_ENABLE_REMOTE_ARTWORK=1` enables the legacy references. Before using
this option on a public deployment, supply appropriate provider authorization,
credits and the approved TMDB logo. TMDB documents its attribution and usage
conditions at https://developer.themoviedb.org/docs/faq and image URL format at
https://developer.themoviedb.org/docs/image-basics. This project does not claim that
image URL access grants usage rights or that this optional integration is complete.

## Naming and software

Netflix is referenced only as project inspiration. ReelFind is this demo's display
name. Neither Netflix nor TMDB endorses this project.

Python dependencies retain their own licenses. The lock files record the versions
used for release verification; this notice does not change those licenses or grant
a new license to the project's original source code.
