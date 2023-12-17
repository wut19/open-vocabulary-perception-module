# Open Vocabulary Perpection Module

1. Clone [segment anything](https://github.com/facebookresearch/segment-anything) 
2. Cd `robocraft-mash-recon`, `segment-anything` and `vild` respectively and `pip install -e .`
3. Download `vild` model, `clip` model and `sam` model, and modify the path to the model in `vild/vild.py` and `segment-anything/segment/segment.py`
4. Customize your data by replace the data in obervations with you own data(Notice follow the same data structure). 