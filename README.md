# FashionColor-0

## Color extraction from clothing images.

Color modelling and extraction is an important topic in fashion. It can help build a wide range of applications, for example, recommender systems, color-based retrieval, fashion design, etc. We aim to develop and test models that can extract the dominant colors of clothing and accessory items. The approach we propose has three stages: (1) Mask-RCNN to segment the clothing items, (2) cluster the colors into a predefined number of groups, and (3) combine the detected colors based on the hue scores and the probability of each score. We use Clothing Co-Parsing and ModaNet datasets for evaluation. We also scrape fashion images from the WWW and use our models to discover the fashion color trend. Subjectively, we were able to extract colors even when clothing items have multiple colors. Moreover, we are able to extract colors along with the probability of them appearing in clothes. The method can provide the color baseline drive for more advanced fashion systems.

See Probabilistic color modeling of clothing elements in http://www.tara.tcd.ie/handle/2262/93334

## Python 3.7

## Dependencies:
- https://pytorch.org/
- https://pypi.org/project/opencv-python/
- https://pypi.org/project/scikit-learn/ 



## Fig.1: Color extraction
![Fig.1](https://github.com/morawi/FashionColor-0/blob/main/Figures/Fig1.png)

## Fig.2: Color extraction
![Fig.2](https://github.com/morawi/FashionColor-0/blob/main/Figures/Fig2.png)

## Fig.3: Color extraction
![Fig.3](https://github.com/morawi/FashionColor-0/blob/main/Figures/Fig3.png)

## Fig.4: Color extraction
![Fig.4](https://github.com/morawi/FashionColor-0/blob/main/Figures/Fig4.png)

## Fig.5: Color distribution
![Fig.5](https://github.com/morawi/FashionColor-0/blob/main/Figures/Fig5.png)


## Fig.6: Extracting seasonal color trend from data
![Fig.6](https://github.com/morawi/FashionColor-0/blob/main/Figures/Fig6.png)



