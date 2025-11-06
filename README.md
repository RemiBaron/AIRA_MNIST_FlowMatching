# MNIST image generation with Flow Matching 

Cher Gabriele Facciolo,

Voici notre repo GitHub pour notre projet Image. Ici nous proposons une implémentation de Flow Matching sur le dataset MNIST.

## Informations importantes

Pour avoir une démo de génération d'image faite par notre modèle, nous vous recommandons fortement de regarder demo.ipynb.

A part ça, pour résumer, on implémente Flow Matching en se collant sur le papier l'introduisant, donc en réutilisant les mêmes équations. Notre modèle, un Unet, apprend donc le champs de vecteur des images MNIST. En intégrant ceci et en l'appliquant sur une image de bruit pur, on peut générer des images. On obtient par exemple ceci.

![Moults images où on peut distinguer des chiffres flous](generated_images/images_boosted_par_collab(V4).png?raw=true "Exemple d'images générées")
![Moults images où on peut distinguer des chiffres flous, en normalisé](generated_images/images_boosted_par_collab(V4_nomalized).png?raw=true "Exemple d'images générées")

## Fichiers
- Pour installer les dépendances :
```bash
pip install requirements
```

- Pour entraîner le modèle :
```bash
python train.py
```

- Pour tester le modèle en générant des images :
```bash
python test.py
```

L'architecture du modèle utilisé peut être trouvée dans unet.py. Les checkpoints des modèles entraînés par nos soins peuvent être trouvé dans saved_models/.

## Conclusion

Kind regards,

Abel Verley et Rémi Baron