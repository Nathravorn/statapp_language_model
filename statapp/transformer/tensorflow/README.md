## Quelques notes sur `regr_lin_model`

* Privilégier le calcul vectoriel à la boucle `for` :
    Il faut passer tout le dataset (ou un batch) et pas x,y au compte-goutte...
* En commentaire : OptimizerAdam. Le code marche, il reste à savoir quoi mettre comme paramètre
* Ne pas prendre un intervalle avec des valeurs trop élevées sinon :
    * les valeurs explosent très rapidement
    * on perd en précision ! (plus de précisions vers 0) 
