# ESM-AlphaFold-GO

## enviroment and platform
    Some works on features used Google Colab.
    traing platform using AliCloud, with Nvidia A10 GPU
    enviroment is the same as ./enviroment.yml which provided by the author of Struct2GO

## results 
    some traing and test log could be found in the ./results/log/, but some of them are lost, because I 
    own AliCloud the fee, before I download all the logs, they deleted my workspace. Sorry for that.
    But fortunately, I have downloaded all the test data, you can also just check them at ./results/best_models/bestmodel_test, and best models at ./results/best_models.

## key steps
    1. Process the struct data(pdb files) using ./data_processing/predicted_protein_struct2map.py
    2. Get node embeddiings, refer to ./data_processing/node2vec.ipynb and /read_node2vec_feature.py
    3. Download sequence data from UniProt, and then find out ESM-2, downlod their open-sorce pretrained model (esm2_t33_650M_UR50D), refer to ./data_processing/ESeq2Vec.ipynb;
    4. Run labels_load.py in the path model/. remember the tags number of each GO catgory, we need that in the training part
    5. run trian_Struct2GO2.py, fill the parameters.
