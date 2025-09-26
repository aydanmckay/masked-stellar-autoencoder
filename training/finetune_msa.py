import yaml
import h5py
import argparse
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from astropy.table import Table
import pandas as pd


from models.model import make_model, TabResnetWrapper

def main():

    parser = argparse.ArgumentParser(description="Train MSA")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    # load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seeds = np.random.randint(0, 1000, size=100).tolist()

    featurescaler = RobustScaler()

    # loading the finetuning dataset
    data = Table.read('/arc/projects/k-pop/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits').to_pandas()
    errordata = data.copy()
    errordata = errordata[errorcols]
    errordata = errordata.fillna(errordata.max())

    trainset, validset, etrainset, evalidset = train_test_split(data.to_numpy(), errordata.to_numpy(), test_size=0.2, random_state=42)
    validset, testset, evalidset, etestset = train_test_split(validset, evalidset, test_size=0.33, random_state=42)
    target_train = trainset[:, :10]
    trainset = trainset[:, 10:]
    target_valid = validset[:, :10]
    validset = validset[:, 10:]
    target_test = testset[:, :10]
    testset = testset[:, 10:]

    # scaling the targets (individually in case of single task finetuning) and features
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaler3 = StandardScaler()
    scaler4 = StandardScaler()
    scaler5 = StandardScaler()
    label1 = scaler1.fit_transform(target_train[:, 0].reshape(-1, 1))
    label2 = scaler2.fit_transform(target_train[:, 2].reshape(-1, 1))
    label3 = scaler3.fit_transform(target_train[:, 4].reshape(-1, 1))
    label4 = scaler4.fit_transform(target_train[:, 6].reshape(-1, 1))
    label5 = scaler5.fit_transform(target_train[:, 8].reshape(-1, 1))
    elabel1 = target_train[:, 1] / scaler1.scale_
    elabel2 = target_train[:, 3] / scaler2.scale_
    elabel3 = target_train[:, 5] / scaler3.scale_
    elabel4 = target_train[:, 7] / scaler4.scale_
    elabel5 = target_train[:, 9] / scaler5.scale_
    target_set = target_test[:, [0, 2, 4, 6, 8]]

    vlabel1 = scaler1.transform(target_valid[:, 0].reshape(-1, 1))
    vlabel2 = scaler2.transform(target_valid[:, 2].reshape(-1, 1))
    vlabel3 = scaler3.transform(target_valid[:, 4].reshape(-1, 1))
    vlabel4 = scaler4.transform(target_valid[:, 6].reshape(-1, 1))
    vlabel5 = scaler5.transform(target_valid[:, 8].reshape(-1, 1))
    velabel1 = target_valid[:, 1] / scaler1.scale_
    velabel2 = target_valid[:, 3] / scaler2.scale_
    velabel3 = target_valid[:, 5] / scaler3.scale_
    velabel4 = target_valid[:, 7] / scaler4.scale_
    velabel5 = target_valid[:, 9] / scaler5.scale_

    # ---------------------------------------------------------------------
    with open('/arc/home/aydanmckay/bprp_mae/final_model/ensembling/ft-sweep-params-modified-0602.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # ---------------------------------------------------------------------

    scaler6 = StandardScaler()
    label6 = scaler6.fit_transform(trainset[:, -4].reshape(-1, 1))
    elabel6 = etrainset[:, -4] / scaler6.scale_
    vlabel6 = scaler6.transform(validset[:, -4].reshape(-1, 1))
    velabel6 = evalidset[:, -4] / scaler6.scale_

    labelled_set = np.concatenate([label1, label2, label3, label4, label5, label6], axis=1)
    e_labelled_set = np.concatenate([elabel1.reshape(-1, 1),
                                elabel2.reshape(-1, 1), 
                                elabel3.reshape(-1, 1), 
                                elabel4.reshape(-1, 1), 
                                elabel5.reshape(-1, 1),
                                elabel6.reshape(-1, 1)], axis=1)
    scalers = [scaler1, scaler2, scaler3, scaler4, scaler5, scaler6]
    target_set = np.concatenate([target_set, testset[:, -4].reshape(-1, 1)], axis=1)
    labels = ['teff', 'logg', 'fe_h', 'alpha', 'age', 'parallax']
    vlabelled_set = np.concatenate([vlabel1, vlabel2, vlabel3, vlabel4, vlabel5, vlabel6], axis=1)
    e_vlabelled_set = np.concatenate([velabel1.reshape(-1, 1), 
                                velabel2.reshape(-1, 1),
                                velabel3.reshape(-1, 1),
                                velabel4.reshape(-1, 1),
                                velabel5.reshape(-1, 1),
                                velabel6.reshape(-1, 1)], axis=1)

    if config['refit']:
        featurescaler.fit(trainset)
    trainset = featurescaler.transform(trainset)
    validset = featurescaler.transform(validset)
    testset = featurescaler.transform(testset)
    scale_factors = featurescaler.scale_  # This is the IQR used by RobustScaler for each feature
    etrainset = etrainset / scale_factors
    evalidset = evalidset / scale_factors
    etestset = etestset / scale_factors

    test_stuff = (testset, target_set, scalers, labels)

    for seed in seeds:

        random.seed(seed)
        torch.manual_seed(seed)
        
        continuous_cols = 140 # fixed for now
        model = modelfile.TabResnetSingleEncoder(
            continuous_cols=continuous_cols,
            blocks_dims=blocks_dims,
            output_cols=135,
            activ=cactivation,
            d_embedding=cd_embedding,
            norm=cnormalization,
            do=0.,
        )

        # model.load_state_dict(torch.load('/arc/home/aydanmckay/bprp_mae/.model_instances/checkpoint_60_fullmodelrealmags0605.pth'))

        # Initialize the pretraining wrapper
        pretrain_wrapper = TabResnetWrapper(
            model=model,
            datafile=None,
            scaler=featurescaler,
            xp_masking_ratio=0.9,
            m_masking_ratio=0.6,
            num_classes=5,
            latent_size=blocks_dims[-1],
            lr = clearning_rate,
            optimizer = coptimizer,
            wd=cweight_decay,
            lasso=classo,
            lf=closs_fn,
        )

        pretrain_wrapper.fit(trainset,etrainset,labelled_set,e_y_train=e_labelled_set,X_val=validset,eX_val=evalidset,
                            y_val=vlabelled_set,e_y_val=e_vlabelled_set,num_epochs=101,mini_batch=512,linearprobe=False, 
                            maskft=True,multitask=True,rncloss=False,ftlr=1e-4,ftopt='adamw',
                            ftact='relu',ftl2=0.0,ftlf='quantile',ftdim='test',
                            ftlabeldim=6,traintype='normal',pert_features=True,feature_seed=seed,
                            pert_labels=False,ensemblepath='quantilemodel_nodo_'+str(seed),
                            )

if __name__ == '__main__':
    main()