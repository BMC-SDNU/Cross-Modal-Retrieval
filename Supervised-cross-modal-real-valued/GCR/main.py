
import torch.optim as optim
from model import *
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label

######################################################################
# Start running

print(torch.version.cuda)
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':

    dataset = 'pascnn'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    DATA_DIR = 'data/' + dataset + '/'
    MAX_EPOCH = 80
    MAX_EPOCHGAN = 100
    batch_size = 100
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0.01

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)

    print('...Data loading is completed...')

    model_ft = CrossGCN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                        output_dim=input_data_par['num_class']).to(device)

    dis_ft = DiscriminatorV().to(device)
    params_to_update = list(model_ft.parameters())
    params_dis = list(dis_ft.parameters())

    gen_ft = GeneratorV().to(device)
    params_gen = list(gen_ft.parameters())

    #########
    disT_ft = DiscriminatorT().to(device)
    params_disT = list(disT_ft.parameters())

    genT_ft = GeneratorT().to(device)
    params_genT = list(genT_ft.parameters())


    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    optimizer_dis = optim.Adam(params_dis, lr=lr, betas=betas)
    optimizer_gen = optim.Adam(params_gen, lr=lr, betas=betas, weight_decay=weight_decay)
    optimizer_disT = optim.Adam(params_disT, lr=lr, betas=betas)
    optimizer_genT = optim.Adam(params_genT, lr=lr, betas=betas, weight_decay=weight_decay)

    print('...Training is beginning...')
    # Train and evaluate
    start = time.time()
    genT_ft, gen_ft, model_ft = train_model(genT_ft, disT_ft, gen_ft, dis_ft,model_ft,
                                            data_loader, optimizer_genT, optimizer_disT, optimizer_gen, optimizer_dis, optimizer, num_epochs=MAX_EPOCH,num_epochsGAN=MAX_EPOCHGAN)
    end = time.time()
    print('Total train time:', end-start)
    print('...Training is completed...')

    print('...Evaluation on testing data...')

    label = torch.argmax(torch.tensor(input_data_par['label_val']), dim=1)
    label_retrieval = torch.argmax(torch.tensor(input_data_par['label_retrieval']), dim=1)

    view1_feature, view2_feature, view1_predict, view2_predict = model_ft(torch.tensor(input_data_par['img_val']).to(device), torch.tensor(input_data_par['text_val']).to(device))

    view1_feature = view1_feature.detach().cpu()
    view2_feature = view2_feature.detach().cpu()
    view1_predict = view1_predict.detach().cpu()
    view2_predict = view2_predict.detach().cpu()

    view1_feature_retrieval, view2_feature_retrieval, view1_predict_retrieval, view2_predict_retrieval  = model_ft(
        torch.tensor(input_data_par['img_retrieval']).to(device), torch.tensor(input_data_par['text_retrieval']).to(device))

    view1_feature_retrieval = view1_feature_retrieval.detach().cpu()
    view2_feature_retrieval = view2_feature_retrieval.detach().cpu()
    view1_predict_retrieval = view1_predict_retrieval.detach().cpu()
    view2_predict_retrieval = view2_predict_retrieval.detach().cpu()

    img_to_txt = fx_calc_map_label(view1_feature, view2_feature_retrieval, label)
    print('Original...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature_retrieval, label)
    print('Original...Text to Image MAP = {}'.format(txt_to_img))

    print('Original...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))


    gen_img = gen_ft(torch.tensor(input_data_par['img_val']).to(device))


    gen_txt = genT_ft(torch.tensor(input_data_par['text_val']).to(device))


    gen_img = gen_img.detach().cpu()
    gen_txt = gen_txt.detach().cpu()

    gen_img_retrieval = gen_ft(torch.tensor(input_data_par['img_retrieval']).to(device))
    gen_img_retrieval = gen_img_retrieval.detach().cpu()

    gen_txt_retrieval = genT_ft(torch.tensor(input_data_par['text_retrieval']).to(device))
    gen_txt_retrieval = gen_txt_retrieval.detach().cpu()

    img_to_txt = fx_calc_map_label(gen_img, gen_txt_retrieval, label)
    print('GCR...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(gen_txt, gen_img_retrieval, label)
    print('GCR...Text to Image MAP = {}'.format(txt_to_img))

    print('GCR...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))




