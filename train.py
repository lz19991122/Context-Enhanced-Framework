import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from utils import save, load, train, test, data_to_device, data_concatenate
from datasets import MIMIC, IUXRAY
from losses import CELossTotal
from models import CNN, MVCNN, TNN, Classifier, Generator, Context

def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    return threshold[ix]

def infer(data_loader, model, device='cpu', threshold=None):
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)
            if threshold != None:
                output = model(image=source[0], history=source[3], threshold=threshold)
                # output = model(image=source[0], threshold=threshold)
                # output = model(image=source[0], history=source[3], label=source[2])
                # output = model(image=source[0], label=source[2])
            else:
                # output = model(source[0], source[1])
                output = model(source[0])

            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)

    return outputs, targets
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
torch.set_num_threads(1)
torch.manual_seed(seed=123)

RELOAD = False
PHASE = 'TRAIN'  # TRAIN or INFER
DATASET_NAME = 'IUXRAY'
BACKBONE_NAME = 'DenseNet121'
MODEL_NAME = 'Context'

if DATASET_NAME == 'MIMIC':
    EPOCHS = 50
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64
    MILESTONES = [25]
    
elif DATASET_NAME == 'IUXRAY':
    EPOCHS = 50
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64
    MILESTONES = [25]
    
else:
    raise ValueError('Invalid DATASET')

if __name__ == "__main__":
    if MODEL_NAME in ['Context']:
        SOURCES = ['image','caption','label','history']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','label','history']
        KW_TGT = None
        KW_OUT = None
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')

    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2
        
        dataset = MIMIC('./mimic/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=True, debug_mode=False, train_phase=(PHASE == 'TRAIN'))
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
            
    elif DATASET_NAME == 'IUXRAY':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2

        dataset = IUXRAY('./iu_xray/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
        
    else:
        raise ValueError('Invalid DATASET_NAME')

    if BACKBONE_NAME == 'DenseNet121':
        backbone = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)
        FC_FEATURES = 1024
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')

    if MODEL_NAME == 'Context':
        LR = 6e-5
        WD = 1e-2
        DROPOUT = 0.1
        NUM_EMBEDS = 256
        FWD_DIM = 256
        NUM_HEADS = 8
        NUM_LAYERS = 1
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        NUM_HEADS = 1
        NUM_LAYERS = 12
        
        cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
        gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)
        
        model = Context(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
        criterion = CELossTotal(ignore_index=3)
    else:
        raise ValueError('Invalid MODEL_NAME')

    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))
    
    last_epoch = -1
    best_metric = 1e9

    checkpoint_path_from = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    checkpoint_path_to = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    
    if RELOAD:
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))

    if PHASE == 'TRAIN':
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(last_epoch+1, EPOCHS):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, scaler=scaler)
            val_loss = test(val_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            test_loss = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            
            scheduler.step()
            
            if best_metric > val_loss:
                best_metric = val_loss
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print('New Best Metric: {}'.format(best_metric)) 
                print('Saved To:', checkpoint_path_to)

    elif PHASE == 'INFER':
        txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.15)
        gen_outputs = txt_test_outputs[0]
        gen_targets = txt_test_targets[0]

        out_file_ref = open('outputs/x_{}_{}_{}_{}_Ref.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_hyp = open('outputs/x_{}_{}_{}_{}_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_lbl = open('outputs/x_{}_{}_{}_{}_Lbl.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')

        for i in range(len(gen_outputs)):
            candidate = ''
            for j in range(len(gen_outputs[i])):
                tok = dataset.vocab.id_to_piece(int(gen_outputs[i,j]))
                if tok == '</s>':
                    break 
                elif tok == '<s>':
                    continue
                elif tok == '▁':
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' '
                elif tok in [',', '.', '-', ':']:
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' ' + tok + ' '
                    else:
                        candidate += tok + ' '
                else:
                    candidate += tok
            out_file_hyp.write(candidate + '\n')

            reference = ''
            for j in range(len(gen_targets[i])):
                tok = dataset.vocab.id_to_piece(int(gen_targets[i,j]))
                if tok == '</s>':
                    break
                elif tok == '<s>':
                    continue
                elif tok == '▁':
                    if len(reference) and reference[-1] != ' ':
                        reference += ' '
                elif tok in [',', '.', '-', ':']:
                    if len(reference) and reference[-1] != ' ':
                        reference += ' ' + tok + ' '
                    else:
                        reference += tok + ' '
                else: # letter
                    reference += tok
            out_file_ref.write(reference + '\n')

        for i in tqdm(range(len(test_data))):
            target = test_data[i][1]
            out_file_lbl.write(' '.join(map(str,target[1])) + '\n')

    else:
        raise ValueError('Invalid PHASE')
