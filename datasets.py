import os
import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import sentencepiece as spm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MIMIC(data.Dataset):
    def __init__(self, directory, input_size=(256,256), random_transform=True,
                view_pos=['AP', 'PA', 'LATERAL'], max_views=2, sources=['image','history'], targets=['label'], 
                max_len=1000, vocab_file='mimic_unigram_1000.model'):

        self.source_sections = ['INDICATION:', 'HISTORY:', 'CLINICAL HISTORY:', 'REASON FOR EXAM:', 'REASON FOR EXAMINATION:', 'CLINICAL INFORMATION:', 'CLINICAL INDICATION:', 'PATIENT HISTORY:']
        self.target_sections = ['FINDINGS:']
        self.vocab = spm.SentencePieceProcessor(model_file=directory + vocab_file)
        self.vocab_file = vocab_file 

        self.sources = sources 
        self.targets = targets
        self.max_views = max_views
        self.view_pos = view_pos
        self.max_len = max_len
        
        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.__input_data(binary_mode=True)
        
        if random_transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1), 
                    transforms.RandomRotation(15, expand=True)]),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
            
    def __len__(self):
        return len(self.idx_pidsid)

    def __getitem__(self, idx):
        idx = self.idx_pidsid[idx]

        sources = []
        targets = []

        if 'image' in self.sources:
            imgs, vpos = [], []
            new_orders = np.random.permutation(len(self.img_files[idx]))
            img_files = np.array(self.img_files[idx])[new_orders].tolist()
            for i in range(min(self.max_views,len(img_files))):
                img_file = self.dir + 'images/' + idx[0] + '/' + idx[1] + '/' + img_files[i]
                pos = self.img_positions[img_files[i][:-4]]
                img = Image.open(img_file).convert('RGB')
                imgs.append(self.transform(img).unsqueeze(0))
                vpos.append(self.dict_positions[pos])
            cur_len = len(vpos)
            for i in range(cur_len, self.max_views):
                imgs.append(torch.zeros_like(imgs[0]))
                vpos.append(-1) 
            
            imgs = torch.cat(imgs, dim=0) 
            vpos = np.array(vpos, dtype=np.int64) 
        info = self.img_captions[idx]
        
        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = ' '.join(source_info)
        
        encoded_source_info = [self.vocab.bos_id()] + self.vocab.encode(source_info) + [self.vocab.eos_id()]
        source_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        source_info[:min(len(encoded_source_info), self.max_len)] = encoded_source_info[:min(len(encoded_source_info), self.max_len)]

        target_info = []
        for section, content in info.items():
            if section in self.target_sections:
                target_info.append(content)
        target_info = ' '.join(target_info)
        
        # Compute extra labels (noun phrases)
        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in target_info:
                np_labels[i] = 1
                
        encoded_target_info = [self.vocab.bos_id()] + self.vocab.encode(target_info) + [self.vocab.eos_id()]
        target_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        target_info[:min(len(encoded_target_info), self.max_len)] = encoded_target_info[:min(len(encoded_target_info), self.max_len)]

        for i in range(len(self.sources)):
            if self.sources[i] == 'image':
                sources.append((imgs,vpos))
            if self.sources[i] == 'history':
                sources.append(source_info)
            if self.sources[i] == 'label':
                sources.append(np.concatenate([self.img_labels[idx], np_labels]))
            if self.sources[i] == 'caption':
                sources.append(target_info)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_target_info), self.max_len))
                
        for i in range(len(self.targets)):
            if self.targets[i] == 'label':
                targets.append(np.concatenate([self.img_labels[idx], np_labels]))
            if self.targets[i] == 'caption':
                targets.append(target_info)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_target_info), self.max_len))
                
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]

    def __get_reports_images(self, file_name='reports.json'):
        caption_file = json.load(open(self.dir + file_name, 'r'))
        img_captions = {}
        img_files = {}
        for file_name, report in caption_file.items():
            k = file_name[-23:-4]
            pid,sid = k.split('/')
            try:
                # List all available images in each folder
                file_list = os.listdir(self.dir + 'images/' + pid + '/' + sid)
                # Select only images in self.view_pos
                file_list = [f for f in file_list if self.img_positions[f[:-4]] in self.view_pos]
                # Make sure there is atleast one image in each folder, and a non-empty findings section in each report
                if len(file_list) and ('FINDINGS:' in report) and (report['FINDINGS:'] != ''): 
                    img_files[(pid,sid)] = file_list
                    img_captions[(pid,sid)] = report
            except Exception as e:
                pass
        return img_captions, img_files

    def __get_view_positions(self, file_name='mimic-cxr-2.0.0-metadata.csv'):
        txt_file = self.dir + file_name
        data = pd.read_csv(txt_file, dtype=object)
        data = data.to_numpy().astype(str)
        return dict(zip(data[:,0].tolist(), data[:,4].tolist())), np.unique(data[:,4]).tolist()

    def __get_labels(self, binary_mode, file_name='mimic-cxr-2.0.0-chexpert.csv'):
        txt_file = self.dir + 'mimic-cxr-2.0.0-chexpert.csv'
        data = pd.read_csv(txt_file, dtype=object)

        label_names = list(data.columns.values[2:])
        data = data.to_numpy().astype(str)
        if binary_mode:
            data[data == '-1.0'] = "1"
            data[data ==  'nan'] = "0"
        else:
            data[data == '-1.0'] = "2"
            data[data ==  'nan'] = "3"
        
        img_labels = {}
        for i in range(len(data)):
            pid = 'p' + data[i,0].item()
            sid = 's' + data[i,1].item()
            labels = data[i,2:].astype(float)
            img_labels[(pid,sid)] = labels
        return img_labels, label_names

    def __get_nounphrase(self, top_k=100, file_name='count_nounphrase.json'):
        count_np = json.load(open(self.dir + file_name, 'r'))
        sorted_count_np = sorted([(k,v) for k,v in count_np.items()], key=lambda x: x[1], reverse=True)
        top_nounphrases = [k for k,v in sorted_count_np][:top_k]
        return top_nounphrases
           
    def __input_data(self, binary_mode=True):
        self.img_positions, self.list_positions = self.__get_view_positions()
        self.dict_positions = dict(zip(self.list_positions, range(len(self.list_positions))))
        self.img_captions, self.img_files = self.__get_reports_images()
        self.img_labels, self.list_diseases = self.__get_labels(binary_mode)
        self.dict_diseases = dict(zip(self.list_diseases, range(len(self.list_diseases))))
        self.idx_pidsid = list(self.img_captions.keys())
        self.top_np = self.__get_nounphrase()
        
    def __generate_splits(self, test_size=0.2, seed=0, file_name='mimic-cxr-2.0.0-chexpert.csv'):
        train_val_file = open(self.dir + 'train_val_list.txt', 'w')
        test_file = open(self.dir + 'test_list.txt', 'w')

        txt_file = self.dir + 'mimic-cxr-2.0.0-chexpert.csv'
        data = pd.read_csv(txt_file, dtype=object)
        data = data.to_numpy().astype(str)

        pid_sid = {}
        for i in range(len(data)):
            pid = data[i,0].item()
            sid = data[i,1].item() 
            
            if pid in pid_sid:
                pid_sid[pid].append(sid)
            else:
                pid_sid[pid] = [sid]

        np.random.seed(seed)
        unique_pid = np.unique(data[:,0])        
        random_pid = np.random.permutation(unique_pid)

        pvt = int((1-test_size) * len(unique_pid))
        train_pid = random_pid[:pvt]
        test_pid = random_pid[pvt:]

        for pid in train_pid:
            for sid in pid_sid[pid]:
                if ('p'+pid,'s'+sid) in self.img_captions:
                    train_val_file.write('p' + pid + '/' + 's' + sid + '\n')
        
        for pid in test_pid:
            for sid in pid_sid[pid]:
                if ('p'+pid,'s'+sid) in self.img_captions:
                    test_file.write('p' + pid + '/' + 's' + sid + '\n')

    def get_subsets(self, pvt=0.9, seed=0, generate_splits=True, debug_mode=False, train_phase=True):
        if generate_splits:
            self.__generate_splits(seed=0)
            print('New splits generated')
            
        train_files = np.loadtxt(self.dir + 'train_val_list.txt', dtype=str)
        test_files = np.loadtxt(self.dir + 'test_list.txt', dtype=str)

        train_files = np.array([f.split('/') for f in train_files])
        test_files = np.array([f.split('/') for f in test_files])
        
        np.random.seed(seed)
        indices = np.random.permutation(len(train_files))
        pivot = int(len(train_files) * pvt)
        train_indices = indices[:pivot]
        val_indices = indices[pivot:]

        train_dataset = MIMIC(self.dir, self.input_size, self.random_transform, 
                              self.view_pos, self.max_views, self.sources, self.targets, 
                              self.max_len, self.vocab_file)
        train_dataset.idx_pidsid = [(pid,sid) for pid,sid in train_files[train_indices]] if not debug_mode else [(pid,sid) for pid,sid in train_files[train_indices]][:10000]
        
        val_dataset = MIMIC(self.dir, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, 
                            self.max_len, self.vocab_file)
        val_dataset.idx_pidsid = [(pid,sid) for pid,sid in train_files[val_indices]] if not debug_mode else [(pid,sid) for pid,sid in train_files[val_indices]][:1000]

        test_dataset = MIMIC(self.dir, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, 
                            self.max_len, self.vocab_file)
        test_dataset.idx_pidsid = [(pid,sid) for pid,sid in test_files] if not debug_mode else [(pid,sid) for pid,sid in test_files][:1000]

        subset_size = 100000
        
        val_idx = np.random.choice(len(val_dataset.idx_pidsid), size=min(subset_size, len(val_dataset.idx_pidsid)), replace=False)
        test_idx = np.random.choice(len(test_dataset.idx_pidsid), size=min(subset_size, len(test_dataset.idx_pidsid)), replace=False)
        
        train_dataset.idx_pidsid = train_dataset.idx_pidsid[:]
        val_dataset.idx_pidsid = [val_dataset.idx_pidsid[i] for i in val_idx]
        test_dataset.idx_pidsid = [test_dataset.idx_pidsid[i] for i in test_idx]
        
        return train_dataset, val_dataset, test_dataset

class IUXRAY(data.Dataset):
    def __init__(self, directory, input_size=(256,256), random_transform=True,
                view_pos=['AP', 'PA', 'LATERAL'], max_views=2, sources=['image','history'], targets=['label'], 
                max_len=1000, vocab_file='nlmcxr_unigram_1000.model'):
        
        self.source_sections = ['INDICATION', 'COMPARISON']
        self.target_sections = ['FINDINGS']
        self.vocab = spm.SentencePieceProcessor(model_file=directory + vocab_file)
        self.vocab_file = vocab_file

        self.sources = sources
        self.targets = targets
        self.max_views = max_views
        self.view_pos = view_pos
        self.max_len = max_len

        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.__input_data(binary_mode=True)
        
        if random_transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1), 
                    transforms.RandomRotation(15, expand=True)]),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        sources, targets = [], []
        tmp_rep = self.captions[self.file_report[file_name]['image'][0] + '.png']
        if 'image' in self.sources:
            imgs, vpos = [], []
            images = self.file_report[file_name]['image']
            new_orders = np.random.permutation(len(images))
            img_files = np.array(images)[new_orders].tolist()

            for i in range(min(self.max_views,len(img_files))):
                img_file = self.dir + 'images/' + img_files[i] + '.png'
                img = Image.open(img_file).convert('RGB')
                imgs.append(self.transform(img).unsqueeze(0))
                vpos.append(1)
            cur_len = len(vpos)
            for i in range(cur_len, self.max_views):
                imgs.append(torch.zeros_like(imgs[0]))
                vpos.append(-1) 
            
            imgs = torch.cat(imgs, dim=0)
            vpos = np.array(vpos, dtype=np.int64) 
        info = self.file_report[file_name]['report']
        
        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = ' '.join(source_info)
        
        encoded_source_info = [self.vocab.bos_id()] + self.vocab.encode(source_info) + [self.vocab.eos_id()]
        source_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        source_info[:min(len(encoded_source_info), self.max_len)] = encoded_source_info[:min(len(encoded_source_info), self.max_len)]

        target_info = []
        for section, content in info.items():
            if section in self.target_sections:
                target_info.append(content)
        target_info = tmp_rep
        
        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in target_info:
                np_labels[i] = 1
        
        encoded_target_info = [self.vocab.bos_id()] + self.vocab.encode(target_info) + [self.vocab.eos_id()]
        target_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        target_info[:min(len(encoded_target_info), self.max_len)] = encoded_target_info[:min(len(encoded_target_info), self.max_len)]

        for i in range(len(self.sources)):
            if self.sources[i] == 'image':
                sources.append((imgs,vpos))
            if self.sources[i] == 'history':
                sources.append(source_info)
            if self.sources[i] == 'label':
                sources.append(np.concatenate([np.array(self.file_labels[file_name]), np_labels]))
            if self.sources[i] == 'caption':
                sources.append(target_info)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_target_info), self.max_len))
                
        for i in range(len(self.targets)):
            if self.targets[i] == 'label':
                targets.append(np.concatenate([np.array(self.file_labels[file_name]), np_labels]))
            if self.targets[i] == 'caption':
                targets.append(target_info)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_target_info), self.max_len))
                
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]

    def __get_nounphrase(self, top_k=100, file_name='count_nounphrase.json'):
        count_np = json.load(open(self.dir + file_name, 'r'))
        sorted_count_np = sorted([(k,v) for k,v in count_np.items()], key=lambda x: x[1], reverse=True)
        top_nounphrases = [k for k,v in sorted_count_np][:top_k]
        return top_nounphrases

    def __input_data(self, binary_mode=True):
        self.__input_caption()
        self.__input_report()
        self.__input_label()
        self.__filter_inputs()
        self.top_np = self.__get_nounphrase()
        
    def __input_label(self):
        with open(self.dir + 'file2label.json') as f:
            labels = json.load(f)
        self.file_labels = labels
        
    def __input_caption(self):
        with open(self.dir + 'captions.json') as f:
            captions = json.load(f)
        self.captions = captions
        
    def __input_report(self):
        with open(self.dir + 'reports_ori.json') as f:
            reports = json.load(f)
        self.file_list = [k for k in reports.keys()]
        self.file_report = reports

    def __filter_inputs(self):
        filtered_file_report = {}
        for k, v in self.file_report.items():
            if (len(v['image']) > 0) and (('FINDINGS' in v['report']) and (v['report']['FINDINGS'] != '')): # or (('IMPRESSION' in v['report']) and (v['report']['IMPRESSION'] != ''))):
                filtered_file_report[k] = v
        self.file_report = filtered_file_report
        self.file_list = [k for k in self.file_report.keys()]

    def get_subsets(self, train_size=0.7, val_size=0.1, test_size=0.2, seed=0):
        np.random.seed(seed)
        indices = np.random.permutation(len(self.file_list))
        train_pvt = int(train_size * len(self.file_list))
        val_pvt = int((train_size + val_size) * len(self.file_list))
        train_indices = indices[:train_pvt]
        val_indices = indices[train_pvt:val_pvt]
        test_indices = indices[val_pvt:]

        master_file_list = np.array(self.file_list)

        train_dataset = IUXRAY(self.dir, self.input_size, self.random_transform,
                              self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        train_dataset.file_list = master_file_list[train_indices].tolist()

        val_dataset = IUXRAY(self.dir, self.input_size, False,
                            self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        val_dataset.file_list = master_file_list[val_indices].tolist()

        test_dataset = IUXRAY(self.dir, self.input_size, False,
                             self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        test_dataset.file_list = master_file_list[test_indices].tolist()

        return train_dataset, val_dataset, test_dataset
    
