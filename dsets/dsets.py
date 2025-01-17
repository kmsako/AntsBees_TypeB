
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import glob
from collections import namedtuple
import os


CandidateInfoTuple = namedtuple('CandidateInfoTuple', ['label','path'])


def make_path_list(root_dir,phase='train',):
    '''
    データのファイルパスを格納したリストを作成する。

    Parameters:
      phase(str): 'train'または'val'

    Returns:
      path_list(list): 画像データのパスを格納したリスト
    '''
    # 画像ファイルのルートディレクトリ
    # 画像ファイルパスのフォーマットを作成
    # rootpath +
    #   train/ants/*.jpg
    #   train/bees/*.jpg
    #   val/ants/*.jpg
    #   val/bees/*.jpg
    tt= phase +'/**/*.jpg'
    target_path = os.path.join(root_dir,tt)
    # ファイルパスを格納するリスト
    candidateInfo_list = []  # ここに格納する

    # glob()でファイルパスを取得してリストに追加
    for path in glob.glob(target_path):
        p=Path(path)
        label=p.parts[-2]
        if label == 'ants':
          label = 1 # Antsは,1
        else:
           label =0 # Beesは,0

        candidateInfo_list.append(
            CandidateInfoTuple(
                label,
                path
            )           
        )

    return candidateInfo_list



# ファイルパスのリストを生成
#train_list = make_datapath_list(phase='train')
#val_list = make_datapath_list(phase='val')



class MakeDataset(Dataset):
    '''
    アリとハチの画像のDatasetクラス
    PyTorchのDatasetクラスを継承

    Attributes:
      file_list(list): 画像のパスを格納したリスト
      transform(object): 前処理クラスのインスタンス
      phase(str): 'train'または'val'
    Returns:
      img_transformed: 前処理後の画像データ
      label(int): 正解ラベル
    '''
    def __init__(self, file_list, transform=None, phase='train'):
        '''インスタンス変数の初期化
        '''
        self.file_list = file_list[phase]  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase          # 'train'または'val'

    def __len__(self):
        '''len(obj)で実行されたときにコールされる関数
        画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''Datasetクラスの__getitem__()をオーバーライド
           obj[i]のようにインデックスで指定されたときにコールバックされる

           Parameters:
             index(int): データのインデックス
           Returns:

          前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # ファイルパスのリストからindex番目の画像をロード
        img_path = self.file_list[index]
        # ファイルを開く -> (高さ, 幅, RGB)
        img = Image.open(img_path)

        # 画像を前処理  -> torch.Size([3, 224, 224])
        img_transformed = self.transform(
            img, self.phase)
        
        p=Path(img_path)
        label=p.parts[-2]
        # 正解ラベルをファイル名から切り出す
        #if self.phase == 'train':
            # 訓練データはファイルパスの31文字から34文字が'ants'または'bees'
        #    p=Path(img_path)
        #    label=p.parts[-2]
        #elif self.phase == 'val':
        #    # 検証データはファイルパスの29文字から32文字が'ants'または'bees'
        #    label=p.parts[-2]

        # 正解ラベルの文字列を数値に変更する
        if label == 'ants':
            label = 1 # アリは0
        elif label == 'bees':
            label = 0 # ハチは1

        return img_transformed, label
        

import random 

class MakeBalancedDataset(Dataset):
    '''
    アリとハチの画像のDatasetクラス
    PyTorchのDatasetクラスを継承

    Attributes:
      file_list(list): 画像のパスを格納したリスト
      transform(object): 前処理クラスのインスタンス
      phase(str): 'train'または'val'
    Returns:
      img_transformed: 前処理後の画像データ
      label(int): 正解ラベル
    '''
    def __init__(self, file_list, ratio_int=True, transform=None, phase='train',records=300):
        '''インスタンス変数の初期化
        '''
        self.file_list = file_list[phase]  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase          # 'train'または'val'
        self.ratio_int= ratio_int
        self.records = records

        random.shuffle(self.file_list)
        
        self.ants_list = [nt for nt in self.file_list if not nt.label]        
        self.bees_list = [nt for nt in self.file_list if nt.label]

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.ants_list)
            random.shuffle(self.bees_list)

    def __len__(self):
        if self.ratio_int:
            return self.records
        else:
            return len(self.file_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)
            #奇数のとき
            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.ants_list)
                candidateInfo_tup = self.ants_list[neg_ndx]
            #偶数のとき
            else:
                pos_ndx %= len(self.bees_list)
                candidateInfo_tup = self.bees_list[pos_ndx]
        else:
            candidateInfo_tup = self.file_list[ndx]

        # ファイルを開く -> (高さ, 幅, RGB)
        img = Image.open(candidateInfo_tup.path)

        # 画像を前処理  -> torch.Size([3, 224, 224])
        img_transformed = self.transform(
            img, self.phase)        

        return img_transformed, candidateInfo_tup.label