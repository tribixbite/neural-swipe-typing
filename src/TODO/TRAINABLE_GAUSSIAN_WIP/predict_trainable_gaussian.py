# ! Если use_vocab_for_generation == True 
# многопоточность почему-то сильно медленнее, чем выполнение в главном потоке.
# Поэтому num_workers должно быть равно 0 (это запуск без многопоточности).
# Предполагаю, что замедление связано с переносом из одного потока в другой
# очень большого словаря, хранящегося в генераторе слов


# Сейчас предсказания отдельных моделей сохраняются как список списков
# кортежей (-log(prob), pred_word).  Пусть модель заточена под раскладку
# клавиатуры grid_name.  Пусть dataset[grid_name] – это датасет,
# сохраняющий порядок исходного датасета, но исключающий все экземпляры
# с другой раскладкой.  Пусть предсказание хранится в переменной preds.
# Тогда preds[i] - это список предсказаний для кривой dataset[grid_name][i].
# Данный список представлен кортежами (-log(prob), pred_word).


# На входе скрипта предсказания:
# * модели, от которых мы хотим получить предсказания. Модели имеют:
#   * название раскладки
#   * название архитктуры
#   * путь к весам
# * Алгоритм декодирования слова и его аргументы
# * датасет в виде JSON, для которого хотим получить предсказания (точнее,
#       пердсказания хотим получить для поднабора этого датасета,
#       с клавитурами конкретной раскладки)
#
#
# Гипотетически могут быть модели, умеющие работать сразу с
# множеством раскладок.  Предсказание для таких моделей делается
# точно также, отдельно для каждой раскладки.


# Результат модуля предсказаний будет подан на вход скрипту аггрегации,
# а также обучения аггрегации.
# Вход аггрегации в рамках одной раскладки должен быть следующим:
# * список с элементами (pred_id, pred)
# * состояние аггрегатора
# * название раскладки
# В качестве pred_id может выступать
# f"{weights_path}__{generator_type}__{generator_call_kwargs}__{grid_name}".
# Лучше это будет буквально id, сохраненный где-то
# в отдельном файле / базе данных.

KB_X_SCALER = lambda x: x/1080
KB_Y_SCALER = lambda x: x/667
GRID_NAME = "default"
DATA_ROOT = "./data/data_preprocessed"



from typing import Iterable, List, Tuple, Dict, Optional, Any
import os
import json
import pickle
import argparse
from dataclasses import dataclass, asdict


import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
# import pandas as pd

from model import MODEL_GETTERS_DICT
from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizerv1
from dataset import CurveDataset, CurveDatasetSubset
from word_generators_v2 import GENERATOR_CTORS_DICT
from feature_extractors import get_val_transform, weights_function_v1
from grid_processing_utils import get_grid


RawPredictionType = List[List[Tuple[float, str]]]


@dataclass
class Prediction:
    prediction: RawPredictionType
    model_name: str
    model_weights: str
    generator_name: str
    generator_call_kwargs: dict
    use_vocab_for_generation: bool
    grid_name: str
    dataset_split: str
    include_coords: bool
    include_time: bool
    include_velocities: bool
    include_accelerations: bool
    transform_name: str



def get_vocab(vocab_path: str) -> List[str]:
    with open(vocab_path, 'r', encoding = "utf-8") as f:
        return f.read().splitlines()


def get_n_coord_feats(include_coords: bool,
                      inculde_time: bool,
                      include_velocities: bool,
                      include_accelerations: bool,
                      ) -> int:
    return 2 * (include_coords + include_velocities + include_accelerations) + inculde_time


class Predictor:
    """
    Creates a prediction for a whole dataset.

    The resulting prediction contains metadata with information about
    model and decoding algorithm useful later for predictions aggregation.
    """

    

    def __init__(self,
                 model_architecture_name: str,
                 model_weights_path: str,
                 include_coords: bool,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 word_generator_type: str,
                 use_vocab_for_generation: bool,
                 n_classes: int,
                 generator_call_kwargs,
                 ) -> None:
        DEVICE = torch.device('cpu')

        self.word_generator_type = word_generator_type
        word_generator_ctor = GENERATOR_CTORS_DICT[word_generator_type]
        self.model_architecture_name = model_architecture_name
        self.model_weights_path = model_weights_path
        model_getter = MODEL_GETTERS_DICT[model_architecture_name]

        self.include_coords = include_coords
        self.include_time = include_time
        self.include_velocities = include_velocities
        self.include_accelerations = include_accelerations

        n_coord_feats = get_n_coord_feats(
            include_coords = include_coords,
            inculde_time = include_time,
            include_velocities = include_velocities,
            include_accelerations = include_accelerations
        )

        print(n_coord_feats)





        kb_tokenizer=KeyboardTokenizerv1()

        kb_centers_tensor, kb_centers_dict = get_kb_centers_tensor(
            grid_path = os.path.join(DATA_ROOT, "gridname_to_grid.json"),
            grid_name=GRID_NAME,
            kb_tokenizer=kb_tokenizer,
            kb_x_scaler = KB_X_SCALER,
            kb_y_scaler = KB_Y_SCALER
        )




        model = model_getter(DEVICE, model_weights_path, n_coord_feats=n_coord_feats, key_centers=kb_centers_tensor)
        self.word_char_tokenizer = CharLevelTokenizerv2(config['voc_path'])

        self.use_vocab_for_generation = use_vocab_for_generation

        word_generator_init_kwargs = {}
        if use_vocab_for_generation:
            word_generator_init_kwargs = {
                'vocab': get_vocab(config['voc_path']),
                'max_token_id': n_classes - 1
            }

        self.word_generator = word_generator_ctor(
            model, self.word_char_tokenizer, DEVICE, 
            **word_generator_init_kwargs)
        
        self.generator_call_kwargs = generator_call_kwargs

        

    def _predict_example(self,
                         data: Tuple[int, Tuple[Tensor, Tensor]]
                         ) -> Tuple[int, List[Tuple[float, str]]]:
        """
        Predicts a single example.

        Arguments:
        ----------
        data: Tuple[i, gen_in]
            i: int
                Index of the example in the dataset.
            gen_in: Unoin[Tensor, Tuple[Tensor, Tensor]]
                Tuple of (traj_feats, kb_tokens)

        Returns:
        --------
        i: int
            Index of the example in the dataset.
        pred: List[Tuple[log_probability, char_sequence]]
            log_probability: float
            char_sequence: str
        """

        i, gen_in = data
        pred = self.word_generator(gen_in, **self.generator_call_kwargs)
        return i, pred
    
    def _predict_raw_mp(self, dataset: CurveDataset,
                        num_workers: int) -> List[List[Tuple[float, str]]]:
        """
        Creates predictions given a word generator
        
        Arguments:
        ----------
        dataset: CurveDataset
            The dataset is supposed to be a subset of the original dataset
            containing only examples with the same grid_name as the predictor.
        num_workers: int
            Number of processes.

        Returns:
        --------
        preds: List[one_swipe_preds]
            one_swipe_preds: List[Tuple[log_probability, char_sequence]]
                log_probability: float
                char_sequence: str
        """
        preds = [None] * len(dataset)

        data = [(i, encoder_in)
                for i, ((encoder_in, _), _) in enumerate(dataset)]     
        
        if num_workers <= 0:

            for i, pred in tqdm(map(self._predict_example, data), total=len(dataset)):
                preds[i] = pred
            return preds
        
        with ProcessPoolExecutor(num_workers) as executor:
            for i, pred in tqdm(executor.map(self._predict_example, data), total=len(dataset)):
                preds[i] = pred

        return preds

    def predict(self, dataset: CurveDataset, 
                grid_name: str, dataset_split: str,
                transform_name: str, num_workers: int) -> Prediction:
        """
        Creates predictions given a word generator
        
        Arguments:
        ----------
        dataset: CurveDataset
            Output[i] is a list of predictions for dataset[i] curve.
        grid_name: str
        dataset_split: str
        num_workers: int
            Number of processes.
        """
        preds = self._predict_raw_mp(dataset, num_workers)

        preds_with_meta = Prediction(
            prediction=preds, 
            model_name=self.model_architecture_name,
            model_weights=self.model_weights_path, 
            generator_name=self.word_generator_type,
            generator_call_kwargs=self.generator_call_kwargs, 
            use_vocab_for_generation=self.use_vocab_for_generation,
            grid_name=grid_name, 
            dataset_split=dataset_split, 
            include_coords=self.include_coords,
            include_time=self.include_time,
            include_velocities=self.include_velocities,
            include_accelerations=self.include_accelerations,
            transform_name=transform_name)
        
        return preds_with_meta


# def create_new_df() -> pd.DataFrame:
#     df = pd.DataFrame(columns = [
#             'predictor_id', 'model_name', 'model_weights', 'generator_name', 'grid_name', 'dataset_split'])

# def load_df(preds_csv_path: str) -> pd.DataFrame:
#     if not os.path.exists(preds_csv_path):
#         print(f"Warning: {preds_csv_path} does not exist. Creating a new df...")
#         create_new_df()
    

def save_predictions(preds_wtih_meta: Prediction,
                     out_path: str,
                     preds_csv_path: str) -> None:
    with open(out_path, 'wb') as f:
        pickle.dump(
            preds_wtih_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

#     # df = load_df(preds_csv_path)
#     # update_database(df, preds_wtih_meta)
#     # df.to_csv(preds_csv_path, index=False)


def get_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    prediction_config = dict(full_config['prediction_config'])

    data_split = prediction_config['data_split']
    data_path = prediction_config['data_split__to__path'][data_split]
    prediction_config['data_path'] = data_path

    return prediction_config


def get_gridname_to_dataset(config) -> Dict[str, Dataset]:
    char_tokenizer = CharLevelTokenizerv2(config['voc_path'])

    transform = get_val_transform(
        gridname_to_grid_path=config['grid_name_to_grid__path'],
        grid_names=('default', 'extra'),
        transform_name=config['transform_name'],
        char_tokenizer=char_tokenizer,
        uniform_noise_range=0,
        include_time=config['include_time'],
        include_velocities=config['include_velocities'],
        include_accelerations=config['include_accelerations'],
        ds_paths_list=[config['data_path']],
        dist_weights_func=weights_function_v1,
        totals = [10_000],
        kb_x_scaler = KB_X_SCALER,
        kb_y_scaler = KB_Y_SCALER
    )

    print("Creating dataset...")
    dataset = CurveDataset(
        data_path=config['data_path'],
        store_gnames = True,
        init_transform=transform,
        get_item_transform=None,
        total = 10_000,
    )

    gridname_to_dataset = {
        'default': CurveDatasetSubset(dataset, grid_name='default'),
        'extra': CurveDatasetSubset(dataset, grid_name='extra'),
    }

    return gridname_to_dataset


def check_all_weights_exist(model_params: Iterable, models_root: str) -> None:
    for _, _, w_fname in model_params:
        if not os.path.exists(os.path.join(models_root, w_fname)):
            raise ValueError(f"Path {w_fname} does not exist.")
        

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--config', type=str)
    args = p.parse_args()
    return args 








from typing import Dict, Tuple
def get_key_center(hitbox: Dict[str, int]) -> Tuple[float, float]:
    x = hitbox['x'] + hitbox['w'] / 2
    y = hitbox['y'] + hitbox['h'] / 2
    return x, y
    

def get_id_to_kb_centers_list(kb_tokenizer, keys):
    kb_centers_dict = dict()
    for key in keys:
        if 'label' not in key:
            continue
        if key['label'] not in kb_tokenizer.t2i:
            continue
        key_id = kb_tokenizer.t2i[key['label']]
        kb_centers_dict[key_id] = get_key_center(key['hitbox'])
    
    for t, i in kb_tokenizer.t2i.items():
        if i not in kb_centers_dict:
            kb_centers_dict[i] = (-1, -1)

    return kb_centers_dict
    


def dict_to_sorted_list(d):
    return [v for k, v in sorted(d.items())]


from typing import Callable
def get_kb_centers_tensor(grid_path: str,
                          grid_name: str,
                          kb_tokenizer: KeyboardTokenizerv1,
                          kb_x_scaler: Callable, 
                          kb_y_scaler: Callable):
    grid = get_grid(grid_name, grid_path)
    keys = grid['keys']


    kb_tokenizer = KeyboardTokenizerv1()
    # legacy thing
    assert len(kb_tokenizer.t2i) == len(kb_tokenizer.i2t)
    if len(kb_tokenizer.t2i) != 37:
        kb_tokenizer.i2t.append('<extra>')
        kb_tokenizer.t2i['<extra>'] = len(kb_tokenizer.i2t) - 1
    assert len(kb_tokenizer.t2i) == 37


    kb_centers_dict = get_id_to_kb_centers_list(kb_tokenizer, keys)
    kb_centers_sorted_lst = dict_to_sorted_list(kb_centers_dict)
    kb_centers_tensor = torch.tensor(kb_centers_sorted_lst).float()
    mask = kb_centers_tensor == -1
    kb_centers_tensor[:, 0].apply_(kb_x_scaler)
    kb_centers_tensor[:, 1].apply_(kb_y_scaler)
    kb_centers_tensor = kb_centers_tensor.masked_fill(mask, -1)

    return kb_centers_tensor, kb_centers_dict










if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config)

    check_all_weights_exist(config['model_params'], config['models_root'])

    assert os.path.exists(config['out_path']), f"config['out_path'] doesn't exist ({config['out_path']})"

    gridname_to_dataset = get_gridname_to_dataset(config)

    for grid_name, model_getter_name, weights_f_name in config['model_params']:

        out_path = os.path.join(config['out_path'],
                                f"{weights_f_name.replace('/', '__')}.pkl")
        
        if os.path.exists(out_path):
            print(f"Path {out_path} exists. Skipping.")
            continue

        predictor = Predictor(
            model_getter_name,
            os.path.join(config['models_root'], weights_f_name),
            include_coords=config['include_coords'],
            include_time=config['include_time'],
            include_velocities=config['include_velocities'],
            include_accelerations=config['include_accelerations'],
            word_generator_type = config['generator'],
            use_vocab_for_generation = config['use_vocab_for_generation'],
            n_classes = config['n_classes'],
            generator_call_kwargs=config['generator_call_kwargs'],
        )

        preds_and_meta = predictor.predict(
            gridname_to_dataset[grid_name],
            grid_name, config['data_split'], 
            config['transform_name'], args.num_workers)

        save_predictions(preds_and_meta, out_path, config["csv_path"])
