
"""
Blurinator 9000 Ultra – улучшенная версия с раздельными постоянными времени пикселя,
моделью зрения (размытие) и корректной обработкой краёв.
Добавлена поддержка таблиц времени отклика монитора.
"""

import matplotlib
import sys
import time
import copy,threading, itertools
matplotlib.use('TkAgg')  # обязательно до импорта pyplot для интерактивных окон
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import torch
import cv2
import numpy as np
import os
from scipy.optimize import curve_fit
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
from scipy.ndimage import gaussian_filter1d, find_objects
from scipy.signal import savgol_filter, find_peaks
# В начало файла, после остальных импортов
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment,_clip_iqa_get_anchor_vectors
from transformers.modeling_outputs import BaseModelOutputWithPooling
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("SAM2 не установлен. Установите: pip install segment-anything-2")




# ----------------------------------------------------------------------
# Класс для работы с таблицами времени отклика
# ----------------------------------------------------------------------
class ResponseTimeTable:
    """
    Хранит и интерполирует таблицы времени отклика.
    Поддерживает неполные матрицы и различные методы интерполяции.
    """

    def __init__(self):
        self.rise_matrix = None  # полная матрица 256x256 (после интерполяции)
        self.fall_matrix = None
        self.loaded = False
        self.interp_method = 'linear'  # 'linear', 'cubic', 'rbf', 'overdrive', 'none'

    def load_from_csv(self, rise_path, fall_path, interp_method='linear', **kwargs):
        """
        Загружает матрицы из CSV, интерполирует до 256x256.

        Параметры:
            rise_path, fall_path : str – пути к файлам CSV.
            interp_method : str – метод интерполяции:
                'linear' / 'cubic' – RegularGridInterpolator,
                'rbf' – Radial Basis Function (thin_plate_spline),
                'overdrive' – физическая модель overdrive (подгонка параметров),
                'none' – без интерполяции (ближайший сосед).
            **kwargs – дополнительные параметры (например, smoothing для RBF).
        """

        try:
            rise_df = pd.read_csv(rise_path, header=None, index_col=None)
            fall_df = pd.read_csv(fall_path, header=None, index_col=None)
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV: {e}")

        # Автоопределение заголовков
        def is_numeric_column(col):
            return pd.to_numeric(col, errors='coerce').notna().all()

        if not is_numeric_column(rise_df.iloc[:, 0]):
            rise_df = rise_df.set_index(rise_df.columns[0])
            fall_df = fall_df.set_index(fall_df.columns[0])
        if not is_numeric_column(rise_df.iloc[0, :]):
            rise_df.columns = rise_df.iloc[0, :]
            fall_df.columns = fall_df.iloc[0, :]
            rise_df = rise_df.iloc[1:, :]
            fall_df = fall_df.iloc[1:, :]

        try:
            rise_df.index = rise_df.index.astype(float)
            fall_df.index = fall_df.index.astype(float)
            rise_df.columns = rise_df.columns.astype(float)
            fall_df.columns = fall_df.columns.astype(float)
        except:
            pass

        old_levels = rise_df.index.values.astype(float)
        new_levels = rise_df.columns.values.astype(float)

        rise_vals = rise_df.values.astype(float) / 1000.0  # мс -> с
        fall_vals = fall_df.values.astype(float) / 1000.0

        self.interp_method = interp_method

        if interp_method == 'overdrive':
            self.rise_matrix = self._fit_overdrive_model(old_levels, new_levels, rise_vals)
            self.fall_matrix = self._fit_overdrive_model(old_levels, new_levels, fall_vals)
        elif interp_method == 'rbf':
            smoothing = kwargs.get('smoothing', 0.0)
            self.rise_matrix = self._interpolate_rbf(old_levels, new_levels, rise_vals, smoothing)
            self.fall_matrix = self._interpolate_rbf(old_levels, new_levels, fall_vals, smoothing)
        elif interp_method in ('linear', 'cubic'):
            self.rise_matrix = self._interpolate_grid(old_levels, new_levels, rise_vals, method=interp_method)
            self.fall_matrix = self._interpolate_grid(old_levels, new_levels, fall_vals, method=interp_method)
        else:  # 'none'
            self.rise_matrix = self._resize_nearest(rise_vals, old_levels, new_levels)
            self.fall_matrix = self._resize_nearest(fall_vals, old_levels, new_levels)

        self.rise_matrix = np.maximum(self.rise_matrix, 0.0)
        self.fall_matrix = np.maximum(self.fall_matrix, 0.0)
        self.loaded = True

    # ------------------------------------------------------------------
    # Метод 2: модель Overdrive (физическая подгонка)
    # ------------------------------------------------------------------
    def _fit_overdrive_model(self, old_levels, new_levels, measured_matrix):
        """
        Подгонка параметров физической модели времени отклика.
        Модель: τ(old, new) = τ0 + A * |Δ|^p + B * (old/255)^q * (1 - new/255)^r
        Использует нормализацию и барьеры для устойчивости.
        """
        eps = 1e-6
        old_grid, new_grid = np.meshgrid(old_levels, new_levels, indexing='ij')
        points = np.stack([old_grid.ravel(), new_grid.ravel()], axis=-1)
        values = measured_matrix.ravel()

        # Убираем NaN и отрицательные значения
        mask = ~np.isnan(values) & (values > 0)
        pts = points[mask] / 255.0  # нормализуем к [0,1]
        vals = values[mask]

        # Нормализуем значения для лучшей сходимости
        scale = np.median(vals)
        vals_norm = vals / scale

        def model_func(params, x):
            tau0, A, p, B, q, r = params
            old_norm = x[:, 0]
            new_norm = x[:, 1]
            delta = np.abs(old_norm - new_norm) * 255.0  # масштаб обратно в [0,255]
            # Добавляем эпсилон, чтобы избежать 0^отрицательное
            term1 = A * (delta ** p)
            term2 = B * ((old_norm + eps) ** q) * ((1.0 - new_norm + eps) ** r)
            return tau0 + term1 + term2

        def loss(params):
            pred = model_func(params, pts)
            # Штраф за выход предсказаний за разумные границы
            penalty = np.sum(np.maximum(0, -pred)) + np.sum(np.maximum(0, pred - 1.0))
            mse = np.mean((pred - vals_norm) ** 2)
            return mse + 0.1 * penalty

        # Начальное приближение в нормализованных единицах
        p0 = [0.001 / scale, 0.001 / scale, 1.0, 0.001 / scale, 1.0, 1.0]
        bounds = [
            (0, None),  # τ0
            (0, None),  # A
            (0.5, 3.0),  # p
            (0, None),  # B
            (0.5, 3.0),  # q
            (0.5, 3.0)  # r
        ]

        res = minimize(loss, p0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12})
        best_params = res.x * np.array([scale, scale, 1.0, scale, 1.0, 1.0])
        best_params[0] = max(best_params[0], 0)  # τ0 неотрицательное
        print(f"Overdrive model fitted (scale={scale:.4f}): "
              f"τ0={best_params[0]:.4f}, A={best_params[1]:.4f}, p={best_params[2]:.2f}, "
              f"B={best_params[3]:.4f}, q={best_params[4]:.2f}, r={best_params[5]:.2f}")

        # Генерация полной матрицы 256x256
        full_old, full_new = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
        full_points = np.stack([full_old.ravel(), full_new.ravel()], axis=-1) / 255.0
        full_values = model_func(best_params, full_points).reshape(256, 256)
        return full_values

    # ------------------------------------------------------------------
    # Интерполяция RBF (опционально)
    # ------------------------------------------------------------------
    def _interpolate_rbf(self, old_levels, new_levels, measured_matrix, smoothing=0.0):
        old_grid, new_grid = np.meshgrid(old_levels, new_levels, indexing='ij')
        points = np.stack([old_grid.ravel(), new_grid.ravel()], axis=-1)
        values = measured_matrix.ravel()

        mask = ~np.isnan(values)
        pts = points[mask]
        vals = values[mask]

        # Вычитаем диагональ для улучшения точности
        diag_vals = np.array(
            [measured_matrix[i, i] for i in range(len(old_levels)) if not np.isnan(measured_matrix[i, i])])
        if len(diag_vals) > 0:
            diag_interp = np.interp(old_levels, old_levels[~np.isnan(np.diag(measured_matrix))], diag_vals)
            vals_no_diag = vals.copy()
            for i, (x, y) in enumerate(pts):
                if x == y:
                    vals_no_diag[i] = 0.0
                else:
                    idx_x = np.argmin(np.abs(old_levels - x))
                    idx_y = np.argmin(np.abs(new_levels - y))
                    vals_no_diag[i] -= (diag_interp[idx_x] + diag_interp[idx_y]) / 2
        else:
            vals_no_diag = vals

        rbf = RBFInterpolator(pts, vals_no_diag, kernel='thin_plate_spline', smoothing=smoothing)

        full_old, full_new = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
        full_pts = np.stack([full_old.ravel(), full_new.ravel()], axis=-1)
        interp_no_diag = rbf(full_pts).reshape(256, 256)

        # Восстанавливаем диагональ
        if len(diag_vals) > 0:
            full_diag = np.interp(np.arange(256), old_levels, diag_interp)
            for i in range(256):
                for j in range(256):
                    interp_no_diag[i, j] += (full_diag[i] + full_diag[j]) / 2

        return interp_no_diag

    def plot_matrices(self, title="Response Time Matrices"):
        """
        Отображает матрицы rise и fall в виде цветовых карт.
        """
        if not self.loaded:
            raise ValueError("Таблицы не загружены")

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(self.rise_matrix.T, origin='lower', cmap='viridis',
                             extent=[0, 255, 0, 255], aspect='auto')
        axes[0].set_title("Rise time (old → new ≥ old)")
        axes[0].set_xlabel("New level")
        axes[0].set_ylabel("Old level")
        plt.colorbar(im1, ax=axes[0], label='Time (s)')

        im2 = axes[1].imshow(self.fall_matrix.T, origin='lower', cmap='viridis',
                             extent=[0, 255, 0, 255], aspect='auto')
        axes[1].set_title("Fall time (old → new < old)")
        axes[1].set_xlabel("New level")
        axes[1].set_ylabel("Old level")
        plt.colorbar(im2, ax=axes[1], label='Time (s)')

        fig.suptitle(title)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Интерполяция на регулярной сетке (linear/cubic)
    # ------------------------------------------------------------------
    def _interpolate_grid(self, old_levels, new_levels, measured_matrix, method='linear'):
        points = (old_levels, new_levels)
        interp = RegularGridInterpolator(points, measured_matrix, method=method,
                                         bounds_error=False, fill_value=None)

        full_old, full_new = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
        full_pts = np.stack([full_old.ravel(), full_new.ravel()], axis=-1)
        return interp(full_pts).reshape(256, 256)

    # ------------------------------------------------------------------
    # Без интерполяции (ближайший сосед)
    # ------------------------------------------------------------------
    def _resize_nearest(self, matrix, old_levels, new_levels):
        full = np.zeros((256, 256))
        old_idx = np.round(np.linspace(0, 255, len(old_levels))).astype(int)
        new_idx = np.round(np.linspace(0, 255, len(new_levels))).astype(int)
        for i, oi in enumerate(old_idx):
            for j, nj in enumerate(new_idx):
                full[oi, nj] = matrix[i, j]
        # Растягиваем повторением
        full = np.repeat(np.repeat(matrix, 256 // len(old_levels), axis=0),
                         256 // len(new_levels), axis=1)
        return full[:256, :256]

    # ------------------------------------------------------------------
    # Получить время перехода
    # ------------------------------------------------------------------
    def get_transition_time(self, old_level, new_level):
        if not self.loaded:
            return None
        old = int(round(np.clip(old_level, 0, 255)))
        new = int(round(np.clip(new_level, 0, 255)))
        if new >= old:
            return self.rise_matrix[old, new]
        else:
            return self.fall_matrix[old, new]

def _to_float01(img):
    img = img.astype(np.float32)
    if img.max() > 1.5:  # похоже на 0..255
        img /= 255.0
    return img



class BlurTester:
    """
    Автоматизированное тестирование: оценка резкости (blur) и двоения (crosstalk)
    с помощью CLIP‑IQA (torchmetrics).
    """

    # ---------- Текстовые промпты (храним для ясности) ----------
    PROMPT_BLUR = "sharpness"                     # встроенный anchor
    PROMPT_CROSSTALK_POS = "a single solid object without any ghosting or double image"
    PROMPT_CROSSTALK_NEG = "an image with noticeable ghosting, double edges, or motion artifacts"

    # ---------- Классовые переменные для кэширования метрик ----------
    _blur_metric = None
    _crosstalk_metric = None
    _device = None

    def __init__(self, base_params, frames, obj_mask=None, device=None, batch_size=16, equalize_clahe=False,blur_use_mask=False):
        self.base_params = base_params
        self.frames = frames
        self.obj_mask = obj_mask
        self.device = device if device is not None else torch.device('cpu')
        self.batch_size = batch_size
        self.equalize_clahe = equalize_clahe
        self.blur_use_mask = blur_use_mask
        self.results = []
        self._init_metrics()

    # ------------------------------------------------------------------
    # Инициализация метрик CLIP‑IQA (один раз для всех экземпляров)
    # ------------------------------------------------------------------
    @classmethod
    def _init_metrics(cls):
        if cls._blur_metric is not None:
            return

        cls._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Резкость: абсолютная оценка, ОБЯЗАТЕЛЬНО кортеж из одного промпта
        cls._blur_metric = CLIPImageQualityAssessment("openai/clip-vit-large-patch14-336",
            prompts=("sharpness",) # ← именно кортеж
        ).to(cls._device)

        # Двоение (crosstalk): относительная оценка – пара промптов
        cls._crosstalk_metric = CLIPImageQualityAssessment("openai/clip-vit-large-patch14-336",
            prompts=("strobe_crosstalk",)
        ).to(cls._device)

    # ------------------------------------------------------------------
    # Публичные методы оценки
    # ------------------------------------------------------------------
    @classmethod
    def compute_clipiqa_blur(cls, image, obj_mask=None, use_mask=False):
        tensor = cls._image_to_tensor(image, obj_mask=obj_mask, use_mask=use_mask)
        return float(cls._blur_metric(tensor).item())

    @classmethod
    def compute_clipiqa_crosstalk(cls, image, obj_mask=None):
        """
        Возвращает оценку отсутствия двоения (0‑1, выше = меньше ghosting).
        Если obj_mask задана, обрезаем изображение до bounding‑box объекта
        (с запасом), чтобы анализировать только область с возможным двоением.
        """
        tensor = cls._image_to_tensor(image, obj_mask=obj_mask, use_mask=True)
        return float(cls._crosstalk_metric(tensor).item())

    # ------------------------------------------------------------------
    # Вспомогательный метод: numpy -> torch tensor (B=1, C=3, H, W)
    # ------------------------------------------------------------------
    @classmethod
    def _image_to_tensor(cls, image, obj_mask=None, use_mask=False):
        img = np.asarray(image, dtype=np.float32)
        if img.max() > 1.5:   # предполагаем 0..255
            img /= 255.0

        # Если нужен обрез по маске (только для crosstalk)
        if use_mask and obj_mask is not None and obj_mask.any():
            rows = np.any(obj_mask, axis=1)
            cols = np.any(obj_mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            margin = max(5, int((xmax - xmin) * 0.2))
            xmin = max(0, xmin - margin)
            xmax = min(img.shape[1] - 1, xmax + margin)
            ymin = max(0, ymin - margin)
            ymax = min(img.shape[0] - 1, ymax + margin)
            img = img[ymin:ymax+1, xmin:xmax+1, :]

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(cls._device)
    # ----------------------- Метод run_test – использует новые метрики -----------------------
    def run_test(self, param_ranges, fixed_params=None, progress_callback=None):
        import itertools

        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        combinations = list(itertools.product(*values))
        total = len(combinations)
        self.results = []

        fixed_dict = fixed_params if fixed_params else {}

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_combos = combinations[batch_start:batch_end]
            batch_params = []
            for combo in batch_combos:
                params = copy.deepcopy(self.base_params)
                for k, v in fixed_dict.items():
                    setattr(params, k, v)
                for k, v in zip(keys, combo):
                    setattr(params, k, v)
                params.update_T_frame()
                batch_params.append(params)

            sim = BatchGPUSimulator(batch_params, self.frames, obj_mask=self.obj_mask, device=self.device)
            results_no, results_with = sim.run()

            for idx_in_batch, (params, combo) in enumerate(zip(batch_params, batch_combos)):
                img_no = results_no[idx_in_batch]
                img_with = results_with[idx_in_batch]

                # Применяем локальное адаптивное выравнивание, если включено
                if self.equalize_clahe:
                    img_no = self.apply_clahe(img_no)
                    img_with = self.apply_clahe(img_with)

                # НОВОЕ: оценка через CLIP-IQA
                blur_no = self.compute_clipiqa_blur(img_no, obj_mask=self.obj_mask,
                                                    use_mask=self.blur_use_mask)
                blur_with = self.compute_clipiqa_blur(img_with, obj_mask=self.obj_mask,
                                                      use_mask=self.blur_use_mask)
                crosstalk_no = self.compute_clipiqa_crosstalk(img_no, obj_mask=self.obj_mask)
                crosstalk_with = self.compute_clipiqa_crosstalk(img_with, obj_mask=self.obj_mask)

                result = {
                    **{k: v for k, v in zip(keys, combo)},
                    'blur_no_glasses': blur_no,
                    'blur_with_glasses': blur_with,
                    'improvement': blur_with - blur_no,   # улучшение резкости: положительное = стало чётче
                    'crosstalk_no_glasses': crosstalk_no,
                    'crosstalk_with_glasses': crosstalk_with,
                    'crosstalk_improvement': crosstalk_with - crosstalk_no  # положительное = меньше двоения
                }
                self.results.append(result)

                if progress_callback:
                    current = batch_start + idx_in_batch + 1
                    progress_callback(current, total, result)

        return self.results

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Применяет CLAHE (Contrast Limited Adaptive Histogram Equalization)
        к каналу L изображения в LAB.
        Вход: numpy-массив (H, W, 3) float [0, 1].
        Выход: обработанный массив того же диапазона.
        """
        img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return rgb_eq.astype(np.float32) / 255.0

    def generate_report(self, output_dir=None, base_name="blur_test_report", plot_types=None):
        """
        Создаёт CSV-файл и набор графиков с результатами тестирования.

        Параметры:
            output_dir : str – папка для сохранения.
            base_name : str – базовое имя файлов.
            plot_types : list – типы графиков для построения:
                ['line', 'hist', 'heatmap', 'box', 'pairplot', '3d'].
                Если None, строит все доступные в зависимости от числа параметров.
        """
        if not self.results:
            raise ValueError("Нет результатов тестирования. Сначала запустите run_test().")

        df = pd.DataFrame(self.results)
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        # Сохраняем CSV
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Отчёт CSV сохранён в {csv_path}")

        # Определяем варьируемые параметры (те, которые имеют более одного уникального значения)
        varying_params = []
        metric_cols = ['blur_no_glasses', 'blur_with_glasses', 'improvement',
                       'crosstalk_no_glasses', 'crosstalk_with_glasses', 'crosstalk_improvement']
        for col in df.columns:
            if col not in metric_cols:
                if df[col].nunique() > 1:
                    varying_params.append(col)

        if not varying_params:
            print("Нет варьируемых параметров, графики не строятся.")
            return csv_path

        # Создаём подпапку для графиков
        plots_dir = os.path.join(output_dir, f"{base_name}_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Определяем, какие типы графиков строить
        if plot_types is None:
            plot_types = ['line', 'hist', 'box']
            if len(varying_params) >= 2:
                plot_types.append('heatmap')
            if len(varying_params) >= 3:
                plot_types.append('pairplot')
                plot_types.append('3d')

        # 1. Для каждого параметра строим линейный график зависимости метрик
        if 'line' in plot_types:
            for param in varying_params:
                # Группировка для blur-метрик
                grouped = df.groupby(param)[
                    ['blur_no_glasses', 'blur_with_glasses', 'improvement']
                ].mean().reset_index()

                # Отдельная группировка для crosstalk-метрик
                grouped_cross = df.groupby(param)[
                    ['crosstalk_no_glasses', 'crosstalk_with_glasses', 'crosstalk_improvement']
                ].mean().reset_index()

                # График размытия
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].plot(grouped[param], grouped['blur_no_glasses'], 'o-', label='Без очков')
                axes[0].plot(grouped[param], grouped['blur_with_glasses'], 's-', label='С очками')
                axes[0].set_xlabel(param)
                axes[0].set_ylabel('Четкость')
                axes[0].set_title(f'Зависимость резкости от {param}')
                axes[0].legend()
                axes[0].grid(True)

                axes[1].plot(grouped[param], grouped['improvement'], 'd-', color='green')
                axes[1].set_xlabel(param)
                axes[1].set_ylabel('Улучшение резкости')
                axes[1].set_title(f'Улучшение от очков при вариации {param}')
                axes[1].grid(True)

                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"{base_name}_line_{param}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

                # График crosstalk (используем grouped_cross)
                fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
                axes2[0].plot(grouped_cross[param], grouped_cross['crosstalk_no_glasses'],
                              'o-', label='Без очков', color='purple')
                axes2[0].plot(grouped_cross[param], grouped_cross['crosstalk_with_glasses'],
                              's-', label='С очками', color='magenta')
                axes2[0].set_xlabel(param)
                axes2[0].set_ylabel('Двоение')
                axes2[0].set_title(f'Зависимость crosstalk от {param}')
                axes2[0].legend()
                axes2[0].grid(True)

                axes2[1].plot(grouped_cross[param], grouped_cross['crosstalk_improvement'],
                              'd-', color='cyan')
                axes2[1].set_xlabel(param)
                axes2[1].set_ylabel('Ухудшение/улучшение двоения')
                axes2[1].set_title(f'Улучшение crosstalk при вариации {param}')
                axes2[1].grid(True)

                plt.tight_layout()
                fig2.savefig(os.path.join(plots_dir, f"{base_name}_crosstalk_line_{param}.png"), dpi=150,
                             bbox_inches='tight')
                plt.close(fig2)

        # 2. Гистограммы распределения метрик
        if 'hist' in plot_types:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes[0, 0].hist(df['blur_no_glasses'], bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('Размытие без очков')
            axes[0, 1].hist(df['blur_with_glasses'], bins=20, alpha=0.7, color='orange')
            axes[0, 1].set_title('Размытие с очками')
            axes[0, 2].hist(df['improvement'], bins=20, alpha=0.7, color='green')
            axes[0, 2].set_title('Улучшение резкости')

            axes[1, 0].hist(df['crosstalk_no_glasses'], bins=20, alpha=0.7, color='purple')
            axes[1, 0].set_title('Двоение без очков')
            axes[1, 1].hist(df['crosstalk_with_glasses'], bins=20, alpha=0.7, color='magenta')
            axes[1, 1].set_title('Двоение с очками')
            axes[1, 2].hist(df['crosstalk_improvement'], bins=20, alpha=0.7, color='cyan')
            axes[1, 2].set_title('Уменьшение двоения')
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"{base_name}_histograms.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

        # 3. Бокс-плоты для дискретных параметров
        if 'box' in plot_types:
            for param in varying_params:
                # Если параметр имеет небольшое количество уникальных значений (<=10)
                if df[param].nunique() <= 10:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    df.boxplot(column='blur_no_glasses', by=param, ax=axes[0])
                    axes[0].set_title(f'Размытие без очков по {param}')
                    axes[0].set_xlabel(param)
                    axes[0].set_ylabel('Дисперсия Лапласиана')
                    df.boxplot(column='blur_with_glasses', by=param, ax=axes[1])
                    axes[1].set_title(f'Размытие с очками по {param}')
                    axes[1].set_xlabel(param)
                    axes[1].set_ylabel('Дисперсия Лапласиана')
                    plt.suptitle('')
                    plt.tight_layout()
                    fig.savefig(os.path.join(plots_dir, f"{base_name}_box_{param}.png"), dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    # Box-плоты для crosstalk
                    fig_cb, axes_cb = plt.subplots(1, 2, figsize=(12, 5))
                    df.boxplot(column='crosstalk_no_glasses', by=param, ax=axes_cb[0])
                    axes_cb[0].set_title(f'Двоение без очков по {param}')
                    axes_cb[0].set_xlabel(param)
                    axes_cb[0].set_ylabel('Двоение')
                    df.boxplot(column='crosstalk_with_glasses', by=param, ax=axes_cb[1])
                    axes_cb[1].set_title(f'Двоение с очками по {param}')
                    axes_cb[1].set_xlabel(param)
                    axes_cb[1].set_ylabel('Двоение')
                    plt.suptitle('')
                    plt.tight_layout()
                    fig_cb.savefig(os.path.join(plots_dir, f"{base_name}_crosstalk_box_{param}.png"), dpi=150,
                                   bbox_inches='tight')
                    plt.close(fig_cb)

        # 4. Тепловая карта для двух варьируемых параметров
        if 'heatmap' in plot_types and len(varying_params) >= 2:
            from matplotlib.colors import TwoSlopeNorm

            p1, p2 = varying_params[0], varying_params[1]
            pivot_no = df.pivot_table(index=p1, columns=p2, values='blur_no_glasses', aggfunc='mean')
            pivot_with = df.pivot_table(index=p1, columns=p2, values='blur_with_glasses', aggfunc='mean')
            pivot_imp = df.pivot_table(index=p1, columns=p2, values='improvement', aggfunc='mean')

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            im1 = axes[0].imshow(pivot_no.values, origin='lower', aspect='auto', cmap='viridis')
            axes[0].set_xticks(range(len(pivot_no.columns)))
            axes[0].set_xticklabels([f"{x:.2g}" for x in pivot_no.columns])
            axes[0].set_yticks(range(len(pivot_no.index)))
            axes[0].set_yticklabels([f"{x:.2g}" for x in pivot_no.index])
            axes[0].set_xlabel(p2)
            axes[0].set_ylabel(p1)
            axes[0].set_title('Размытие без очков')
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(pivot_with.values, origin='lower', aspect='auto', cmap='viridis')
            axes[1].set_xticks(range(len(pivot_with.columns)))
            axes[1].set_xticklabels([f"{x:.2g}" for x in pivot_with.columns])
            axes[1].set_yticks(range(len(pivot_with.index)))
            axes[1].set_yticklabels([f"{x:.2g}" for x in pivot_with.index])
            axes[1].set_xlabel(p2)
            axes[1].set_ylabel(p1)
            axes[1].set_title('Размытие с очками')
            plt.colorbar(im2, ax=axes[1])

            # Улучшение: центрирование на 0, только если 0 между min и max
            vmin_imp = pivot_imp.values.min()
            vmax_imp = pivot_imp.values.max()
            if vmin_imp < 0 < vmax_imp:
                norm_imp = TwoSlopeNorm(vmin=vmin_imp, vcenter=0, vmax=vmax_imp)
                cmap_imp = 'RdBu'
            else:
                norm_imp = None
                cmap_imp = 'viridis'
            im3 = axes[2].imshow(pivot_imp.values, origin='lower', aspect='auto',
                                 cmap=cmap_imp, norm=norm_imp)
            axes[2].set_xticks(range(len(pivot_imp.columns)))
            axes[2].set_xticklabels([f"{x:.2g}" for x in pivot_imp.columns])
            axes[2].set_yticks(range(len(pivot_imp.index)))
            axes[2].set_yticklabels([f"{x:.2g}" for x in pivot_imp.index])
            axes[2].set_xlabel(p2)
            axes[2].set_ylabel(p1)
            axes[2].set_title('Улучшение резкости')
            plt.colorbar(im3, ax=axes[2])

            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"{base_name}_heatmap_{p1}_{p2}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Тепловая карта crosstalk
            pivot_cross_no = df.pivot_table(index=p1, columns=p2, values='crosstalk_no_glasses', aggfunc='mean')
            pivot_cross_with = df.pivot_table(index=p1, columns=p2, values='crosstalk_with_glasses', aggfunc='mean')
            pivot_cross_imp = df.pivot_table(index=p1, columns=p2, values='crosstalk_improvement', aggfunc='mean')

            fig_cross, axes_cross = plt.subplots(1, 3, figsize=(18, 5))
            im1c = axes_cross[0].imshow(pivot_cross_no.values, origin='lower', aspect='auto', cmap='plasma')
            axes_cross[0].set_xticks(range(len(pivot_cross_no.columns)))
            axes_cross[0].set_xticklabels([f"{x:.2g}" for x in pivot_cross_no.columns])
            axes_cross[0].set_yticks(range(len(pivot_cross_no.index)))
            axes_cross[0].set_yticklabels([f"{x:.2g}" for x in pivot_cross_no.index])
            axes_cross[0].set_xlabel(p2)
            axes_cross[0].set_ylabel(p1)
            axes_cross[0].set_title('Crosstalk без очков')
            plt.colorbar(im1c, ax=axes_cross[0])

            im2c = axes_cross[1].imshow(pivot_cross_with.values, origin='lower', aspect='auto', cmap='plasma')
            axes_cross[1].set_xticks(range(len(pivot_cross_with.columns)))
            axes_cross[1].set_xticklabels([f"{x:.2g}" for x in pivot_cross_with.columns])
            axes_cross[1].set_yticks(range(len(pivot_cross_with.index)))
            axes_cross[1].set_yticklabels([f"{x:.2g}" for x in pivot_cross_with.index])
            axes_cross[1].set_xlabel(p2)
            axes_cross[1].set_ylabel(p1)
            axes_cross[1].set_title('Crosstalk с очками')
            plt.colorbar(im2c, ax=axes_cross[1])

            # Улучшение crosstalk: центрирование на 0, только если 0 между min и max
            vmin_cross = pivot_cross_imp.values.min()
            vmax_cross = pivot_cross_imp.values.max()
            if vmin_cross < 0 < vmax_cross:
                norm_cross = TwoSlopeNorm(vmin=vmin_cross, vcenter=0, vmax=vmax_cross)
                cmap_cross = 'coolwarm'
            else:
                norm_cross = None
                cmap_cross = 'plasma'
            im3c = axes_cross[2].imshow(pivot_cross_imp.values, origin='lower', aspect='auto',
                                        cmap=cmap_cross, norm=norm_cross)
            axes_cross[2].set_xticks(range(len(pivot_cross_imp.columns)))
            axes_cross[2].set_xticklabels([f"{x:.2g}" for x in pivot_cross_imp.columns])
            axes_cross[2].set_yticks(range(len(pivot_cross_imp.index)))
            axes_cross[2].set_yticklabels([f"{x:.2g}" for x in pivot_cross_imp.index])
            axes_cross[2].set_xlabel(p2)
            axes_cross[2].set_ylabel(p1)
            axes_cross[2].set_title('Улучшение crosstalk')
            plt.colorbar(im3c, ax=axes_cross[2])

            plt.tight_layout()
            fig_cross.savefig(os.path.join(plots_dir, f"{base_name}_crosstalk_heatmap_{p1}_{p2}.png"), dpi=150,
                              bbox_inches='tight')
            plt.close(fig_cross)

        # 5. Pairplot (матрица диаграмм рассеяния) для всех числовых параметров и метрик
        if 'pairplot' in plot_types and len(varying_params) >= 2:
            try:
                import seaborn as sns
                numeric_cols = varying_params + ['blur_no_glasses', 'blur_with_glasses', 'improvement']
                pair_df = df[numeric_cols].dropna()
                if not pair_df.empty:
                    g = sns.pairplot(pair_df, diag_kind='kde')
                    g.fig.suptitle('Матрица парных зависимостей', y=1.02)
                    g.savefig(os.path.join(plots_dir, f"{base_name}_pairplot.png"), dpi=150, bbox_inches='tight')
                    plt.close(g.fig)
            except ImportError:
                print("Seaborn не установлен, pairplot пропущен.")

        # 6. 3D-график для двух параметров и метрики
        if '3d' in plot_types and len(varying_params) >= 2:
            from mpl_toolkits.mplot3d import Axes3D
            p1, p2 = varying_params[0], varying_params[1]
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            grouped = df.groupby([p1, p2])[['blur_no_glasses', 'blur_with_glasses']].mean().reset_index()
            ax.plot_trisurf(grouped[p1], grouped[p2], grouped['blur_no_glasses'], cmap='Blues', alpha=0.7,
                            label='Без очков')
            ax.plot_trisurf(grouped[p1], grouped[p2], grouped['blur_with_glasses'], cmap='Oranges', alpha=0.7,
                            label='С очками')
            ax.set_xlabel(p1)
            ax.set_ylabel(p2)
            ax.set_zlabel('Резкость')
            ax.set_title(f'Зависимость резкости от {p1} и {p2}')
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"{base_name}_3d_{p1}_{p2}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Опционально: создаём HTML-отчёт с Plotly (если установлен)
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            html_path = os.path.join(output_dir, f"{base_name}_interactive.html")
            figs = []

            if len(varying_params) >= 1:
                p = varying_params[0]
                fig1 = px.line(df, x=p, y=['blur_no_glasses', 'blur_with_glasses'],
                               title=f'Размытие vs {p}', markers=True)
                figs.append(fig1)

            if len(varying_params) >= 2:
                p1, p2 = varying_params[0], varying_params[1]
                pivot = df.pivot_table(index=p1, columns=p2, values='blur_no_glasses', aggfunc='mean')
                fig2 = px.imshow(pivot, title=f'Тепловая карта размытия без очков ({p1} vs {p2})')
                figs.append(fig2)

            with open(html_path, 'w', encoding='utf-8') as f:
                for i, fig in enumerate(figs):
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn' if i == 0 else False))
                    f.write("<br><hr><br>")
            print(f"Интерактивный отчёт сохранён в {html_path}")
        except ImportError:
            pass

        print(f"Все графики сохранены в {plots_dir}")
        return csv_path

# ----------------------------------------------------------------------
# Параметры симуляции
# ----------------------------------------------------------------------
class SimParams:
    def __init__(self):
        self.fps = 60
        self.T_frame = 1.0 / self.fps
        self.speed = 5.0
        self.tau_rise = 0.005
        self.tau_fall = 0.005
        self.T_scan = self.T_frame
        self.backlight_mode = 'constant'
        self.backlight_duration = 30
        self.backlight_phase = 0
        self.glasses_enabled = True
        self.tau_glasses = 0.002
        self.glasses_duration = 20
        self.glasses_phase = 0
        self.eye_sigma = 0.0
        self.dt = self.T_frame / 500
        self.num_frames = 4
        self.tracking_mode = 'fixed'
        self.equalize_hist = False
        self.vblank_percent = 0
        self.use_response_table = False
        self.response_table = None

        # Параметры синхронизации и аппаратных неидеальностей
        self.sync_jitter = 0.0
        self.show_carrier = False
        self.show_sync_pulses = False

        # Фотоприёмник и синхросигнал
        self.sync_threshold = 0.5
        self.sync_hysteresis = 0.02
        self.sync_false_positive_rate = 0.0
        self.sync_false_negative_rate = 0.0
        self.backlight_brightness = 1.0

        # Драйверы затворов
        self.driver_rise_time = 0.0
        self.driver_ringing = False
        self.driver_ringing_amplitude = 0.05

        # Тактовый генератор
        self.mcu_freq = 8.0

        # Паразитные связи
        self.channel_skew = 0.0
        self.sync_crosstalk = 0.0
        self.show_second_channel = False

        # Синхромаркер в углу
        self.sync_pattern_enabled = True
        self.wrap_around = False             # циклическое движение объекта по горизонтали

        self.sync_mode = 'internal'  # 'internal' (по таймеру) или 'pattern' (по маркеру)
        self.photo_tau = 1e-6  # постоянная времени фотодиода, сек
        self.photo_noise_std = 0.0  # СКО шума фотоприёмника (0..1)
        self.sync_min_period = 0.5 * self.T_frame  # минимальный интервал между синхроимпульсами


        self.update_T_frame()

    def update_T_frame(self):
        self.T_frame = 1.0 / self.fps
        self.T_scan = self.T_frame * (1.0 - self.vblank_percent / 100.0)
        self.dt = self.T_frame / 500
        # обновляем зависимые от T_frame параметры
        self.sync_min_period = 0.5 * self.T_frame

    def min_time_constant(self):
        taus = [self.tau_rise, self.tau_fall]
        if self.glasses_enabled and self.tau_glasses > 0:
            taus.append(self.tau_glasses)
        if self.use_response_table and self.response_table is not None:
            min_table = min(np.min(self.response_table.rise_matrix),
                            np.min(self.response_table.fall_matrix))
            taus.append(min_table)
        return min(taus)

class GPUSimulator:
    def __init__(self, params, frames, obj_mask=None, device=None):
        self.p = params
        self.frames = [torch.from_numpy(f).to(device).float() for f in frames]
        self.H, self.W, self.C = frames[0].shape
        self.obj_mask = obj_mask

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if obj_mask is not None:
            self.obj_mask = torch.from_numpy(obj_mask).to(self.device).bool()
        else:
            self.obj_mask = None

        min_tau = self.p.min_time_constant()
        if min_tau > 0:
            self.p.dt = min(self.p.dt, min_tau / 20.0)

        self.t_start = 0.0
        self.t_end = self.p.num_frames * self.p.T_frame
        self.Nt = int((self.t_end - self.t_start) / self.p.dt) + 1
        self.t_cpu = np.linspace(self.t_start, self.t_end, self.Nt)

        self.backlight_history = np.zeros(self.Nt)
        self.glasses_history = np.zeros(self.Nt)
        self.glasses_history_second = np.zeros(self.Nt)
        self.pixel_history = np.zeros((self.Nt, 3))

        self.image_no_glasses = torch.zeros((self.H, self.W, self.C), device=self.device)
        self.image_with_glasses = torch.zeros((self.H, self.W, self.C), device=self.device)

        self.y = torch.arange(self.H, device=self.device).float()

        self.rise_tensor = None
        self.fall_tensor = None
        if self.p.use_response_table and self.p.response_table is not None:
            self.rise_tensor = torch.from_numpy(self.p.response_table.rise_matrix).to(self.device)
            self.fall_tensor = torch.from_numpy(self.p.response_table.fall_matrix).to(self.device)

        self._precompute_driver_waveforms()
        self._init_sync_state()

    def _init_sync_state(self):
        self.sync_state = 0.0
        self.photo_voltage = 0.0
        self.current_frame_idx = -1
        self.t_frame_start = -self.p.T_frame
        self.last_sync_time = -self.p.T_frame * 2
        self.glasses_time_offset = 0.0  # ← сдвиг фазы очков относительно идеального таймера

    def _get_pattern_brightness(self, frame_idx: int, backlight: float) -> float:
        if not self.p.sync_pattern_enabled:
            return 0.0
        pattern_value = 1.0 if (frame_idx % 2 == 0) else 0.0
        return pattern_value * backlight

    def _update_sync(self, t: float, pattern_brightness: float):
        if self.p.sync_mode != 'pattern':
            return

        dt = self.p.dt
        alpha = dt / (self.p.photo_tau + dt) if self.p.photo_tau > 0 else 1.0
        noise = np.random.normal(0, self.p.photo_noise_std) if self.p.photo_noise_std > 0 else 0.0
        self.photo_voltage += alpha * (pattern_brightness + noise - self.photo_voltage)

        high = self.p.sync_threshold
        low = max(0.0, high - self.p.sync_hysteresis)

        new_state = self.sync_state
        if self.photo_voltage > high:
            new_state = 1.0
        elif self.photo_voltage < low:
            new_state = 0.0

        if not self.sync_state and new_state:
            if t - self.last_sync_time >= self.p.sync_min_period:
                self.t_frame_start = t
                if self.current_frame_idx < self.p.num_frames - 1:
                    self.current_frame_idx += 1
                self.last_sync_time = t
                # Вычисляем сдвиг фазы очков относительно идеального кадра
                ideal_k = int(round(t / self.p.T_frame))
                self.glasses_time_offset = t - ideal_k * self.p.T_frame

        self.sync_state = new_state

    def _precompute_driver_waveforms(self):
        t = self.t_cpu
        dt = t[1] - t[0]
        freq_scale = 8.0 / self.p.mcu_freq
        carrier = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 500 * freq_scale * t)))

        if self.p.driver_rise_time > 0:
            tau_drv = self.p.driver_rise_time * 1e-6 / 2.2
            alpha = dt / (tau_drv + dt)
            filtered = np.zeros_like(carrier)
            for i in range(1, len(t)):
                filtered[i] = filtered[i-1] + alpha * (carrier[i] - filtered[i-1])
            carrier = filtered

        if self.p.driver_ringing:
            ringing = np.zeros_like(t)
            for i in range(1, len(t)):
                if abs(carrier[i] - carrier[i-1]) > 0.5:
                    t0 = t[i]
                    for j in range(i, min(i + 50, len(t))):
                        dt_ring = t[j] - t0
                        ringing[j] += self.p.driver_ringing_amplitude * np.exp(-dt_ring/1e-7) * np.sin(2*np.pi*5e6*dt_ring)
            carrier = np.clip(carrier + ringing, 0, 1)

        self.cached_carrier_no_glitch = carrier.copy()
        self.driver_noise = np.zeros_like(t)

    def backlight(self, t):
        if self.p.backlight_mode == 'constant':
            return 1.0
        phase = (t % self.p.T_frame) / self.p.T_frame
        start = self.p.backlight_phase / 100.0
        end = start + self.p.backlight_duration / 100.0
        if start <= phase < end or (end > 1.0 and phase < end - 1.0):
            return 1.0
        return 0.0

    def glasses_ideal(self, t):
        if not self.p.glasses_enabled:
            return 1.0
        phase = (t % self.p.T_frame) / self.p.T_frame
        start = self.p.glasses_phase / 100.0
        end = start + self.p.glasses_duration / 100.0
        if start <= phase < end or (end > 1.0 and phase < end - 1.0):
            return 1.0
        return 0.0

    def get_tau_matrix(self, old_frame, new_frame):
        if self.p.use_response_table and self.rise_tensor is not None:
            old_gray = (0.299 * old_frame[..., 0] + 0.587 * old_frame[..., 1] + 0.114 * old_frame[..., 2]) * 255.0
            new_gray = (0.299 * new_frame[..., 0] + 0.587 * new_frame[..., 1] + 0.114 * new_frame[..., 2]) * 255.0
            old_gray = torch.clamp(torch.round(old_gray), 0, 255).long()
            new_gray = torch.clamp(torch.round(new_gray), 0, 255).long()
            tau = torch.where(new_gray >= old_gray,
                              self.rise_tensor[old_gray, new_gray],
                              self.fall_tensor[old_gray, new_gray])
            return tau.unsqueeze(-1)
        else:
            tau_rise = torch.tensor(self.p.tau_rise, device=self.device)
            tau_fall = torch.tensor(self.p.tau_fall, device=self.device)
            direction = new_frame[..., 0] > old_frame[..., 0]
            return torch.where(direction, tau_rise, tau_fall).unsqueeze(-1)

    def _get_current_frame_idx(self, t):
        """Возвращает индекс текущего кадра."""
        if self.p.sync_mode == 'pattern':
            return max(0, self.current_frame_idx)
        else:
            return int(t // self.p.T_frame)

    def run(self):
        if self.p.tracking_mode == 'fixed':
            self._run_fixed()
        else:
            self._run_smooth_pursuit()

        self.result_no_glasses = np.clip(self.image_no_glasses.cpu().numpy(), 0, 1)
        self.result_with_glasses = np.clip(self.image_with_glasses.cpu().numpy(), 0, 1)

        if self.p.eye_sigma > 0:
            self.result_no_glasses = cv2.GaussianBlur(self.result_no_glasses, (0,0), sigmaX=self.p.eye_sigma, sigmaY=self.p.eye_sigma)
            self.result_with_glasses = cv2.GaussianBlur(self.result_with_glasses, (0,0), sigmaX=self.p.eye_sigma, sigmaY=self.p.eye_sigma)

    def _run_fixed(self):
        self.image_no_glasses.zero_()
        self.image_with_glasses.zero_()

        g_filtered = 1.0
        alpha_g = 1.0 - np.exp(-self.p.dt / self.p.tau_glasses) if self.p.tau_glasses > 0 else 1.0

        self._init_sync_state()
        prev_n = -1

        for i in range(self.Nt):
            t = self.t_cpu[i]
            b = self.backlight(t)

            # --- Очки: фаза сдвинута на glasses_time_offset, если включён pattern sync ---
            if self.p.sync_mode == 'pattern':
                t_glasses = t - self.glasses_time_offset
            else:
                t_glasses = t
            g_ideal = self.glasses_ideal(t_glasses)

            # --- Синхронизация по маркеру (только обновление состояния) ---
            if self.p.sync_mode == 'pattern':
                old_n = max(0, self.current_frame_idx) if self.current_frame_idx >= 0 else 0
                pattern_b = self._get_pattern_brightness(old_n, b)
                self._update_sync(t, pattern_b)

            # --- Развёртка всегда по внутреннему таймеру ---
            n = int(t // self.p.T_frame)
            n = min(n, self.p.num_frames - 1)
            n_next = min(n + 1, self.p.num_frames - 1)

            if n != prev_n:
                old_frame = self.frames[n]
                new_frame = self.frames[n_next]
                tau = self.get_tau_matrix(old_frame, new_frame)
                prev_n = n

            t_update = n * self.p.T_frame + (self.y / self.H) * self.p.T_scan
            dt_update = torch.clamp(t - t_update, min=0.0).unsqueeze(1).unsqueeze(1)
            mask_updated = (t >= t_update).float().unsqueeze(1).unsqueeze(1)

            alpha = 1.0 - torch.exp(-dt_update / tau)
            p = old_frame + (new_frame - old_frame) * alpha * mask_updated

            if self.p.glasses_enabled and self.p.tau_glasses > 0:
                g_filtered += alpha_g * (g_ideal - g_filtered)
                g = g_filtered
            else:
                g = g_ideal

            contrib = p * b * self.p.dt
            self.image_no_glasses += contrib
            self.image_with_glasses += contrib * g

            debug_y, debug_x = self.H // 2, self.W // 2
            self.pixel_history[i] = p[debug_y, debug_x].cpu().numpy()

        total_time = self.t_end - self.t_start
        self.image_no_glasses /= total_time
        self.image_with_glasses /= total_time

    def _run_smooth_pursuit(self):
        if self.obj_mask is None:
            self._run_fixed()
            return

        y_idx, x_idx = torch.where(self.obj_mask)
        X0 = float(x_idx.float().mean().item())
        Y0 = float(y_idx.float().mean().item())

        Yr = torch.arange(self.H, device=self.device).float().view(-1, 1).expand(self.H, self.W)
        Xr = torch.arange(self.W, device=self.device).float().view(1, -1).expand(self.H, self.W)

        self.image_no_glasses.zero_()
        self.image_with_glasses.zero_()

        g_filtered = 1.0
        alpha_g = 1.0 - np.exp(-self.p.dt / self.p.tau_glasses) if self.p.tau_glasses > 0 else 1.0

        self._init_sync_state()
        prev_n = -1

        for i in range(self.Nt):
            t = self.t_cpu[i]
            b = self.backlight(t)

            # --- Очки ---
            if self.p.sync_mode == 'pattern':
                t_glasses = t - self.glasses_time_offset
            else:
                t_glasses = t
            g_ideal = self.glasses_ideal(t_glasses)

            # --- Синхронизация ---
            if self.p.sync_mode == 'pattern':
                old_n = max(0, self.current_frame_idx) if self.current_frame_idx >= 0 else 0
                pattern_b = self._get_pattern_brightness(old_n, b)
                self._update_sync(t, pattern_b)

            # --- Внутренний таймер ---
            n = int(t // self.p.T_frame)
            n = min(n, self.p.num_frames - 1)
            n_next = min(n + 1, self.p.num_frames - 1)

            if n != prev_n:
                old_frame = self.frames[n]
                new_frame = self.frames[n_next]
                tau_full = self.get_tau_matrix(old_frame, new_frame)
                prev_n = n

            frac = (t - n * self.p.T_frame) / self.p.T_frame
            Xobj = X0 + (n + frac) * self.p.speed
            Xsrc = Xr + (Xobj - X0)
            Ysrc = Yr

            if self.p.wrap_around:
                Xsrc = Xsrc % self.W
                valid = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)
            else:
                valid = (Xsrc >= 0) & (Xsrc < self.W) & (Ysrc >= 0) & (Ysrc < self.H)

            Xsrc_int = torch.round(Xsrc).long().clamp(0, self.W - 1)
            Ysrc_int = torch.round(Ysrc).long().clamp(0, self.H - 1)

            t_update = n * self.p.T_frame + (Ysrc / self.H) * self.p.T_scan
            dt_update = torch.clamp(t - t_update, min=0.0).unsqueeze(-1)
            mask_updated = (t >= t_update).float().unsqueeze(-1)

            old_val = old_frame[Ysrc_int, Xsrc_int]
            new_val = new_frame[Ysrc_int, Xsrc_int]
            tau = tau_full[Ysrc_int, Xsrc_int]

            alpha = 1.0 - torch.exp(-dt_update / tau)
            p = old_val + (new_val - old_val) * alpha * mask_updated * valid.unsqueeze(-1)

            if self.p.glasses_enabled and self.p.tau_glasses > 0:
                g_filtered += alpha_g * (g_ideal - g_filtered)
                g = g_filtered
            else:
                g = g_ideal

            contrib = p * b * self.p.dt
            self.image_no_glasses += contrib
            self.image_with_glasses += contrib * g

            debug_y = int(Y0)
            debug_x = int(X0)
            self.pixel_history[i] = p[debug_y, debug_x].cpu().numpy()

        total_time = self.t_end - self.t_start
        self.image_no_glasses /= total_time
        self.image_with_glasses /= total_time

class BatchGPUSimulator:
    def __init__(self, params_list, frames, obj_mask=None, device=None):
        self.batch_size = len(params_list)
        if self.batch_size == 0:
            raise ValueError("params_list пуст")

        # Проверка общих временны́х и геометрических параметров
        p0 = params_list[0]
        for p in params_list:
            if p.fps != p0.fps or p.num_frames != p0.num_frames:
                raise ValueError("fps и num_frames должны совпадать")
            if p.tracking_mode != p0.tracking_mode:
                raise ValueError("tracking_mode должен быть одинаковым")
            if p.vblank_percent != p0.vblank_percent:
                raise ValueError("vblank_percent должен быть одинаковым")
            if p.speed != p0.speed:
                raise ValueError("speed должна быть одинаковой")

        self.params_list = params_list
        self.fps = p0.fps
        self.num_frames = p0.num_frames
        self.tracking_mode = p0.tracking_mode
        self.vblank_percent = p0.vblank_percent
        self.speed = p0.speed
        self.T_frame = 1.0 / self.fps
        self.T_scan = self.T_frame * (1.0 - self.vblank_percent / 100.0)

        # Определяем dt (минимальный среди всех экземпляров)
        min_tau = min(p.min_time_constant() for p in params_list)
        dt_base = self.T_frame / 500
        self.dt = min(dt_base, min_tau / 20.0) if min_tau > 0 else dt_base

        self.t_start = 0.0
        self.t_end = self.num_frames * self.T_frame
        self.Nt = int((self.t_end - self.t_start) / self.dt) + 1
        self.t_cpu = np.linspace(self.t_start, self.t_end, self.Nt)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Кадры (одинаковы для всех)
        self.frames = [torch.from_numpy(f).to(self.device).float() for f in frames]
        self.H, self.W, self.C = frames[0].shape

        # Маска объекта (если есть)
        if obj_mask is not None:
            self.obj_mask = torch.from_numpy(obj_mask).to(self.device).bool()
        else:
            self.obj_mask = None

        # Координатные сетки (для возможного smooth_pursuit)
        self.y_grid = torch.arange(self.H, device=self.device).float()
        self.Xr = torch.arange(self.W, device=self.device).float().view(1, -1).expand(self.H, self.W)
        self.Yr = self.y_grid.view(-1, 1).expand(self.H, self.W)

        # --- Параметры, зависящие от экземпляра (B,) ---
        self.tau_rise = torch.tensor([p.tau_rise for p in params_list], device=self.device)
        self.tau_fall = torch.tensor([p.tau_fall for p in params_list], device=self.device)
        self.tau_glasses = torch.tensor([p.tau_glasses for p in params_list], device=self.device)
        self.glasses_enabled = torch.tensor([p.glasses_enabled for p in params_list],
                                            device=self.device, dtype=torch.bool)
        self.eye_sigma = torch.tensor([p.eye_sigma for p in params_list], device=self.device)

        self.backlight_duration = torch.tensor([p.backlight_duration for p in params_list], device=self.device)
        self.backlight_phase = torch.tensor([p.backlight_phase for p in params_list], device=self.device)
        self.glasses_duration = torch.tensor([p.glasses_duration for p in params_list], device=self.device)
        self.glasses_phase = torch.tensor([p.glasses_phase for p in params_list], device=self.device)
        self.backlight_mode_constant = torch.tensor(
            [1.0 if p.backlight_mode == 'constant' else 0.0 for p in params_list], device=self.device)

        # Таблицы времени отклика
        self.use_table = [p.use_response_table for p in params_list]
        if any(self.use_table):
            rise_list, fall_list = [], []
            for p in params_list:
                if p.use_response_table and p.response_table is not None:
                    rise_list.append(torch.from_numpy(p.response_table.rise_matrix).to(self.device))
                    fall_list.append(torch.from_numpy(p.response_table.fall_matrix).to(self.device))
                else:
                    rise_list.append(torch.zeros(256, 256, device=self.device))
                    fall_list.append(torch.zeros(256, 256, device=self.device))
            self.rise_batch = torch.stack(rise_list)
            self.fall_batch = torch.stack(fall_list)
        else:
            self.rise_batch = None
            self.fall_batch = None

        # Параметры синхронизации (для pattern sync)
        sync_params = ['sync_mode', 'photo_tau', 'photo_noise_std', 'sync_threshold',
                       'sync_hysteresis', 'sync_false_positive_rate', 'sync_false_negative_rate']
        self.sync_modes = [p.sync_mode for p in params_list]
        # Проверяем, одинаковы ли все синхро-параметры, если хотя бы у одного pattern sync
        any_pattern = any(m == 'pattern' for m in self.sync_modes)
        if any_pattern:
            # Собираем значения для всех экземпляров
            self.sync_params_equal = all(
                all(getattr(p, name) == getattr(params_list[0], name) for p in params_list[1:])
                for name in sync_params
            )
        else:
            self.sync_params_equal = True   # не важно, pattern не используется

        # Если можем батчить синхронизацию, запоминаем общие параметры
        if any_pattern and self.sync_params_equal:
            p_sync = params_list[0]
            self.sync_threshold = p_sync.sync_threshold
            self.sync_hysteresis = p_sync.sync_hysteresis
            self.sync_false_positive_rate = p_sync.sync_false_positive_rate
            self.sync_false_negative_rate = p_sync.sync_false_negative_rate
            self.photo_tau = p_sync.photo_tau
            self.photo_noise_std = p_sync.photo_noise_std
            self.sync_min_period = p_sync.sync_min_period

        # Выходные изображения (B, H, W, C)
        self.images_no_glasses = torch.zeros((self.batch_size, self.H, self.W, self.C), device=self.device)
        self.images_with_glasses = torch.zeros_like(self.images_no_glasses)

        # Фильтр очков (B,)
        self.g_filtered = torch.ones(self.batch_size, device=self.device)

    # ------------------------------------------------------------------
    # Вспомогательные методы (подсветка, очки)
    # ------------------------------------------------------------------
    def _backlight_batch(self, t):
        phase = (t % self.T_frame) / self.T_frame
        start = self.backlight_phase / 100.0
        end = start + self.backlight_duration / 100.0
        in_strobe = (start <= phase) & (phase < end)
        wrap = (end > 1.0) & (phase < end - 1.0)
        in_window = in_strobe | wrap
        return torch.where(self.backlight_mode_constant.bool(),
                           torch.ones_like(in_window, dtype=torch.float32),
                           in_window.float())

    def _glasses_ideal_batch(self, t, glasses_time_offset):
        t_glasses = t - glasses_time_offset
        phase = (t_glasses % self.T_frame) / self.T_frame
        start = self.glasses_phase / 100.0
        end = start + self.glasses_duration / 100.0
        in_window = (start <= phase) & (phase < end)
        wrap = (end > 1.0) & (phase < end - 1.0)
        in_window = in_window | wrap
        return torch.where(self.glasses_enabled, in_window.float(),
                           torch.ones_like(in_window, dtype=torch.float32))

    def _get_tau_matrix_batch(self, old_frame, new_frame):
        """Возвращает tau формы (B, H, W, 1)."""
        gray_from = (0.299*old_frame[...,0] + 0.587*old_frame[...,1] + 0.114*old_frame[...,2]) * 255.0
        gray_to   = (0.299*new_frame[...,0] + 0.587*new_frame[...,1] + 0.114*new_frame[...,2]) * 255.0
        gray_from = torch.clamp(torch.round(gray_from), 0, 255).long()
        gray_to   = torch.clamp(torch.round(gray_to), 0, 255).long()

        if self.rise_batch is not None:
            idx = gray_from * 256 + gray_to  # (H,W)
            idx_flat = idx.view(-1)
            rise_flat = self.rise_batch.view(self.batch_size, -1)  # (B, 65536)
            fall_flat = self.fall_batch.view(self.batch_size, -1)
            rise_vals = rise_flat[:, idx_flat]  # (B, H*W)
            fall_vals = fall_flat[:, idx_flat]
            direction = gray_to >= gray_from
            dir_flat = direction.view(-1)
            tau_flat = torch.where(dir_flat, rise_vals, fall_vals)
            tau = tau_flat.view(self.batch_size, self.H, self.W, 1)
        else:
            tau_rise_b = self.tau_rise.view(-1,1,1,1)
            tau_fall_b = self.tau_fall.view(-1,1,1,1)
            direction = (new_frame[...,0] > old_frame[...,0]).unsqueeze(0).unsqueeze(-1)
            tau = torch.where(direction, tau_rise_b, tau_fall_b)
        return tau

    # ------------------------------------------------------------------
    # Логика синхронизации по маркеру (единая для батча)
    # ------------------------------------------------------------------
    def _init_sync_state(self):
        self.sync_state = 0.0
        self.photo_voltage = 0.0
        self.current_frame_idx = -1
        self.t_frame_start = -self.T_frame
        self.last_sync_time = -self.T_frame * 2
        self.glasses_time_offset = 0.0

    def _update_sync(self, t, backlight):
        """Обновляет состояние синхронизации (скалярные переменные)."""
        if self.sync_modes[0] != 'pattern':   # все одинаковы, проверяем первый
            return

        # Яркость маркера: pattern_value * backlight (backlight - скаляр яркости для батча?
        # Поскольку параметры синхронизации одинаковы, backlight тоже одинаков,
        # но метод _backlight_batch возвращает (B,), так что берём первое значение.
        b_val = backlight[0].item()  # синхронизация одна на всех, бэкграунд должен быть одинаков
        pattern_value = 1.0 if (self.current_frame_idx % 2 == 0) else 0.0
        pattern_brightness = pattern_value * b_val

        dt = self.dt
        alpha = dt / (self.photo_tau + dt) if self.photo_tau > 0 else 1.0
        noise = np.random.normal(0, self.photo_noise_std) if self.photo_noise_std > 0 else 0.0
        self.photo_voltage += alpha * (pattern_brightness + noise - self.photo_voltage)

        high = self.sync_threshold
        low = max(0.0, high - self.sync_hysteresis)

        new_state = self.sync_state
        if self.photo_voltage > high:
            new_state = 1.0
        elif self.photo_voltage < low:
            new_state = 0.0

        if not self.sync_state and new_state:
            if t - self.last_sync_time >= self.sync_min_period:
                # Ложные срабатывания / пропуски
                if np.random.random() < self.sync_false_negative_rate:
                    pass  # пропуск
                else:
                    if np.random.random() < self.sync_false_positive_rate:
                        # ложное срабатывание – всё равно фиксируем как реальное
                        pass
                    self.t_frame_start = t
                    if self.current_frame_idx < self.num_frames - 1:
                        self.current_frame_idx += 1
                    self.last_sync_time = t
                    ideal_k = int(round(t / self.T_frame))
                    self.glasses_time_offset = t - ideal_k * self.T_frame

        self.sync_state = new_state

    # ------------------------------------------------------------------
    # Основной цикл (фиксированный взгляд) с поддержкой pattern sync
    # ------------------------------------------------------------------
    def _run_fixed_with_sync(self):
        """Используется, если все синхропараметры одинаковы и есть pattern sync."""
        self._init_sync_state()
        self.images_no_glasses.zero_()
        self.images_with_glasses.zero_()
        self.g_filtered.fill_(1.0)

        prev_n = -1
        for i in range(self.Nt):
            t = self.t_cpu[i]

            # Вычисляем подсветку и идеальные очки (B,) – очки используют общий glasses_time_offset
            b_val = self._backlight_batch(t)                       # (B,)
            g_ideal = self._glasses_ideal_batch(t, self.glasses_time_offset)  # (B,)

            # Синхронизация по маркеру (единая)
            self._update_sync(t, b_val)

            # После синхронизации получаем номер текущего кадра (единый для всех, т.к. синхронизация одна)
            n = max(0, self.current_frame_idx)
            n = min(n, self.num_frames - 1)
            n_next = min(n + 1, self.num_frames - 1)

            if n != prev_n:
                old_frame = self.frames[n]
                new_frame = self.frames[n_next]
                tau = self._get_tau_matrix_batch(old_frame, new_frame)   # (B,H,W,1)
                prev_n = n

            # Развёртка с учётом t_frame_start (скаляр)
            t_update = self.t_frame_start + (self.y_grid / self.H) * self.T_scan  # (H,)
            dt_update = torch.clamp(t - t_update, min=0.0)                     # (H,)
            dt_update = dt_update.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)    # (1,H,1,1)
            mask_updated = (t >= t_update).float().unsqueeze(0).unsqueeze(-1)

            # Пиксель: p (B,H,W,C)
            alpha = 1.0 - torch.exp(-dt_update / tau)   # (B,H,W,1)
            p = old_frame.unsqueeze(0) + (new_frame.unsqueeze(0) - old_frame.unsqueeze(0)) * alpha * mask_updated

            # Фильтрация очков
            alpha_g = 1.0 - torch.exp(-self.dt / self.tau_glasses)   # (B,)
            self.g_filtered += alpha_g * (g_ideal - self.g_filtered)
            g = self.g_filtered.view(-1,1,1,1)

            b_val = b_val.view(-1,1,1,1)

            contrib = p * b_val * self.dt
            self.images_no_glasses += contrib
            self.images_with_glasses += contrib * g

        total_time = self.t_end - self.t_start
        self.images_no_glasses /= total_time
        self.images_with_glasses /= total_time

    def _run_fixed_no_sync(self):
        """Без pattern sync (только внутренний таймер)."""
        self.images_no_glasses.zero_()
        self.images_with_glasses.zero_()
        self.g_filtered.fill_(1.0)

        prev_n = -1
        for i in range(self.Nt):
            t = self.t_cpu[i]
            b_val = self._backlight_batch(t)                       # (B,)
            g_ideal = self._glasses_ideal_batch(t, 0.0)           # offset=0

            n = int(t // self.T_frame)
            n = min(n, self.num_frames - 1)
            n_next = min(n + 1, self.num_frames - 1)

            if n != prev_n:
                old_frame = self.frames[n]
                new_frame = self.frames[n_next]
                tau = self._get_tau_matrix_batch(old_frame, new_frame)
                prev_n = n

            t_frame_start = n * self.T_frame
            t_update = t_frame_start + (self.y_grid / self.H) * self.T_scan
            dt_update = torch.clamp(t - t_update, min=0.0)
            dt_update = dt_update.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mask_updated = (t >= t_update).float().unsqueeze(0).unsqueeze(-1)

            alpha = 1.0 - torch.exp(-dt_update / tau)
            p = old_frame.unsqueeze(0) + (new_frame.unsqueeze(0) - old_frame.unsqueeze(0)) * alpha * mask_updated

            alpha_g = 1.0 - torch.exp(-self.dt / self.tau_glasses)
            self.g_filtered += alpha_g * (g_ideal - self.g_filtered)
            g = self.g_filtered.view(-1,1,1,1)

            b_val = b_val.view(-1,1,1,1)

            contrib = p * b_val * self.dt
            self.images_no_glasses += contrib
            self.images_with_glasses += contrib * g

        total_time = self.t_end - self.t_start
        self.images_no_glasses /= total_time
        self.images_with_glasses /= total_time

    def _run_smooth_pursuit_no_sync(self):
        """Следящий взгляд без синхронизации по маркеру."""
        if self.obj_mask is None:
            self._run_fixed_no_sync()
            return

        # Центр объекта
        y_idx, x_idx = torch.where(self.obj_mask)
        X0 = float(x_idx.float().mean().item())
        Y0 = float(y_idx.float().mean().item())

        self.images_no_glasses.zero_()
        self.images_with_glasses.zero_()
        self.g_filtered.fill_(1.0)

        prev_n = -1

        for i in range(self.Nt):
            t = self.t_cpu[i]
            b_val = self._backlight_batch(t)  # (B,)
            g_ideal = self._glasses_ideal_batch(t, 0.0)  # offset = 0

            n = int(t // self.T_frame)
            n = min(n, self.num_frames - 1)
            n_next = min(n + 1, self.num_frames - 1)

            if n != prev_n:
                old_frame = self.frames[n]
                new_frame = self.frames[n_next]
                tau_full = self._get_tau_matrix_batch(old_frame, new_frame)  # (B,H,W,1)
                prev_n = n

            # Положение объекта на сетчатке
            frac = (t - n * self.T_frame) / self.T_frame
            Xobj = X0 + (n + frac) * self.speed
            Xsrc = self.Xr + (Xobj - X0)
            Ysrc = self.Yr

            wrap = self.params_list[0].wrap_around
            if wrap:
                Xsrc = Xsrc % self.W
                valid = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)
            else:
                valid = (Xsrc >= 0) & (Xsrc < self.W) & (Ysrc >= 0) & (Ysrc < self.H)

            Xsrc_int = torch.round(Xsrc).long().clamp(0, self.W - 1)
            Ysrc_int = torch.round(Ysrc).long().clamp(0, self.H - 1)

            # Развёртка: момент начала кадра по таймеру
            t_frame_start = n * self.T_frame
            t_update = t_frame_start + (Ysrc / self.H) * self.T_scan  # (H,W)
            dt_update = torch.clamp(t - t_update, min=0.0)  # (H,W)
            dt_update = dt_update.unsqueeze(0).unsqueeze(-1)  # (1,H,W,1)
            mask_updated = (t >= t_update).float().unsqueeze(0).unsqueeze(-1)

            # Значения из кадров (одинаковы для всех элементов батча)
            old_val = old_frame[Ysrc_int, Xsrc_int]  # (H,W,C)
            new_val = new_frame[Ysrc_int, Xsrc_int]

            # tau из батчевого тензора (B, H, W, 1)
            tau = tau_full[:, Ysrc_int, Xsrc_int]  # (B, H, W, 1)

            # Текущее состояние пикселя p: (B, H, W, C)
            alpha = 1.0 - torch.exp(-dt_update / tau)  # (B,H,W,1)
            p = old_val.unsqueeze(0) + (new_val.unsqueeze(0) - old_val.unsqueeze(0)) * alpha * mask_updated

            # Фильтрация очков
            alpha_g = 1.0 - torch.exp(-self.dt / self.tau_glasses)  # (B,)
            self.g_filtered += alpha_g * (g_ideal - self.g_filtered)
            g = self.g_filtered.view(-1, 1, 1, 1)

            b_val = b_val.view(-1, 1, 1, 1)

            # Вклад только для valid пикселей
            valid = valid.unsqueeze(0).unsqueeze(-1)  # (1,H,W,1)
            contrib = p * b_val * valid * self.dt
            self.images_no_glasses += contrib
            self.images_with_glasses += contrib * g

        total_time = self.t_end - self.t_start
        self.images_no_glasses /= total_time
        self.images_with_glasses /= total_time

    def _run_smooth_pursuit_with_sync(self):
        """Следящий взгляд с синхронизацией по маркеру."""
        if self.obj_mask is None:
            self._run_fixed_with_sync()
            return

        y_idx, x_idx = torch.where(self.obj_mask)
        X0 = float(x_idx.float().mean().item())
        Y0 = float(y_idx.float().mean().item())

        self._init_sync_state()
        self.images_no_glasses.zero_()
        self.images_with_glasses.zero_()
        self.g_filtered.fill_(1.0)

        prev_n = -1

        for i in range(self.Nt):
            t = self.t_cpu[i]

            # Подсветка и идеальные очки с учётом сдвига фазы
            b_val = self._backlight_batch(t)
            g_ideal = self._glasses_ideal_batch(t, self.glasses_time_offset)

            # Обновление синхронизации
            self._update_sync(t, b_val)

            n = max(0, self.current_frame_idx)
            n = min(n, self.num_frames - 1)
            n_next = min(n + 1, self.num_frames - 1)

            if n != prev_n:
                old_frame = self.frames[n]
                new_frame = self.frames[n_next]
                tau_full = self._get_tau_matrix_batch(old_frame, new_frame)
                prev_n = n

            frac = (t - self.t_frame_start) / self.T_frame
            Xobj = X0 + (n + frac) * self.speed
            Xsrc = self.Xr + (Xobj - X0)
            Ysrc = self.Yr

            wrap = self.params_list[0].wrap_around
            if wrap:
                Xsrc = Xsrc % self.W
                valid = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)
            else:
                valid = (Xsrc >= 0) & (Xsrc < self.W) & (Ysrc >= 0) & (Ysrc < self.H)

            Xsrc_int = torch.round(Xsrc).long().clamp(0, self.W - 1)
            Ysrc_int = torch.round(Ysrc).long().clamp(0, self.H - 1)

            # Развёртка от t_frame_start (из синхронизации)
            t_update = self.t_frame_start + (Ysrc / self.H) * self.T_scan
            dt_update = torch.clamp(t - t_update, min=0.0)
            dt_update = dt_update.unsqueeze(0).unsqueeze(-1)
            mask_updated = (t >= t_update).float().unsqueeze(0).unsqueeze(-1)

            old_val = old_frame[Ysrc_int, Xsrc_int]
            new_val = new_frame[Ysrc_int, Xsrc_int]
            tau = tau_full[:, Ysrc_int, Xsrc_int]

            alpha = 1.0 - torch.exp(-dt_update / tau)
            p = old_val.unsqueeze(0) + (new_val.unsqueeze(0) - old_val.unsqueeze(0)) * alpha * mask_updated

            alpha_g = 1.0 - torch.exp(-self.dt / self.tau_glasses)
            self.g_filtered += alpha_g * (g_ideal - self.g_filtered)
            g = self.g_filtered.view(-1, 1, 1, 1)

            b_val = b_val.view(-1, 1, 1, 1)
            valid = valid.unsqueeze(0).unsqueeze(-1)

            contrib = p * b_val * valid * self.dt
            self.images_no_glasses += contrib
            self.images_with_glasses += contrib * g

        total_time = self.t_end - self.t_start
        self.images_no_glasses /= total_time
        self.images_with_glasses /= total_time

    # ------------------------------------------------------------------
    # Публичный run – выбирает стратегию
    # ------------------------------------------------------------------
    def run(self):
        # Если pattern sync с разными параметрами -> последовательный GPUSimulator (как было)
        if any(m == 'pattern' for m in self.sync_modes) and not self.sync_params_equal:
            results_no, results_with = [], []
            for p in self.params_list:
                sim = GPUSimulator(p, [f.cpu().numpy() for f in self.frames],
                                   obj_mask=self.obj_mask.cpu().numpy() if self.obj_mask is not None else None,
                                   device=self.device)
                sim.run()
                results_no.append(sim.result_no_glasses)
                results_with.append(sim.result_with_glasses)
            return results_no, results_with

        # Выбор батчевой логики
        if self.tracking_mode == 'fixed':
            if self.sync_modes[0] == 'pattern':
                self._run_fixed_with_sync()
            else:
                self._run_fixed_no_sync()
        else:  # smooth_pursuit
            if self.sync_modes[0] == 'pattern':
                self._run_smooth_pursuit_with_sync()
            else:
                self._run_smooth_pursuit_no_sync()

        # Постобработка (размытие глаза, clip)
        results_no, results_with = [], []
        for b in range(self.batch_size):
            img_no = self.images_no_glasses[b].cpu().numpy()
            img_with = self.images_with_glasses[b].cpu().numpy()
            sigma = self.eye_sigma[b].item()
            if sigma > 0:
                img_no = cv2.GaussianBlur(img_no, (0, 0), sigmaX=sigma, sigmaY=sigma)
                img_with = cv2.GaussianBlur(img_with, (0, 0), sigmaX=sigma, sigmaY=sigma)
            results_no.append(np.clip(img_no, 0, 1))
            results_with.append(np.clip(img_with, 0, 1))
        return results_no, results_with

# ----------------------------------------------------------------------
# Вспомогательная функция эквализации гистограммы (в LAB)
# ----------------------------------------------------------------------
def equalize_lab(image):
    img_uint8 = (image * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return rgb_eq.astype(np.float32) / 255.0

# ----------------------------------------------------------------------
# GUI приложение
# ----------------------------------------------------------------------
class BlurinatorUltraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blurinator 9000 Ultra – расширенная версия")
        self.root.geometry("1500x950")

        self.params = SimParams()
        self.original_image = None
        self.obj_mask = None
        self.image_loaded = False
        self.device_choice = tk.StringVar(value='auto')

        # Переменные для чекбоксов
        self.show_carrier_var = tk.BooleanVar(value=self.params.show_carrier)
        self.show_sync_var = tk.BooleanVar(value=self.params.show_sync_pulses)
        self.show_second_var = tk.BooleanVar(value=self.params.show_second_channel)
        self.driver_ringing_var = tk.BooleanVar(value=self.params.driver_ringing)
        self.sync_pattern_var = tk.BooleanVar(value=self.params.sync_pattern_enabled)

        self.create_widgets()
        self.run_simulation()

    def create_widgets(self):
        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_panel)
        main_panel.add(left_frame, weight=2)

        right_frame = ttk.Frame(main_panel)
        main_panel.add(right_frame, weight=1)

        # Левая часть – изображения (без изменений)
        fig_left = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax_left = fig_left.subplots(1, 2)
        self.ax_left[0].set_title("Без очков")
        self.ax_left[1].set_title("С очками")
        self.canvas_left = FigureCanvasTkAgg(fig_left, master=left_frame)
        self.canvas_left.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Правая часть: верх (прокручиваемые параметры) и низ (график)
        top_frame = ttk.Frame(right_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(right_frame, height=300)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_frame.pack_propagate(False)  # фиксированная высота графика

        # Холст с прокруткой для параметров
        canvas = tk.Canvas(top_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(top_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ===== Все элементы управления переносятся в scrollable_frame =====
        ctrl_frame = ttk.LabelFrame(scrollable_frame, text="Параметры симуляции")
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        # --- Загрузка изображения и выделение объекта ---
        load_frame = ttk.Frame(ctrl_frame)
        load_frame.pack(fill=tk.X, pady=5)
        ttk.Button(load_frame, text="Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(load_frame, text="Сохранить результаты", command=self.save_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(load_frame, text="Выделить объект (клик)", command=self.select_object).pack(side=tk.LEFT, padx=2)
        ttk.Button(load_frame, text="Сбросить маску", command=self.clear_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(load_frame, text="Тестирование", command=self.open_test_dialog).pack(side=tk.LEFT, padx=2)
        self.img_info = tk.StringVar(value="Нет изображения")
        ttk.Label(load_frame, textvariable=self.img_info).pack(side=tk.LEFT, padx=5)

        # Выбор устройства
        device_frame = ttk.Frame(ctrl_frame)
        device_frame.pack(fill=tk.X, pady=5)
        ttk.Label(device_frame, text="Устройство:").pack(side=tk.LEFT)
        ttk.Radiobutton(device_frame, text="Авто", variable=self.device_choice,
                        value='auto').pack(side=tk.LEFT)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_choice,
                        value='cpu').pack(side=tk.LEFT)
        ttk.Radiobutton(device_frame, text="GPU (CUDA)", variable=self.device_choice,
                        value='cuda').pack(side=tk.LEFT)

        # Режим взгляда
        track_frame = ttk.Frame(ctrl_frame)
        track_frame.pack(fill=tk.X, pady=2)
        ttk.Label(track_frame, text="Режим взгляда:").pack(side=tk.LEFT)
        self.tracking_var = tk.StringVar(value='fixed')
        ttk.Radiobutton(track_frame, text="Фиксированный", variable=self.tracking_var,
                        value='fixed', command=self.update_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(track_frame, text="Следящий", variable=self.tracking_var,
                        value='smooth_pursuit', command=self.update_mode).pack(side=tk.LEFT)
        # ======== Синхронизация по паттерну (добавить в scrollable_frame) ========
        sync_pattern_frame = ttk.LabelFrame(scrollable_frame, text="Режим синхронизации")
        sync_pattern_frame.pack(fill=tk.X, padx=5, pady=5)

        self.sync_mode_var = tk.StringVar(value=self.params.sync_mode)
        ttk.Radiobutton(sync_pattern_frame, text="Внутренняя (по таймеру)",
                        variable=self.sync_mode_var, value='internal',
                        command=self.update_sync_mode).pack(anchor=tk.W)
        ttk.Radiobutton(sync_pattern_frame, text="По маркеру (pattern 10x10)",
                        variable=self.sync_mode_var, value='pattern',
                        command=self.update_sync_mode).pack(anchor=tk.W)

        photo_frame = ttk.Frame(sync_pattern_frame)
        photo_frame.pack(fill=tk.X, pady=2)
        self.create_slider(photo_frame, "Инерционность фотодиода (мкс)", 0.0, 100.0,
                           self.params.photo_tau * 1e6,
                           lambda v: self.update_param('photo_tau', float(v) * 1e-6),
                           resolution=1.0)
        self.create_slider(photo_frame, "Шум фотоприёмника (СКО)", 0.0, 0.2,
                           self.params.photo_noise_std,
                           lambda v: self.update_param('photo_noise_std', float(v)),
                           resolution=0.01)
        # ===================================================================

        # Эквализация
        self.equalize_var = tk.BooleanVar(value=self.params.equalize_hist)
        ttk.Checkbutton(ctrl_frame, text="Эквализация гистограммы (LAB)",
                        variable=self.equalize_var, command=self.update_mode).pack(anchor=tk.W, pady=2)

        # Таблицы времени отклика
        table_frame = ttk.LabelFrame(ctrl_frame, text="Таблицы времени отклика")
        table_frame.pack(fill=tk.X, pady=5)
        ttk.Button(table_frame, text="Загрузить rise/fall CSV",
                   command=self.load_response_tables).pack(pady=2)
        ttk.Button(table_frame, text="Показать таблицу",
                   command=self.show_response_table).pack(pady=2)
        self.table_status = tk.StringVar(value="Таблицы не загружены")
        ttk.Label(table_frame, textvariable=self.table_status).pack()
        self.use_table_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(table_frame, text="Использовать таблицу (вместо τ)",
                        variable=self.use_table_var, command=self.update_mode).pack()

        # --- Расширенные аппаратные параметры ---
        hw_adv_frame = ttk.LabelFrame(ctrl_frame, text="Аппаратные параметры (расширенные)")
        hw_adv_frame.pack(fill=tk.X, padx=5, pady=5)
        # Циклическое движение
        self.wrap_around_var = tk.BooleanVar(value=self.params.wrap_around)
        ttk.Checkbutton(hw_adv_frame, text="Циклическое движение (wrap-around)",
                        variable=self.wrap_around_var, command=self.update_mode).pack(anchor=tk.W, pady=2)

        # Синхромаркер
        self.sync_pattern_var = tk.BooleanVar(value=self.params.sync_pattern_enabled)
        ttk.Checkbutton(hw_adv_frame, text="Синхромаркер в углу (10×10)",
                        variable=self.sync_pattern_var, command=self.update_mode).pack(anchor=tk.W, pady=2)

        # Фотоприёмный тракт
        sync_frame = ttk.LabelFrame(hw_adv_frame, text="Фотоприёмник и синхронизация")
        sync_frame.pack(fill=tk.X, padx=5, pady=2)
        self.create_slider(sync_frame, "Порог компаратора (отн.)", 0.1, 0.9,
                           self.params.sync_threshold,
                           lambda v: self.update_param('sync_threshold', float(v)), resolution=0.01)
        self.create_slider(sync_frame, "Гистерезис", 0.0, 0.1,
                           self.params.sync_hysteresis,
                           lambda v: self.update_param('sync_hysteresis', float(v)), resolution=0.001)
        self.create_slider(sync_frame, "Вер. ложн. сраб. (0..0.1)", 0.0, 0.1,
                           self.params.sync_false_positive_rate,
                           lambda v: self.update_param('sync_false_positive_rate', float(v)), resolution=0.001)
        self.create_slider(sync_frame, "Вер. пропуска", 0.0, 0.1,
                           self.params.sync_false_negative_rate,
                           lambda v: self.update_param('sync_false_negative_rate', float(v)), resolution=0.001)
        self.create_slider(sync_frame, "Яркость вспышки (ам.)", 0.1, 1.0,
                           self.params.backlight_brightness,
                           lambda v: self.update_param('backlight_brightness', float(v)), resolution=0.01)

        # Драйверы затворов
        driver_frame = ttk.LabelFrame(hw_adv_frame, text="Драйверы затворов (MOSFET + трансформатор)")
        driver_frame.pack(fill=tk.X, padx=5, pady=2)
        self.create_slider(driver_frame, "Время нарастания (мкс)", 0.0, 50.0,
                           self.params.driver_rise_time,
                           lambda v: self.update_param('driver_rise_time', float(v)), resolution=1.0)
        self.driver_ringing_var = tk.BooleanVar(value=self.params.driver_ringing)
        ttk.Checkbutton(driver_frame, text="Паразитный звон", variable=self.driver_ringing_var,
                        command=lambda: self.update_param('driver_ringing', self.driver_ringing_var.get())
                        ).pack(anchor=tk.W)

        # Тактовый генератор
        mcu_frame = ttk.LabelFrame(hw_adv_frame, text="Тактовый генератор ATtiny85")
        mcu_frame.pack(fill=tk.X, padx=5, pady=2)
        self.create_slider(mcu_frame, "Частота МК (МГц)", 7.0, 9.0,
                           self.params.mcu_freq,
                           lambda v: self.update_param('mcu_freq', float(v)), resolution=0.1)

        # Паразитные связи
        paras_frame = ttk.LabelFrame(hw_adv_frame, text="Перекрёстные помехи")
        paras_frame.pack(fill=tk.X, padx=5, pady=2)
        self.create_slider(paras_frame, "Сдвиг каналов (нс)", 0, 100,
                           self.params.channel_skew,
                           lambda v: self.update_param('channel_skew', int(v)), resolution=1)
        self.show_second_var = tk.BooleanVar(value=self.params.show_second_channel)
        ttk.Checkbutton(paras_frame, text="Показать второй канал", variable=self.show_second_var,
                        command=lambda: self.update_param('show_second_channel', self.show_second_var.get())
                        ).pack(anchor=tk.W)
        self.create_slider(paras_frame, "Наводка на синхро (0..0.3)", 0.0, 0.3,
                           self.params.sync_crosstalk,
                           lambda v: self.update_param('sync_crosstalk', float(v)), resolution=0.01)

        # --- Стандартные слайдеры ---
        self.create_slider(ctrl_frame, "Частота кадров (fps):", 24, 240, self.params.fps,
                           lambda v: self.update_param('fps', int(float(v))), resolution=1)
        self.create_slider(ctrl_frame, "VBlank (%):", 0, 50, self.params.vblank_percent,
                           lambda v: self.update_param('vblank_percent', int(float(v))), resolution=1)
        self.create_slider(ctrl_frame, "Скорость (пикс/кадр):", 0.5, 20.0, self.params.speed,
                           lambda v: self.update_param('speed', float(v)), resolution=0.1)
        self.create_slider(ctrl_frame, "τ нарастания (мс):", 0.1, 20.0, self.params.tau_rise * 1000,
                           lambda v: self.update_param('tau_rise', float(v) / 1000.0), resolution=0.1)
        self.create_slider(ctrl_frame, "τ спада (мс):", 0.1, 20.0, self.params.tau_fall * 1000,
                           lambda v: self.update_param('tau_fall', float(v) / 1000.0), resolution=0.1)
        self.create_slider(ctrl_frame, "τ очков (мс):", 0.0, 10.0, self.params.tau_glasses * 1000,
                           lambda v: self.update_param('tau_glasses', float(v) / 1000.0), resolution=0.1)
        self.create_slider(ctrl_frame, "Размытие глаза σ (пикс):", 0.0, 5.0, self.params.eye_sigma,
                           lambda v: self.update_param('eye_sigma', float(v)), resolution=0.1)
        self.create_slider(ctrl_frame, "Длит. подсветки (%):", 1, 100, self.params.backlight_duration,
                           lambda v: self.update_param('backlight_duration', int(float(v))), resolution=1)
        self.create_slider(ctrl_frame, "Фаза подсветки (%):", 0, 100, self.params.backlight_phase,
                           lambda v: self.update_param('backlight_phase', int(float(v))), resolution=1)
        self.create_slider(ctrl_frame, "Длит. очков (%):", 1, 100, self.params.glasses_duration,
                           lambda v: self.update_param('glasses_duration', int(float(v))), resolution=1)
        self.create_slider(ctrl_frame, "Фаза очков (%):", 0, 100, self.params.glasses_phase,
                           lambda v: self.update_param('glasses_phase', int(float(v))), resolution=1)
        self.create_slider(ctrl_frame, "Кол-во кадров:", 2, 32, self.params.num_frames,
                           lambda v: self.update_param('num_frames', int(float(v))), resolution=1)

        # Режим подсветки
        mode_frame = ttk.Frame(ctrl_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mode_frame, text="Подсветка:").pack(side=tk.LEFT)
        self.backlight_var = tk.StringVar(value=self.params.backlight_mode)
        ttk.Radiobutton(mode_frame, text="Постоянная", variable=self.backlight_var,
                        value='constant', command=self.update_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Строб", variable=self.backlight_var,
                        value='strobe', command=self.update_mode).pack(side=tk.LEFT)

        # Включение очков
        self.glasses_var = tk.BooleanVar(value=self.params.glasses_enabled)
        ttk.Checkbutton(ctrl_frame, text="Очки включены", variable=self.glasses_var,
                        command=self.update_mode).pack(anchor=tk.W, pady=2)

        # Визуализация сигналов
        viz_frame = ttk.LabelFrame(ctrl_frame, text="Визуализация сигналов")
        viz_frame.pack(fill=tk.X, pady=5)
        self.show_carrier_var = tk.BooleanVar(value=self.params.show_carrier)
        ttk.Checkbutton(viz_frame, text="Показывать несущую затворов (500 Гц)",
                        variable=self.show_carrier_var,
                        command=self.update_mode).pack(anchor=tk.W)
        self.show_sync_var = tk.BooleanVar(value=self.params.show_sync_pulses)
        ttk.Checkbutton(viz_frame, text="Показывать синхроимпульсы",
                        variable=self.show_sync_var,
                        command=self.update_mode).pack(anchor=tk.W)

        ttk.Button(ctrl_frame, text="Обновить симуляцию",
                   command=self.run_simulation).pack(pady=5)

        # ===== График временных диаграмм (внизу) =====
        fig_right = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax_right = fig_right.subplots(1, 1)
        self.ax_right.set_title("Временные диаграммы (центр)")
        self.ax_right.set_xlabel("Время (с)")
        self.ax_right.set_ylabel("Уровень")
        self.canvas_right = FigureCanvasTkAgg(fig_right, master=bottom_frame)
        self.canvas_right.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Прокрутка колесом мыши
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # Для Linux/Mac при необходимости можно добавить <Button-4>/<Button-5>

    def create_slider(self, parent, label, from_, to, default, callback, resolution=1.0):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label, width=25).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=default)
        s = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var,
                      command=lambda v: callback(var.get()))
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        show_var = tk.StringVar()
        def update_show(*args):
            val = var.get()
            if resolution == 1:
                show_var.set(str(int(round(val))))
            else:
                show_var.set(f"{val:.1f}")
        var.trace_add('write', update_show)
        update_show()
        ttk.Label(frame, textvariable=show_var, width=8).pack(side=tk.LEFT)
        if not hasattr(self, 'slider_vars'):
            self.slider_vars = []
        self.slider_vars.append(var)

    def update_param(self, name, value):
        setattr(self.params, name, value)
        if name in ('fps', 'vblank_percent'):
            self.params.update_T_frame()

    def update_mode(self):
        self.params.backlight_mode = self.backlight_var.get()
        self.params.glasses_enabled = self.glasses_var.get()
        self.params.tracking_mode = self.tracking_var.get()
        self.params.equalize_hist = self.equalize_var.get()
        self.params.use_response_table = self.use_table_var.get()
        self.params.show_carrier = self.show_carrier_var.get()
        self.params.show_sync_pulses = self.show_sync_var.get()
        self.params.show_second_channel = self.show_second_var.get()
        self.params.driver_ringing = self.driver_ringing_var.get()
        self.params.sync_pattern_enabled = self.sync_pattern_var.get()
        self.params.wrap_around = self.wrap_around_var.get()
        self.params.sync_mode = self.sync_mode_var.get()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return
        try:
            img = Image.open(path).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0
            self.original_image = img_np
            self.image_loaded = True
            self.obj_mask = None
            self.img_info.set(f"{img.width} x {img.height} (цветное)")
            self.run_simulation()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def clear_mask(self):
        self.obj_mask = None
        self.img_info.set(self.img_info.get().split(' (')[0])
        self.run_simulation()

    def select_object(self):
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        if not SAM2_AVAILABLE:
            messagebox.showerror("Ошибка", "Библиотека SAM2 не установлена. Установите: pip install segment-anything-2")
            return

        if not hasattr(self, 'sam_predictor'):
            try:
                weights_path = os.path.join(os.path.dirname(__file__), "sam2_hiera_tiny.pt")
                if not os.path.exists(weights_path):
                    weights_path = filedialog.askopenfilename(
                        title="Выберите файл весов SAM2 (sam2_hiera_tiny.pt)",
                        filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")]
                    )
                    if not weights_path:
                        return
                sam2_model = build_sam2("sam2_hiera_t.yaml", weights_path,
                                        device='cuda' if torch.cuda.is_available() else 'cpu')
                self.sam_predictor = SAM2ImagePredictor(sam2_model)
                messagebox.showinfo("SAM2", "Модель SAM2 tiny загружена")
            except Exception as e:
                messagebox.showerror("Ошибка загрузки SAM2", str(e))
                return

        img_uint8 = (self.original_image * 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        self.sam_predictor.set_image(img_uint8)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_uint8)
        ax.set_title("Кликните на объект для сегментации.\nEnter – принять текущую маску, Esc – отмена, Backspace – сбросить точки.")
        ax.axis('off')

        input_points = []
        input_labels = []
        current_mask = None

        def update_mask():
            nonlocal current_mask
            if not input_points:
                return
            points = np.array(input_points)
            labels = np.array(input_labels)
            with torch.no_grad():
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )
            best_idx = np.argmax(scores)
            mask_data = masks[best_idx]
            if isinstance(mask_data, torch.Tensor):
                mask_data = mask_data.cpu().numpy()
            if mask_data.ndim == 3:
                mask_data = mask_data.squeeze(0)
            current_mask = mask_data.astype(bool)
            redraw()

        def redraw():
            ax.clear()
            ax.imshow(img_uint8)
            for pt, label in zip(input_points, input_labels):
                color = 'lime' if label == 1 else 'red'
                ax.plot(pt[0], pt[1], 'o', color=color, markersize=8, markeredgecolor='white')
            if current_mask is not None:
                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                overlay[current_mask] = [0, 255, 0, 150]
                ax.imshow(overlay)
                contours, _ = cv2.findContours(current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], 'r', linewidth=2)
            ax.axis('off')
            fig.canvas.draw()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            x, y = int(event.xdata), int(event.ydata)
            if event.button == 1:
                input_points.append([x, y])
                input_labels.append(1)
            elif event.button == 3:
                input_points.append([x, y])
                input_labels.append(0)
            update_mask()

        def on_key(event):
            nonlocal current_mask
            if event.key == 'enter':
                if current_mask is not None:
                    self.obj_mask = current_mask
                plt.close(fig)
            elif event.key == 'escape':
                plt.close(fig)
            elif event.key == 'backspace':
                if input_points:
                    input_points.pop()
                    input_labels.pop()
                    if input_points:
                        update_mask()
                    else:
                        current_mask = None
                        redraw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        if self.obj_mask is not None:
            self.run_simulation()

    def load_response_tables(self):
        from tkinter import simpledialog
        rise_path = filedialog.askopenfilename(title="Выберите CSV-файл с временами нарастания (rise)",
                                               filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not rise_path:
            return
        fall_path = filedialog.askopenfilename(title="Выберите CSV-файл с временами спада (fall)",
                                               filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not fall_path:
            return
        method = simpledialog.askstring("Метод интерполяции",
                                        "Выберите метод:\n  linear - линейная\n  cubic - кубическая\n  rbf - радиальные базисные функции\n  overdrive - модель Overdrive\n  none - без интерполяции",
                                        initialvalue="overdrive")
        if not method:
            return
        method = method.strip().lower()
        if method not in ['linear', 'cubic', 'rbf', 'overdrive', 'none']:
            messagebox.showerror("Ошибка", "Неизвестный метод интерполяции")
            return
        try:
            self.params.response_table = ResponseTimeTable()
            self.params.response_table.load_from_csv(rise_path, fall_path, interp_method=method)
            self.table_status.set(f"Таблицы загружены (метод: {method})")
            self.params.use_response_table = self.use_table_var.get()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить таблицы:\n{e}")
            self.table_status.set("Ошибка загрузки")

    def show_response_table(self):
        if self.params.response_table is None or not self.params.response_table.loaded:
            messagebox.showwarning("Предупреждение", "Сначала загрузите таблицы времени отклика")
            return
        try:
            fig = self.params.response_table.plot_matrices(
                title=f"Таблицы времени отклика (метод: {self.params.response_table.interp_method})"
            )
            top = tk.Toplevel(self.root)
            top.title("Визуализация таблиц времени отклика")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, top)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось отобразить таблицы:\n{e}")

    def generate_frames(self):
        if self.image_loaded and self.original_image is not None:
            H, W, C = self.original_image.shape
            speed = self.params.speed
            frames = []
            shift_total = 0
            for k in range(self.params.num_frames):
                if k == 0:
                    frame = self.original_image.copy()
                else:
                    shift_total += speed
                    shift = int(round(shift_total))
                    frame = np.roll(self.original_image, shift=shift, axis=1)
                frames.append(frame)
            obj_mask = self.obj_mask  # теперь obj_mask определена (может быть None)
        else:
            H, W = 128, 256
            grad = np.linspace(0, 1, W).reshape(1, -1, 1)
            background = np.tile(grad, (H, 1, 3))
            frames = []
            size = 20
            y0 = H // 2 - size // 2
            y1 = y0 + size
            x_center0 = W // 4
            for k in range(self.params.num_frames):
                frame = background.copy()
                x_center = x_center0 + int(round(k * self.params.speed))
                x_center = x_center % W
                x_start = max(0, x_center - size // 2)
                x_end = min(W, x_center + size // 2)
                if x_start < x_end:
                    frame[y0:y1, x_start:x_end, 0] = 1.0
                    frame[y0:y1, x_start:x_end, 1:] = 0.0
                frames.append(frame)
            obj_mask = np.zeros((H, W), dtype=bool)
            x_start0 = max(0, x_center0 - size // 2)
            x_end0 = min(W, x_center0 + size // 2)
            obj_mask[y0:y1, x_start0:x_end0] = True

        # Наложение синхромаркера
        if self.params.sync_pattern_enabled:
            for k, frame in enumerate(frames):
                pattern_value = 1.0 if (k % 2 == 0) else 0.0
                frame[:10, :10, :] = pattern_value

        return frames, obj_mask

    def save_results(self):
        if not hasattr(self, 'sim') or self.sim is None:
            messagebox.showwarning("Предупреждение", "Сначала выполните симуляцию")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Сохранить как (базовое имя)"
        )
        if not file_path:
            return

        base, ext = os.path.splitext(file_path)
        if ext.lower() != '.png':
            base = file_path
        path_no = base + "_no_glasses.png"
        path_with = base + "_with_glasses.png"

        img_no = self.sim.result_no_glasses
        img_with = self.sim.result_with_glasses

        if self.params.equalize_hist:
            img_no = equalize_lab(img_no)
            img_with = equalize_lab(img_with)

        img_no_uint8 = (np.clip(img_no, 0, 1) * 255).astype(np.uint8)
        img_with_uint8 = (np.clip(img_with, 0, 1) * 255).astype(np.uint8)

        try:
            Image.fromarray(img_no_uint8).save(path_no)
            Image.fromarray(img_with_uint8).save(path_with)
            messagebox.showinfo("Сохранение", f"Файлы сохранены:\n{path_no}\n{path_with}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файлы:\n{e}")

    def open_test_dialog(self):
        if not self.image_loaded and self.original_image is None:
            if not messagebox.askyesno("Внимание",
                                       "Изображение не загружено. Будет использовано синтетическое тестовое изображение. Продолжить?"):
                return

        dialog = tk.Toplevel(self.root)
        dialog.title("Настройка автоматического тестирования")
        dialog.geometry("550x700")
        dialog.transient(self.root)
        dialog.grab_set()

        param_widgets = {}
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Выберите параметры и диапазоны для тестирования",
                  font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=5)

        canvas = tk.Canvas(main_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ============================================================
        # Расширенный список ВСЕХ параметров симуляции
        # ============================================================
        testable_params = [
            # Базовые
            ('fps', 'Частота кадров (fps)', 'int', 24, 240, 60),
            ('vblank_percent', 'Вертикальное гашение (%)', 'int', 0, 50, 0),
            ('speed', 'Скорость (пикс/кадр)', 'float', 0.5, 20.0, 5.0),
            ('tau_rise', 'τ нарастания (мс)', 'float', 0.1, 20.0, 5.0),
            ('tau_fall', 'τ спада (мс)', 'float', 0.1, 20.0, 5.0),
            ('tau_glasses', 'τ очков (мс)', 'float', 0.0, 10.0, 2.0),
            ('eye_sigma', 'Размытие глаза σ (пикс)', 'float', 0.0, 5.0, 0.0),
            ('backlight_duration', 'Длит. подсветки (%)', 'int', 1, 100, 100),
            ('backlight_phase', 'Фаза подсветки (%)', 'int', 0, 100, 0),
            ('glasses_duration', 'Длит. очков (%)', 'int', 1, 100, 20),
            ('glasses_phase', 'Фаза очков (%)', 'int', 0, 100, 0),
            ('num_frames', 'Кол-во кадров', 'int', 2, 32, 4),

            # Подсветка
            ('backlight_brightness', 'Яркость вспышки (0.1-1.0)', 'float', 0.1, 1.0, 1.0),
            ('backlight_mode', 'Режим подсветки (constant/strobe)', 'str', None, None, 'constant'),

            # Синхронизация
            ('sync_mode', 'Режим синхронизации (internal/pattern)', 'str', None, None, 'internal'),
            ('sync_pattern_enabled', 'Синхромаркер (0-нет, 1-да)', 'int', 0, 1, 1),
            ('sync_threshold', 'Порог компаратора', 'float', 0.1, 0.9, 0.5),
            ('sync_hysteresis', 'Гистерезис', 'float', 0.0, 0.1, 0.02),
            ('sync_false_positive_rate', 'Ложные срабатывания', 'float', 0.0, 0.1, 0.0),
            ('sync_false_negative_rate', 'Пропуски импульсов', 'float', 0.0, 0.1, 0.0),
            ('photo_tau', 'Инерционность фотодиода (с)', 'float', 0.0, 100e-6, 1e-6),
            ('photo_noise_std', 'Шум фотоприёмника (СКО)', 'float', 0.0, 0.2, 0.0),

            # Драйверы затворов
            ('driver_rise_time', 'Время нарастания (мкс)', 'float', 0.0, 50.0, 0.0),
            ('driver_ringing', 'Паразитный звон (0/1)', 'int', 0, 1, 0),
            ('mcu_freq', 'Частота МК (МГц)', 'float', 7.0, 9.0, 8.0),

            # Паразитные связи
            ('channel_skew', 'Сдвиг каналов (нс)', 'int', 0, 100, 0),
            ('sync_crosstalk', 'Наводка на синхро (0-0.3)', 'float', 0.0, 0.3, 0.0),

            # Очки и взгляд
            ('glasses_enabled', 'Очки включены (0/1)', 'int', 0, 1, 1),
            ('tracking_mode', 'Режим взгляда (fixed/smooth_pursuit)', 'str', None, None, 'fixed'),
            ('wrap_around', 'Циклическое движение (0/1)', 'int', 0, 1, 0),
        ]

        row = 0
        for param_name, label, ptype, pmin, pmax, pdef in testable_params:
            frame = ttk.Frame(scrollable_frame)
            frame.grid(row=row, column=0, sticky="ew", pady=2, padx=5)
            scrollable_frame.columnconfigure(0, weight=1)

            ttk.Label(frame, text=label, width=35, anchor='w').pack(side=tk.LEFT)

            include_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(frame, variable=include_var).pack(side=tk.LEFT)

            # Для строковых параметров значение по умолчанию – строка
            if ptype == 'str':
                def_str = str(pdef) if pdef is not None else ''
            else:
                def_str = f"{pmin},{pmax},{pdef}" if pmin is not None else str(pdef)

            values_var = tk.StringVar(value=def_str)
            entry = ttk.Entry(frame, textvariable=values_var, width=25, state='disabled')
            entry.pack(side=tk.LEFT, padx=5)

            # Кнопка для удобного задания диапазона (только для числовых)
            if ptype in ('int', 'float'):
                def make_range_cmd(vv=values_var, ee=entry, pmin=pmin, pmax=pmax, pdef=pdef, ptype=ptype):
                    return lambda: self.ask_range(vv, ee, pmin, pmax, pdef, ptype)

                ttk.Button(frame, text="...", width=3, command=make_range_cmd()).pack(side=tk.LEFT)
            else:
                # Для строковых просто оставляем поле ручного ввода
                pass

            include_var.trace_add('write', lambda *a, e=entry, iv=include_var:
            e.config(state='normal' if iv.get() else 'disabled'))

            param_widgets[param_name] = {
                'include': include_var,
                'values': values_var,
                'entry': entry,
                'type': ptype,
                'min': pmin,
                'max': pmax
            }
            row += 1

        # Типы графиков
        plot_frame = ttk.LabelFrame(scrollable_frame, text="Графики для отчёта")
        plot_frame.grid(row=row, column=0, sticky="ew", pady=5, padx=5)
        row += 1
        self.plot_vars = {}
        plot_options = [
            ('line', 'Линейные графики'),
            ('hist', 'Гистограммы'),
            ('box', 'Бокс-плоты'),
            ('heatmap', 'Тепловые карты'),
            ('pairplot', 'Pairplot (seaborn)'),
            ('3d', '3D-поверхности')
        ]
        for i, (ptype, pdesc) in enumerate(plot_options):
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(plot_frame, text=pdesc, variable=var).grid(row=i // 2, column=i % 2, sticky="w", padx=10)
            self.plot_vars[ptype] = var

        ttk.Label(scrollable_frame, text="Фиксированные параметры (не выбранные выше) берутся из текущего GUI.",
                  font=('TkDefaultFont', 9, 'italic')).grid(row=row, column=0, sticky="w", pady=(10, 0))
        row += 1
        fixed_info = ttk.Label(scrollable_frame, text="")
        fixed_info.grid(row=row, column=0, sticky="w", pady=2)
        row += 1

        # Устройство
        device_frame = ttk.Frame(scrollable_frame)
        device_frame.grid(row=row, column=0, sticky="ew", pady=5)
        ttk.Label(device_frame, text="Устройство для теста:").pack(side=tk.LEFT)
        test_device_var = tk.StringVar(value='auto')
        ttk.Radiobutton(device_frame, text="Авто", variable=test_device_var, value='auto').pack(side=tk.LEFT)
        ttk.Radiobutton(device_frame, text="CPU", variable=test_device_var, value='cpu').pack(side=tk.LEFT)
        ttk.Radiobutton(device_frame, text="GPU", variable=test_device_var, value='cuda').pack(side=tk.LEFT)
        row += 1

        # Прогресс
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=5)
        status_var = tk.StringVar(value="Готов")
        status_label = ttk.Label(main_frame, textvariable=status_var)
        status_label.pack()

        clahe_frame = ttk.Frame(main_frame)
        clahe_frame.pack(fill=tk.X, pady=5)
        self.clahe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(clahe_frame, text="Локальное адаптивное выравнивание (CLAHE)",
                        variable=self.clahe_var).pack(side=tk.LEFT)
        self.blur_mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(clahe_frame, text="Изолировать по маске для оценки размытия",
                        variable=self.blur_mask_var).pack(side=tk.LEFT)

        # Кнопки
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Запустить тест",
                   command=lambda: self.run_test_from_dialog(
                       param_widgets, test_device_var, progress_var, status_var, dialog
                   )).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Закрыть", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def ask_range(self, values_var, entry, pmin, pmax, pdef, ptype):
        """Открывает диалог для ввода списка значений или диапазона с шагом."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Задать значения параметра")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Введите значения через запятую или диапазон в формате:",
                  wraplength=300).pack(pady=5)
        ttk.Label(dialog, text="start:stop:step   или   start,stop,step",
                  font=('TkFixedFont', 9)).pack()

        current = values_var.get()
        entry_var = tk.StringVar(value=current)
        ttk.Entry(dialog, textvariable=entry_var, width=40).pack(pady=10)

        def apply():
            val_str = entry_var.get().strip()
            # Простая проверка
            if val_str:
                # Парсим: либо числа через запятую, либо диапазон
                try:
                    if ':' in val_str:
                        parts = val_str.split(':')
                        start = float(parts[0])
                        stop = float(parts[1])
                        step = float(parts[2]) if len(parts) > 2 else 1.0
                        values = np.arange(start, stop + step/2, step)
                    elif ',' in val_str:
                        values = [float(x.strip()) for x in val_str.split(',') if x.strip()]
                    else:
                        values = [float(val_str)]
                    # Приводим к нужному типу
                    if ptype == 'int':
                        values = [int(round(v)) for v in values]
                    # Формируем строку
                    new_str = ','.join(str(v) for v in values)
                    values_var.set(new_str)
                    entry.config(state='normal')
                    entry.delete(0, tk.END)
                    entry.insert(0, new_str)
                    entry.config(state='disabled')
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Некорректный ввод: {e}")

        ttk.Button(dialog, text="Применить", command=apply).pack(pady=5)
        ttk.Button(dialog, text="Отмена", command=dialog.destroy).pack()

    def run_test_from_dialog(self, param_widgets, test_device_var, progress_var, status_var, dialog):
        param_ranges = {}
        for pname, wdata in param_widgets.items():
            if wdata['include'].get():
                val_str = wdata['values'].get()
                try:
                    parts = [x.strip() for x in val_str.split(',') if x.strip()]
                    ptype = wdata['type']
                    if ptype == 'int':
                        vals = [int(round(float(x))) for x in parts]
                    elif ptype == 'float':
                        vals = [float(x) for x in parts]
                    elif ptype == 'str':
                        vals = parts  # оставляем как строки
                    else:
                        vals = parts
                    param_ranges[pname] = vals
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Неверный формат значений для параметра {pname}: {e}")
                    return

        if not param_ranges:
            messagebox.showwarning("Предупреждение", "Не выбрано ни одного параметра для варьирования.")
            return

        # Определяем устройство
        dev_choice = test_device_var.get()
        if dev_choice == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif dev_choice == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                messagebox.showwarning("Внимание", "CUDA не доступна, используется CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        # Получаем текущие кадры
        frames, obj_mask = self.generate_frames()

        apply_clahe = self.clahe_var.get()
        blur_use_mask = self.blur_mask_var.get()

        # Создаём тестер
        tester = BlurTester(self.params, frames, obj_mask, device, batch_size=32,
                            equalize_clahe=apply_clahe,
                            blur_use_mask=blur_use_mask)
        selected_plots = [ptype for ptype, var in self.plot_vars.items() if var.get()]
        self.selected_plots = selected_plots  # сохраняем для использования в test_finished

        # Функция обратного вызова для обновления прогресса
        def progress_callback(current, total, last_result):
            progress_var.set(current / total * 100)
            status_var.set(f"Выполнено {current} из {total} тестов. Последнее размытие: {last_result['blur_no_glasses']:.2f}")
            dialog.update_idletasks()

        # Запускаем тест в отдельном потоке, чтобы GUI не зависал
        import threading

        def test_thread():
            try:
                results = tester.run_test(param_ranges, progress_callback=progress_callback)
                # После завершения
                dialog.after(0, lambda: self.test_finished(results, tester, status_var, dialog))
            except Exception as e:
                dialog.after(0, lambda: messagebox.showerror("Ошибка", f"Тестирование прервано: {e}"))
                dialog.after(0, lambda: status_var.set("Ошибка"))

        threading.Thread(target=test_thread, daemon=True).start()

    def test_finished(self, results, tester, status_var, dialog):
        """Вызывается после завершения тестирования."""
        status_var.set("Тестирование завершено. Сохранение отчёта...")
        # Сохраняем отчёт
        save_dir = filedialog.askdirectory(title="Выберите папку для сохранения отчёта")
        if save_dir:
            try:
                csv_path = tester.generate_report(output_dir=save_dir, plot_types=self.selected_plots)
                messagebox.showinfo("Готово", f"Отчёт сохранён в:\n{csv_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить отчёт: {e}")
        status_var.set("Готово")
        dialog.destroy()

    def update_sync_mode(self):
        self.params.sync_mode = self.sync_mode_var.get()
        # при переключении на pattern делаем неактивными некоторые внутренние параметры,
        # если нужно. Пока просто обновляем симуляцию.
        self.run_simulation()

    def run_simulation(self):
        frames, obj_mask = self.generate_frames()

        if self.device_choice.get() == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.device_choice.get() == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                messagebox.showwarning("Внимание", "CUDA не доступна, используется CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        self.params.tracking_mode = self.tracking_var.get()
        self.params.equalize_hist = self.equalize_var.get()
        self.params.use_response_table = self.use_table_var.get()

        self.sim = GPUSimulator(self.params, frames, obj_mask=obj_mask, device=device)
        self.sim.run()

        img_no = self.sim.result_no_glasses
        img_with = self.sim.result_with_glasses

        if self.params.equalize_hist:
            img_no = equalize_lab(img_no)
            img_with = equalize_lab(img_with)

        # Левая панель — изображения
        self.ax_left[0].clear()
        self.ax_left[0].imshow(img_no, vmin=0, vmax=1)
        self.ax_left[0].set_title("Без очков")
        self.ax_left[0].axis('off')

        self.ax_left[1].clear()
        self.ax_left[1].imshow(img_with, vmin=0, vmax=1)
        self.ax_left[1].set_title("С очками")
        self.ax_left[1].axis('off')

        self.canvas_left.draw()

        # Правый график
        self.ax_right.clear()
        t = self.sim.t_cpu

        self.ax_right.plot(t, self.sim.backlight_history, label='Подсветка', lw=2)
        self.ax_right.plot(t, self.sim.glasses_history, label='Очки (канал 1)', lw=2)

        if hasattr(self.sim, 'glasses_history_second') and self.params.show_second_channel:
            self.ax_right.plot(t, self.sim.glasses_history_second, label='Очки (канал 2)', lw=1, linestyle=':')

        avg_pixel = self.sim.pixel_history.mean(axis=1)
        self.ax_right.plot(t, avg_pixel, label='Яркость пикселя', lw=1.5, alpha=0.8)

        if self.params.show_sync_pulses:
            if self.params.sync_mode == 'pattern':
                if self.sim.current_frame_idx >= 0:
                    self.ax_right.axvline(x=self.sim.t_frame_start, color='gray', linestyle='--', alpha=0.7,
                                          label='Sync')
            else:
                for i in range(self.params.num_frames):
                    self.ax_right.axvline(x=i * self.params.T_frame, color='gray', linestyle='--', alpha=0.4)

        if self.params.show_carrier and hasattr(self.sim, 'cached_carrier_no_glitch'):
            modulated = self.sim.glasses_history * self.sim.cached_carrier_no_glitch
            self.ax_right.plot(t, modulated, label='Несущая 500Гц', lw=1, alpha=0.6)

        self.ax_right.set_title("Временные диаграммы")
        self.ax_right.set_xlabel("Время (с)")
        self.ax_right.set_ylabel("Уровень")
        self.ax_right.legend(loc='upper right')
        self.ax_right.grid(True)
        self.canvas_right.draw()

    # ----------------------------------------------------------------------
    # Запуск
    # ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BlurinatorUltraApp(root)
    root.mainloop()