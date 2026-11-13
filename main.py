#!/usr/bin/env python3
"""
TRC Signal Plotter — PyQtGraph edition.

GUI для построения сигналов из .trc (PCAN) файлов по описанию .dbc.
Заголовок графика, подписи осей и подписи в легенде редактируются
непосредственно в окне графика. Фон графика и пределы осей настраиваются
в окне графика. Есть дерево кривых с чекбоксами видимости и возможность
перемещать кривую на другую ось.

Правая область окна графика содержит две колонки:
  • «Настройки графика»        — сетка, легенда, фон, пределы осей;
  • «Редактирование отображения» — заголовок, подписи, стиль, оси кривых.
Обе колонки можно скрывать независимо друг от друга.

Сетка по Y включается отдельно для каждой оси Y.

Масштабирование мышью (базовый режим — увеличение области, mouse mode = 1):
ЛКМ-протяжка (область), Shift+ЛКМ (только X), Ctrl+ЛКМ (только Y),
ПКМ — панорамирование, СКМ — авто-пределы осей.

Время интерпретируется как миллисекунды в TRC и переводится в секунды.

Зависимости: PyQt5, pyqtgraph, numpy.
Опционально: scipy (экспорт .mat).
"""

import sys
import os
import re
import json
from collections import defaultdict

import numpy as np

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPointF, QRectF,
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QTreeWidget,
    QTreeWidgetItem, QSplitter, QTabWidget, QPlainTextEdit, QFileDialog,
    QColorDialog, QMessageBox, QProgressBar, QDialog, QGroupBox,
    QFormLayout, QGridLayout, QAbstractItemView, QFrame,
)

import pyqtgraph as pg

try:
    from scipy.io import savemat as scipy_savemat
    HAS_SCIPY = True
except ImportError:
    scipy_savemat = None
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
DEFAULT_COLOR = (0.0, 0.4470, 0.7410)
MATLAB_LINES = [
    (0.0000, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
]
PREFS_PATH = os.path.join(os.path.expanduser('~'), '.trc_plotter_prefs.json')

LINE_STYLES = {
    '-': Qt.SolidLine,
    '--': Qt.DashLine,
    ':': Qt.DotLine,
    '-.': Qt.DashDotLine,
}

MARKERS = ['none', 'o', 's', 't', 'd', '+', 'x', 'p', 'h', 'star']

# Время в TRC хранится в миллисекундах
TIME_SCALE_TO_SECONDS = 1.0 / 1000.0


# ---------------------------------------------------------------------------
# Разбор DBC
# ---------------------------------------------------------------------------
def parse_dbc(filename):
    messages = []
    cur_msg = None
    re_bo = re.compile(r'^BO_ (\d+) (\w+)\s*:\s*(\d+)')
    re_sg = re.compile(
        r'^\s*SG_\s+(\w+)\s*:\s*(\d+)\|(\d+)@(\d)([+-])\s*'
        r'\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*\[[^\]]*\]\s*"([^"]*)"'
    )
    with open(filename, 'r', encoding='latin-1', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n\r')
            m = re_bo.match(line)
            if m:
                id_raw = int(m.group(1))
                cur_msg = {
                    'id': id_raw & 0x1FFFFFFF,
                    'name': m.group(2),
                    'dlc': int(m.group(3)),
                    'signals': [],
                }
                messages.append(cur_msg)
                continue
            m = re_sg.match(line)
            if m and cur_msg is not None:
                cur_msg['signals'].append({
                    'name': m.group(1),
                    'start': int(m.group(2)),
                    'len': int(m.group(3)),
                    'byteorder': m.group(4),
                    'sign': m.group(5),
                    'scale': float(m.group(6)),
                    'offset': float(m.group(7)),
                    'unit': m.group(8),
                })
    return {'messages': messages}


# ---------------------------------------------------------------------------
# Разбор TRC
# ---------------------------------------------------------------------------
def parse_can_log(filename, progress_cb=None, progress_value_cb=None):
    def _p(msg):
        if progress_cb:
            progress_cb(msg)

    def _pv(val):
        if progress_value_cb:
            try:
                progress_value_cb(val)
            except Exception:
                pass

    _pv(0.0)
    messages = []
    try:
        size = os.path.getsize(filename)
    except OSError:
        size = 0
    _p(f'Открыт файл: {filename}')
    _p(f'Размер файла: {size} байт')

    total_lines = 0
    try:
        last_chunk = b''
        with open(filename, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                total_lines += chunk.count(b'\n')
                last_chunk = chunk
        if last_chunk and not last_chunk.endswith(b'\n'):
            total_lines += 1
    except OSError as e:
        _pv(100.0)
        raise RuntimeError(f'Не удалось открыть файл: {e}')

    _p(f'Всего строк в файле: {total_lines}')

    file_version = None
    try:
        with open(filename, 'rb') as f:
            for raw_bytes in f:
                line = raw_bytes.decode('latin-1', errors='replace')
                if line.startswith('$FILEVERSION'):
                    m = re.search(r'=\s*(\d+)\.(\d+)', line)
                    if m:
                        file_version = (int(m.group(1)), int(m.group(2)))
                    break
                s = line.strip()
                if not s or (not s.startswith(';') and not s.startswith('$')):
                    break
    except OSError as e:
        _pv(100.0)
        raise RuntimeError(f'Не удалось открыть файл: {e}')

    if file_version:
        _p(f'Формат TRC: версия {file_version[0]}.{file_version[1]}')
    else:
        _p('Формат TRC: версия в заголовке не указана (автоопределение).')

    line_count = 0
    bytes_read = 0
    last_percent = 0

    def _report_progress(force=False):
        nonlocal last_percent
        if size > 0:
            pct = int(100 * bytes_read / size)
        elif total_lines > 0:
            pct = int(100 * line_count / total_lines)
        else:
            pct = 0 if not force else 100
        if pct > 100:
            pct = 100
        if pct < 0:
            pct = 0
        if force or pct != last_percent:
            last_percent = pct
            _p(f'Процент прочтения {pct:3d}%. '
               f'Строк прочитано {line_count} из {total_lines}, '
               f'сообщений найдено {len(messages)}')
            _pv(float(pct))

    with open(filename, 'rb') as f:
        for raw_bytes in f:
            line_count += 1
            bytes_read += len(raw_bytes)
            line = raw_bytes.decode('latin-1', errors='replace').rstrip('\n\r')

            _report_progress()

            if not line:
                continue
            if line[0] in ';-$':
                continue
            tokens = line.strip().split()
            n = len(tokens)
            if n < 5:
                continue

            if n >= 7 and tokens[2] == 'DT' and tokens[4] in ('Rx', 'Tx'):
                has_num = True
                id_idx, dlc_idx, data_idx = 3, 5, 6
            elif n >= 6 and tokens[2] in ('Rx', 'Tx'):
                has_num = True
                id_idx, dlc_idx, data_idx = 3, 4, 5
            elif tokens[1] in ('Rx', 'Tx'):
                has_num = False
                id_idx, dlc_idx, data_idx = 2, 3, 4
            else:
                continue

            try:
                if has_num:
                    num_str = tokens[0].rstrip(')')
                    msg_num = float(num_str)
                    t = float(tokens[1])
                else:
                    msg_num = len(messages) + 1
                    t = float(tokens[0])
                id_str = tokens[id_idx]
                dlc = int(float(tokens[dlc_idx]))
            except (ValueError, IndexError):
                continue

            if dlc < 0:
                dlc = 0
            if dlc > 8:
                dlc = 8

            id_clean = id_str.lower()
            if id_clean.startswith('0x'):
                id_clean = id_clean[2:]
            try:
                id_val = int(id_clean, 16)
            except ValueError:
                continue

            data = np.zeros(8, dtype=np.uint8)
            n_avail = min(n - data_idx, 8)
            ok = True
            for k in range(n_avail):
                try:
                    data[k] = int(tokens[data_idx + k], 16) & 0xFF
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue

            messages.append({
                'num': msg_num, 'time': t * TIME_SCALE_TO_SECONDS,
                'id': id_str, 'id_val': id_val, 'data': data,
            })

    _report_progress(force=True)
    _pv(100.0)
    _p(f'Разбор завершён: {len(messages)} сообщений '
       f'(строк прочитано {line_count} из {total_lines}).')
    return messages


def extract_intel(data, start_bit, length, is_signed, scale, offset):
    raw = 0
    for i in range(length):
        bp = start_bit + i
        byte_idx = bp // 8
        bit_idx = bp % 8
        if byte_idx >= len(data):
            return float('nan')
        b = (int(data[byte_idx]) >> bit_idx) & 1
        raw |= b << i
    if is_signed and raw >= (1 << (length - 1)):
        raw -= (1 << length)
    return raw * scale + offset


def extract_motorola(data, start_bit, length, is_signed, scale, offset):
    raw = 0
    bp = start_bit
    for _ in range(length):
        byte_idx = bp // 8
        bit_idx = bp % 8
        if byte_idx >= len(data):
            return float('nan')
        b = (int(data[byte_idx]) >> bit_idx) & 1
        raw = (raw << 1) | b
        bp = bp + 15 if bit_idx == 0 else bp - 1
    if is_signed and raw >= (1 << (length - 1)):
        raw -= (1 << length)
    return raw * scale + offset


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def shade_variants(base, n):
    if n <= 0:
        return []
    if n == 1:
        return [base]
    base = np.array(base[:3], dtype=float)
    out = []
    for f in np.linspace(0.0, 0.6, n):
        out.append((1 - f) * base + f * np.ones(3))
    return out


def rgb_to_qcolor(rgb):
    return QColor(int(max(0, min(1, rgb[0])) * 255),
                  int(max(0, min(1, rgb[1])) * 255),
                  int(max(0, min(1, rgb[2])) * 255))


def load_prefs():
    try:
        with open(PREFS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_prefs(p):
    try:
        with open(PREFS_PATH, 'w', encoding='utf-8') as f:
            json.dump(p, f, indent=2)
    except Exception:
        pass


def sanitize_var_name(s):
    if not s:
        return 'signal'
    v = re.sub(r'\W', '_', s)
    if not v or not (v[0].isalpha() or v[0] == '_'):
        v = 's_' + v
    return v


def sanitize_filename(s, default='signals', ext=''):
    s = (s or '').strip()
    if not s:
        s = default
    if ext and not s.lower().endswith(ext):
        s += ext
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s)
    return s


def make_pen(color, width, style):
    if style == 'none':
        return None
    qt_style = LINE_STYLES.get(style, Qt.SolidLine)
    return pg.mkPen(color=color, width=width, style=qt_style)


def _parse_limit(s):
    s = (s or '').strip()
    if not s:
        return None, True
    try:
        return float(s.replace(',', '.')), True
    except ValueError:
        return None, False


# ---------------------------------------------------------------------------
# ViewBox с выделением области
# ---------------------------------------------------------------------------
class TRCViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseMode(pg.ViewBox.RectMode)
        try:
            self.setMouseEnabled(x=True, y=True)
        except Exception:
            pass
        try:
            self.setAcceptedMouseButtons(
                Qt.LeftButton | Qt.RightButton | Qt.MiddleButton)
        except Exception:
            pass

    def mouseDragEvent(self, ev, axis=None):
        if (ev.button() != Qt.LeftButton or
                self.state.get('mouseMode') != pg.ViewBox.RectMode):
            super().mouseDragEvent(ev, axis)
            return

        ev.accept()

        mods = ev.modifiers()
        x_only = bool(mods & Qt.ShiftModifier)
        y_only = bool(mods & Qt.ControlModifier)

        p1 = QPointF(ev.buttonDownPos())
        p2 = QPointF(ev.pos())

        if x_only and not y_only:
            br = self.viewRect()
            p1 = QPointF(p1.x(), br.top())
            p2 = QPointF(p2.x(), br.bottom())
        elif y_only and not x_only:
            br = self.viewRect()
            p1 = QPointF(br.left(), p1.y())
            p2 = QPointF(br.right(), p2.y())

        if ev.isFinish():
            try:
                self.rbScaleBox.hide()
            except AttributeError:
                pass

            ax = QRectF(p1, p2).normalized()
            if ax.width() <= 1.0 or ax.height() <= 1.0:
                return

            ax = self.childGroup.mapRectFromParent(ax)
            self.showAxRect(ax)
            self.axHistoryPointer += 1
            self.axHistory = (
                self.axHistory[:self.axHistoryPointer] + [ax])
        else:
            self.updateScaleBox(p1, p2)


# ---------------------------------------------------------------------------
# Пассивный ViewBox для дополнительных осей
# ---------------------------------------------------------------------------
class PassiveViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=False, y=False)
        try:
            self.setAcceptedMouseButtons(Qt.NoButton)
        except Exception:
            pass
        try:
            self.setAcceptHoverEvents(False)
        except Exception:
            pass
        try:
            self.setFlag(self.ItemIsFocusable, False)
        except Exception:
            pass
        try:
            self.setFlag(self.ItemIsSelectable, False)
        except Exception:
            pass

    def mousePressEvent(self, ev):
        ev.ignore()

    def mouseReleaseEvent(self, ev):
        ev.ignore()

    def mouseMoveEvent(self, ev):
        ev.ignore()

    def mouseDragEvent(self, ev, axis=None):
        ev.ignore()

    def wheelEvent(self, ev, axis=None):
        ev.ignore()

    def hoverEvent(self, ev):
        pass


# ---------------------------------------------------------------------------
# Мульти-осевой виджет
# ---------------------------------------------------------------------------
class MultiAxisPlotWidget(pg.GraphicsLayoutWidget):
    AXIS_WIDTH = 60
    Z_BASE = 1000

    auto_limits_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.main_plot = self.addPlot(row=0, col=0, viewBox=TRCViewBox())
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        self.main_plot.setLabel('bottom', 'Время, с')

        self.main_plot.getViewBox().disableAutoRange()
        self.main_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        try:
            self.main_plot.getViewBox().setMouseEnabled(x=True, y=True)
        except Exception:
            pass

        try:
            self.main_plot.setMenuEnabled(False)
        except Exception:
            pass
        try:
            vb = self.main_plot.getViewBox()
            if hasattr(vb, 'setMenuEnabled'):
                vb.setMenuEnabled(False)
        except Exception:
            pass

        self.legend = self.main_plot.addLegend(offset=(-10, 10))
        self.legend.setVisible(False)

        self.extra_axes = []
        self._resize_connected = False

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self.auto_limits_requested.emit()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def ensure_axes(self, n):
        while len(self.extra_axes) + 1 < n:
            self._add_axis()

    def _add_axis(self):
        idx = len(self.extra_axes)

        vb = PassiveViewBox()
        vb.setXLink(self.main_plot.vb)
        vb.disableAutoRange()
        vb.setZValue(self.Z_BASE + idx)
        vb.setBackgroundColor(None)
        try:
            if hasattr(vb, 'setMenuEnabled'):
                vb.setMenuEnabled(False)
        except Exception:
            pass
        self.main_plot.scene().addItem(vb)

        axis = pg.AxisItem('right')
        axis.setZValue(self.Z_BASE + idx)
        self.main_plot.scene().addItem(axis)
        axis.linkToView(vb)
        try:
            axis.setAcceptedMouseButtons(Qt.NoButton)
        except Exception:
            pass
        try:
            axis.setAcceptHoverEvents(False)
        except Exception:
            pass

        self.extra_axes.append({'vb': vb, 'axis': axis, 'grid_lines': []})

        right_margin = len(self.extra_axes) * self.AXIS_WIDTH
        self.main_plot.setContentsMargins(0, 0, right_margin, 0)

        if not self._resize_connected:
            self.main_plot.vb.sigResized.connect(self._update_positions)
            self._resize_connected = True

        try:
            vb.linkedViewChanged(self.main_plot.vb, vb.XAxis)
        except Exception:
            pass

        self._update_positions()
        QTimer.singleShot(0, self._update_positions)

    def _update_positions(self):
        try:
            main_rect = self.main_plot.vb.sceneBoundingRect()
        except Exception:
            return
        if main_rect.width() <= 0 or main_rect.height() <= 0:
            return
        for i, info in enumerate(self.extra_axes):
            info['vb'].setGeometry(main_rect)
            info['axis'].setGeometry(
                int(main_rect.right() + i * self.AXIS_WIDTH),
                int(main_rect.top()),
                self.AXIS_WIDTH,
                int(main_rect.height()),
            )
            info['axis'].setHeight(int(main_rect.height()))

    def clear_all_curves(self):
        self.legend.clear()
        self.main_plot.clear()
        for info in self.extra_axes:
            for line in info.get('grid_lines', []):
                try:
                    info['vb'].removeItem(line)
                except Exception:
                    pass
            info['grid_lines'] = []
            info['vb'].clear()
            self.main_plot.scene().removeItem(info['vb'])
            self.main_plot.scene().removeItem(info['axis'])
        self.extra_axes = []
        self.main_plot.setContentsMargins(0, 0, 0, 0)

    def add_curve(self, axis_idx, x, y, pen, symbol, symbol_size, legend_label):
        if axis_idx == 0:
            curve = self.main_plot.plot(
                x, y, pen=pen, symbol=symbol, symbolSize=symbol_size)
        else:
            if axis_idx - 1 >= len(self.extra_axes):
                return None
            info = self.extra_axes[axis_idx - 1]
            curve = pg.PlotDataItem(
                x, y, pen=pen, symbol=symbol, symbolSize=symbol_size)
            info['vb'].addItem(curve)
        self.legend.addItem(curve, legend_label)
        return curve

    def set_axis_style(self, axis_idx, label, color):
        if axis_idx == 0:
            ax = self.main_plot.getAxis('left')
        else:
            if axis_idx - 1 >= len(self.extra_axes):
                return
            ax = self.extra_axes[axis_idx - 1]['axis']

        if isinstance(color, QColor):
            qcolor = color
        elif isinstance(color, (tuple, list)):
            qcolor = QColor(int(color[0]), int(color[1]), int(color[2]))
        else:
            qcolor = QColor(color)

        pen = pg.mkPen(qcolor)
        ax.setPen(pen)
        ax.setTextPen(pen)
        try:
            ax.setLabel(label, color=qcolor.name())
        except Exception:
            ax.setLabel(label)

    def set_legend_visible(self, visible):
        self.legend.setVisible(visible)

    def set_ylim(self, axis_idx, ymin, ymax):
        if axis_idx == 0:
            vb = self.main_plot.vb
        else:
            if axis_idx - 1 >= len(self.extra_axes):
                return
            vb = self.extra_axes[axis_idx - 1]['vb']
        vb.setYRange(ymin, ymax, padding=0)

    def set_xlim(self, xmin, xmax):
        self.main_plot.setXRange(xmin, xmax, padding=0)


# ---------------------------------------------------------------------------
# Окно графика
# ---------------------------------------------------------------------------
class PlotWindow(QDialog):
    closed = pyqtSignal(object)
    state_changed = pyqtSignal()

    def __init__(self, app, index, title, background_color=None):
        super().__init__()
        self.app = app
        self.index = index
        self.window_title = title
        self.setWindowTitle(title)
        self.resize(1500, 900)

        self.plots_per_axes = [[]]
        self.axes_colors = [None]
        self.y_limits = [[None, None]]
        self.x_limits = [None, None]
        self.axis_labels = [None]
        self.custom_title = None
        self.show_legend = False
        self.background_color = background_color
        self._fg_qcolor = QColor('#000000')

        self._curve_objects = {}
        self._picked_color = None
        self.y_grid_checkboxes = []

        self._build_ui()

        self.plot_widget.auto_limits_requested.connect(
            self._on_middle_click_auto_limits)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ------- Верхняя строка с переключателями видимости панелей -------
        top_row = QHBoxLayout()
        self.show_panel_cb = QCheckBox('Показать панель настроек справа')
        self.show_panel_cb.setChecked(True)
        self.show_panel_cb.stateChanged.connect(self._toggle_side_panel)
        top_row.addWidget(self.show_panel_cb)

        self.show_edit_cb = QCheckBox('Показать панель редактирования')
        self.show_edit_cb.setChecked(True)
        self.show_edit_cb.stateChanged.connect(self._toggle_edit_panel)
        top_row.addWidget(self.show_edit_cb)

        top_row.addStretch()
        layout.addLayout(top_row)

        # ------- Площадка графика + правая панель (две колонки) -------
        plot_area = QSplitter(Qt.Horizontal)
        plot_area.setChildrenCollapsible(True)
        plot_area.setHandleWidth(4)

        self.plot_widget = MultiAxisPlotWidget()
        plot_area.addWidget(self.plot_widget)

        # Правая панель: контейнер из двух независимых колонок
        self.side_panel = QWidget()
        sp_layout = QHBoxLayout(self.side_panel)
        sp_layout.setContentsMargins(0, 0, 0, 0)
        sp_layout.setSpacing(4)

        self.settings_panel = self._build_side_panel()
        sp_layout.addWidget(self.settings_panel)

        self.edit_panel = self._build_edit_panel()
        sp_layout.addWidget(self.edit_panel)

        plot_area.addWidget(self.side_panel)

        plot_area.setStretchFactor(0, 1)
        plot_area.setStretchFactor(1, 0)
        plot_area.setSizes([700, 700])

        layout.addWidget(plot_area, stretch=1)

        # ------- Дерево кривых (внизу) -------
        curves_panel = QGroupBox('Кривые на графике')
        cp = QVBoxLayout(curves_panel)
        cp.setContentsMargins(4, 4, 4, 4)
        self.curves_tree = QTreeWidget()
        self.curves_tree.setHeaderLabels(['Видимость', 'Кривая', 'Ось'])
        self.curves_tree.setRootIsDecorated(False)
        self.curves_tree.setAlternatingRowColors(True)
        self.curves_tree.setMaximumHeight(150)
        self.curves_tree.setColumnWidth(0, 80)
        self.curves_tree.setColumnWidth(1, 700)
        self.curves_tree.setColumnWidth(2, 70)
        self.curves_tree.itemChanged.connect(
            self._on_curve_visibility_changed)
        cp.addWidget(self.curves_tree)
        row = QHBoxLayout()
        b = QPushButton('Показать все')
        b.clicked.connect(self._show_all_curves)
        row.addWidget(b)
        b = QPushButton('Скрыть все')
        b.clicked.connect(self._hide_all_curves)
        row.addWidget(b)
        row.addStretch()
        cp.addLayout(row)
        layout.addWidget(curves_panel)

        self._refresh_curve_axis_combo()

    # --------------------------------------------------- Левая колонка
    def _build_side_panel(self):
        """Левая колонка правой панели — «Настройки графика»."""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(280)
        v = QVBoxLayout(panel)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        self.y_grid_checkboxes = []

        title = QLabel('Настройки графика')
        title.setStyleSheet('font-weight: bold;')
        v.addWidget(title)

        b = QPushButton('Показать всё')
        b.clicked.connect(self._view_all)
        v.addWidget(b)

        b = QPushButton('Авто-диапазон (все оси)')
        b.clicked.connect(self._on_middle_click_auto_limits)
        v.addWidget(b)

        # --- Сетка
        grid_box = QGroupBox('Сетка')
        gb = QVBoxLayout(grid_box)
        self.grid_x_cb = QCheckBox('По X')
        self.grid_x_cb.setChecked(True)
        self.grid_x_cb.stateChanged.connect(self._update_grid)
        gb.addWidget(self.grid_x_cb)

        self.grid_y_container = QWidget()
        self.grid_y_layout = QVBoxLayout(self.grid_y_container)
        self.grid_y_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_y_layout.setSpacing(2)
        gb.addWidget(self.grid_y_container)
        v.addWidget(grid_box)

        # --- Легенда (единственный переключатель легенды)
        self.side_legend_cb = QCheckBox('Показывать легенду')
        self.side_legend_cb.setChecked(self.show_legend)
        self.side_legend_cb.stateChanged.connect(
            self._toggle_legend_from_panel)
        v.addWidget(self.side_legend_cb)

        # --- Фон графика
        bg_box = QGroupBox('Фон графика')
        bgb = QVBoxLayout(bg_box)
        row = QHBoxLayout()
        self.bg_btn = QPushButton('Авто')
        self.bg_btn.clicked.connect(self.pick_background)
        row.addWidget(self.bg_btn, stretch=1)
        b = QPushButton('Сброс')
        b.clicked.connect(self.reset_background)
        row.addWidget(b)
        bgb.addLayout(row)
        v.addWidget(bg_box)

        # --- Пределы осей
        lim_box = QGroupBox('Пределы осей')
        lb = QVBoxLayout(lim_box)

        row = QHBoxLayout()
        row.addWidget(QLabel('Ось:'))
        self.limit_axis_combo = QComboBox()
        self.limit_axis_combo.addItem('Ось 1')
        self.limit_axis_combo.currentIndexChanged.connect(
            self.sync_limit_edits)
        row.addWidget(self.limit_axis_combo, stretch=1)
        lb.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Ymin:'))
        self.y_min_edit = QLineEdit()
        row.addWidget(self.y_min_edit, stretch=1)
        row.addWidget(QLabel('Ymax:'))
        self.y_max_edit = QLineEdit()
        row.addWidget(self.y_max_edit, stretch=1)
        lb.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Xmin:'))
        self.x_min_edit = QLineEdit()
        row.addWidget(self.x_min_edit, stretch=1)
        row.addWidget(QLabel('Xmax:'))
        self.x_max_edit = QLineEdit()
        row.addWidget(self.x_max_edit, stretch=1)
        lb.addLayout(row)

        row = QHBoxLayout()
        b = QPushButton('Применить')
        b.clicked.connect(self.apply_limits)
        row.addWidget(b)
        b = QPushButton('Авто')
        b.clicked.connect(self.reset_limits)
        row.addWidget(b)
        lb.addLayout(row)

        v.addWidget(lim_box)

        info = QLabel('Мышь:\n'
                      'ЛКМ-протяжка = область\n'
                      'Shift+ЛКМ = только X\n'
                      'Ctrl+ЛКМ = только Y\n'
                      'ПКМ = панорамирование\n'
                      'СКМ = авто-пределы')
        info.setStyleSheet('color: #666; font-size: 10px;')
        v.addWidget(info)

        v.addStretch()

        self._update_bg_btn()
        self._refresh_limit_axis_combo()
        self._refresh_grid_axes()
        return panel

    # -------------------------------------------------- Правая колонка
    def _build_edit_panel(self):
        """Правая колонка правой панели — «Редактирование отображения»."""
        panel = QGroupBox('Редактирование отображения')
        panel.setMinimumWidth(420)
        panel.setMaximumWidth(620)
        edit_layout = QVBoxLayout(panel)
        edit_layout.setSpacing(4)

        # Заголовок
        row = QHBoxLayout()
        row.addWidget(QLabel('Заголовок:'))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText('(пусто = авто)')
        row.addWidget(self.title_edit, stretch=1)
        b = QPushButton('OK')
        b.clicked.connect(self.apply_title)
        row.addWidget(b)
        edit_layout.addLayout(row)

        # Подпись оси
        row = QHBoxLayout()
        row.addWidget(QLabel('Ось:'))
        self.axis_combo = QComboBox()
        self.axis_combo.setMinimumWidth(80)
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        row.addWidget(self.axis_combo)
        self.axis_label_edit = QLineEdit()
        self.axis_label_edit.setPlaceholderText('(пусто = авто)')
        row.addWidget(self.axis_label_edit, stretch=1)
        b = QPushButton('OK')
        b.clicked.connect(self.apply_axis_label)
        row.addWidget(b)
        edit_layout.addLayout(row)

        # Цвет оси
        row = QHBoxLayout()
        row.addWidget(QLabel('Цвет оси:'))
        self.axis_color_btn = QPushButton('Авто')
        self.axis_color_btn.clicked.connect(self.pick_axis_color)
        row.addWidget(self.axis_color_btn, stretch=1)
        b = QPushButton('Сброс')
        b.clicked.connect(self.reset_axis_color)
        row.addWidget(b)
        edit_layout.addLayout(row)

        # Кривая
        row = QHBoxLayout()
        row.addWidget(QLabel('Кривая:'))
        self.curve_combo = QComboBox()
        self.curve_combo.setMinimumWidth(180)
        self.curve_combo.currentIndexChanged.connect(self._on_curve_changed)
        row.addWidget(self.curve_combo, stretch=1)
        edit_layout.addLayout(row)

        # Подпись в легенде
        row = QHBoxLayout()
        row.addWidget(QLabel('Легенда:'))
        self.legend_edit = QLineEdit()
        self.legend_edit.setPlaceholderText('(пусто = авто)')
        row.addWidget(self.legend_edit, stretch=1)
        b = QPushButton('OK')
        b.clicked.connect(self.apply_legend_label)
        row.addWidget(b)
        edit_layout.addLayout(row)

        # Ось кривой
        row = QHBoxLayout()
        row.addWidget(QLabel('Ось кривой:'))
        self.curve_axis_combo = QComboBox()
        self.curve_axis_combo.setMinimumWidth(120)
        row.addWidget(self.curve_axis_combo, stretch=1)
        b = QPushButton('Переместить')
        b.clicked.connect(self.move_curve_to_axis)
        row.addWidget(b)
        edit_layout.addLayout(row)

        # Цвет кривой
        row = QHBoxLayout()
        row.addWidget(QLabel('Цвет:'))
        self.color_btn = QPushButton('Авто')
        self.color_btn.clicked.connect(self.pick_color)
        row.addWidget(self.color_btn, stretch=1)
        b = QPushButton('Сброс')
        b.clicked.connect(self.auto_color)
        row.addWidget(b)
        edit_layout.addLayout(row)

        # Стиль линии — разбит на два ряда, чтобы подписи не наезжали
        row = QHBoxLayout()
        row.addWidget(QLabel('Тип:'))
        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(list(LINE_STYLES.keys()) + ['none'])
        self.line_style_combo.setMinimumWidth(60)
        row.addWidget(self.line_style_combo)
        row.addWidget(QLabel('Толщ.:'))
        self.line_width_combo = QComboBox()
        self.line_width_combo.addItems(
            ['0.5', '1', '1.5', '2', '2.5', '3', '4'])
        self.line_width_combo.setCurrentText('1.5')
        row.addWidget(self.line_width_combo)
        row.addStretch()
        edit_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Марк.:'))
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(MARKERS)
        row.addWidget(self.marker_combo)
        row.addWidget(QLabel('Разм.:'))
        self.marker_size_combo = QComboBox()
        self.marker_size_combo.addItems(
            ['3', '4', '5', '6', '8', '10', '12'])
        self.marker_size_combo.setCurrentText('6')
        row.addWidget(self.marker_size_combo)
        row.addStretch()
        edit_layout.addLayout(row)

        # Кнопки
        row = QHBoxLayout()
        b = QPushButton('Применить стиль')
        b.clicked.connect(self.apply_style)
        row.addWidget(b)
        b = QPushButton('Удалить кривую')
        b.clicked.connect(self.delete_current_curve)
        row.addWidget(b)
        b = QPushButton('Очистить')
        b.clicked.connect(self.clear_plot)
        row.addWidget(b)
        edit_layout.addLayout(row)

        hint = QLabel('Масштаб: ЛКМ = область; Shift+ЛКМ = X; '
                      'Ctrl+ЛКМ = Y; ПКМ = пан.; СКМ = авто')
        hint.setStyleSheet('color: #777; font-size: 10px;')
        hint.setWordWrap(True)
        edit_layout.addWidget(hint)

        edit_layout.addStretch()
        return panel

    # ------------------------------------------------- видимость панелей
    def _toggle_side_panel(self, state):
        self.side_panel.setVisible(state == Qt.Checked)
        QTimer.singleShot(0, self._after_panel_toggle)

    def _toggle_edit_panel(self, state):
        self.edit_panel.setVisible(state == Qt.Checked)
        QTimer.singleShot(0, self._after_panel_toggle)

    def _after_panel_toggle(self):
        try:
            self.plot_widget._update_positions()
            self.plot_widget.update()
            if hasattr(self, 'grid_x_cb'):
                self._update_grid()
        except Exception:
            pass

    def _view_all(self):
        self.x_limits = [None, None]
        for i in range(len(self.y_limits)):
            self.y_limits[i] = [None, None]
        self.redraw()
        self.app.log(f'График #{self.index}: показаны все данные.')

    # ----------------------------------------------------- сетка
    def _refresh_grid_axes(self):
        """Пересоздать чекбоксы «По Y» — по одному на каждую ось."""
        if not hasattr(self, 'grid_y_container'):
            return
        n = len(self.plots_per_axes)
        current = [cb.isChecked() for cb in self.y_grid_checkboxes]
        while self.grid_y_layout.count():
            item = self.grid_y_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.y_grid_checkboxes = []
        for k in range(n):
            cb = QCheckBox(f'По Y (ось {k + 1})')
            cb.blockSignals(True)
            cb.setChecked(current[k] if k < len(current) else True)
            cb.blockSignals(False)
            cb.stateChanged.connect(self._update_grid)
            self.grid_y_layout.addWidget(cb)
            self.y_grid_checkboxes.append(cb)
        self._update_grid()

    def _update_grid(self):
        try:
            x_enabled = self.grid_x_cb.isChecked()
        except Exception:
            x_enabled = True

        y_main = False
        try:
            if self.y_grid_checkboxes:
                y_main = self.y_grid_checkboxes[0].isChecked()
        except Exception:
            pass

        try:
            self.plot_widget.main_plot.showGrid(
                x=x_enabled, y=y_main, alpha=0.3)
        except Exception:
            pass

        try:
            n = len(self.plots_per_axes)
            for ai in range(1, n):
                enabled = (ai < len(self.y_grid_checkboxes) and
                           self.y_grid_checkboxes[ai].isChecked())
                self._set_extra_axis_grid(ai, enabled)
        except Exception:
            pass

    def _set_extra_axis_grid(self, ai, y_enabled):
        """Горизонтальные линии сетки для доп. Y-оси ai (>=1)."""
        if ai - 1 >= len(self.plot_widget.extra_axes):
            return
        info = self.plot_widget.extra_axes[ai - 1]
        vb = info['vb']
        axis = info['axis']

        for line in info.get('grid_lines', []):
            try:
                vb.removeItem(line)
            except Exception:
                pass
        info['grid_lines'] = []

        if not y_enabled:
            return

        try:
            y_min, y_max = vb.viewRange()[1]
        except Exception:
            return
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return
        if y_min >= y_max:
            return

        try:
            size = max(int(vb.boundingRect().height()), 10)
        except Exception:
            size = 100

        values = []
        try:
            tick_levels = axis.tickValues(y_min, y_max, size)
            for level in tick_levels:
                if not isinstance(level, (list, tuple)):
                    continue
                if len(level) < 2:
                    continue
                vals = level[1]
                if not hasattr(vals, '__iter__'):
                    continue
                for vv in vals:
                    try:
                        fv = float(vv)
                    except (TypeError, ValueError):
                        continue
                    if y_min <= fv <= y_max:
                        values.append(fv)
        except Exception:
            values = []

        if not values:
            return

        pen = pg.mkPen(QColor(170, 170, 170, 110))
        for vv in values:
            try:
                line = pg.InfiniteLine(
                    pos=vv, angle=0, movable=False, pen=pen)
                line.setZValue(-100)
                vb.addItem(line)
                info['grid_lines'].append(line)
            except Exception:
                pass

    def _toggle_legend_from_panel(self, state):
        self.show_legend = (state == Qt.Checked)
        self.plot_widget.set_legend_visible(self.show_legend)

    # ------------------------------------------------------- события окна
    def closeEvent(self, event):
        self.closed.emit(self)
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self.plot_widget._update_positions()
        except Exception:
            pass

    def add_plots(self, new_plots, axis_choice, custom_color):
        n_axes = len(self.plots_per_axes)
        if axis_choice >= n_axes:
            self.plots_per_axes.append(list(new_plots))
            self.axes_colors.append(custom_color)
            self.y_limits.append([None, None])
            self.axis_labels.append(None)
        else:
            self.plots_per_axes[axis_choice].extend(new_plots)
        self.redraw()

    def set_background(self, qcolor):
        self.background_color = qcolor
        self.redraw()
        self._update_bg_btn()

    def _is_dark_bg(self):
        if self.background_color is None:
            return False
        bg = self.background_color
        lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
        return lum < 128

    @staticmethod
    def _lighten(c):
        return tuple(min(255, int(v * 0.65 + 255 * 0.35)) for v in c)

    def _apply_theme(self):
        if self.background_color is None:
            bg = QColor('#ffffff')
        else:
            bg = self.background_color

        lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
        fg = QColor('#000000') if lum > 128 else QColor('#ffffff')
        self._fg_qcolor = fg

        self.plot_widget.setBackground(bg)

        for axis_name in ('bottom', 'top'):
            axis = self.plot_widget.main_plot.getAxis(axis_name)
            if axis is not None:
                axis.setPen(fg)
                axis.setTextPen(fg)

        self.plot_widget.main_plot.setLabel(
            'bottom', 'Время, с', color=fg.name())

        legend = self.plot_widget.legend
        try:
            legend.setLabelTextColor(fg)
        except Exception:
            pass
        try:
            legend.setPen(pg.mkPen(fg))
        except Exception:
            pass
        try:
            legend.setBrush(pg.mkBrush(QColor(
                bg.red(), bg.green(), bg.blue(), 220)))
        except Exception:
            pass

    # ------------------------------------------------------------- отрисовка
    def redraw(self):
        n_axes = len(self.plots_per_axes)

        self.plot_widget.clear_all_curves()
        self.plot_widget.ensure_axes(n_axes)
        try:
            self.plot_widget._update_positions()
        except Exception:
            pass
        self._apply_theme()

        self._refresh_grid_axes()

        dark = self._is_dark_bg()
        self._curve_objects = {}

        for ai, plots in enumerate(self.plots_per_axes):
            custom = (self.axes_colors[ai]
                      if ai < len(self.axes_colors) else None)
            if custom is not None:
                raw_colors = shade_variants(custom, max(len(plots), 1))
                line_colors = [
                    tuple(int(x * 255) for x in c[:3]) for c in raw_colors
                ]
                axis_color = tuple(int(c * 255) for c in custom[:3])
            else:
                line_colors = [
                    tuple(int(c * 255) for c in
                          MATLAB_LINES[k % len(MATLAB_LINES)])
                    for k in range(max(len(plots), 1))
                ]
                axis_color = line_colors[0] if line_colors else (0, 0, 0)

            if dark:
                line_colors = [self._lighten(c) for c in line_colors]
                axis_color = self._lighten(axis_color)

            if ai < len(self.axis_labels) and self.axis_labels[ai]:
                y_label = self.axis_labels[ai]
            elif plots:
                y_label = plots[0]['label']
            else:
                y_label = ''

            self.plot_widget.set_axis_style(ai, y_label, axis_color)

            for j, p in enumerate(plots):
                c = p.get('color')
                if c is None:
                    c = line_colors[j % len(line_colors)]
                else:
                    c = tuple(int(x * 255) for x in c[:3])

                ls = p.get('lineStyle', '-')
                lw = p.get('lineWidth', 1.5)
                mk = p.get('marker', 'none')
                ms = float(p.get('markerSize', 6))

                legend_lbl = (p.get('legend_label')
                              or f"Ось {ai + 1}: {p['label']}")

                pen = make_pen(c, lw, ls)
                symbol = None if mk == 'none' else mk

                try:
                    curve = self.plot_widget.add_curve(
                        ai, p['t'], p['v'], pen, symbol, ms, legend_lbl)
                except Exception as e:
                    self.app.log(f'Ошибка добавления кривой ({ai}/{j}): {e}')
                    curve = None

                if curve is not None:
                    self._curve_objects[(ai, j)] = curve
                    try:
                        curve.setVisible(p.get('visible', True))
                    except Exception:
                        pass

        if self.custom_title:
            title_text = self.custom_title
        else:
            first_msg = self.plots_per_axes[0][0]['msgName'] \
                if self.plots_per_axes[0] else '(пусто)'
            total = sum(len(p) for p in self.plots_per_axes)
            title_text = (f'График #{self.index} — {first_msg}  '
                          f'({total} сигнал(ов), {n_axes} осей)')
        self.plot_widget.main_plot.setTitle(
            title_text, color=self._fg_qcolor.name())

        self.plot_widget.set_legend_visible(self.show_legend)
        if hasattr(self, 'side_legend_cb'):
            self.side_legend_cb.blockSignals(True)
            self.side_legend_cb.setChecked(self.show_legend)
            self.side_legend_cb.blockSignals(False)

        self._recompute_ranges()

        try:
            self.plot_widget._update_positions()
        except Exception:
            pass

        self._refresh_axis_combo()
        self._refresh_curve_combo()
        self._refresh_curves_tree()
        self._refresh_curve_axis_combo()
        self._refresh_limit_axis_combo()
        self._update_bg_btn()

        QTimer.singleShot(10, self._recompute_ranges)

        self.state_changed.emit()

    def _recompute_ranges(self):
        xs = []
        for plots in self.plots_per_axes:
            for p in plots:
                if p.get('visible', True):
                    t = np.asarray(p['t'], dtype=float)
                    t = t[np.isfinite(t)]
                    if len(t):
                        xs.append(t)

        xmin, xmax = self.x_limits
        if xs:
            all_x = np.concatenate(xs)
            if xmin is None:
                xmin = float(all_x.min())
            if xmax is None:
                xmax = float(all_x.max())
        if xmin is not None and xmax is not None:
            if np.isfinite(xmin) and np.isfinite(xmax):
                if xmin == xmax:
                    xmin -= 1.0
                    xmax += 1.0
                if xmin < xmax:
                    try:
                        self.plot_widget.set_xlim(xmin, xmax)
                    except Exception:
                        pass

        for ai, plots in enumerate(self.plots_per_axes):
            ys = []
            for p in plots:
                if p.get('visible', True):
                    v = np.asarray(p['v'], dtype=float)
                    v = v[np.isfinite(v)]
                    if len(v):
                        ys.append(v)

            ymin, ymax = (self.y_limits[ai]
                          if ai < len(self.y_limits) else (None, None))

            if ys:
                all_y = np.concatenate(ys)
                if ymin is None:
                    ymin = float(all_y.min())
                if ymax is None:
                    ymax = float(all_y.max())

            if ymin is None or ymax is None:
                continue
            if not (np.isfinite(ymin) and np.isfinite(ymax)):
                continue
            if ymin == ymax:
                ymin -= 1.0
                ymax += 1.0
            if not (ymin < ymax):
                continue

            try:
                self.plot_widget.set_ylim(ai, ymin, ymax)
            except Exception as e:
                self.app.log(f'  [range] ось {ai + 1} ОШИБКА: {e}')

        try:
            self._update_grid()
        except Exception:
            pass

    def _recompute_ranges_for_axis(self, ai):
        if ai >= len(self.plots_per_axes):
            return
        plots = self.plots_per_axes[ai]
        ys = []
        for p in plots:
            if p.get('visible', True):
                v = np.asarray(p['v'], dtype=float)
                v = v[np.isfinite(v)]
                if len(v):
                    ys.append(v)
        ymin, ymax = (self.y_limits[ai]
                      if ai < len(self.y_limits) else (None, None))
        if ys:
            all_y = np.concatenate(ys)
            if ymin is None:
                ymin = float(all_y.min())
            if ymax is None:
                ymax = float(all_y.max())
        if ymin is None or ymax is None:
            return
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        try:
            self.plot_widget.set_ylim(ai, ymin, ymax)
        except Exception:
            pass
        try:
            self._update_grid()
        except Exception:
            pass

    def _refresh_axis_combo(self):
        n = len(self.plots_per_axes)
        old = self.axis_combo.currentIndex()
        self.axis_combo.blockSignals(True)
        self.axis_combo.clear()
        for k in range(n):
            self.axis_combo.addItem(f'Ось {k + 1}')
        if 0 <= old < n:
            self.axis_combo.setCurrentIndex(old)
        self.axis_combo.blockSignals(False)
        self._on_axis_changed()

    def _refresh_curve_combo(self):
        old = self.curve_combo.currentData()
        self.curve_combo.blockSignals(True)
        self.curve_combo.clear()
        idx = 0
        sel = 0
        for ai, plots in enumerate(self.plots_per_axes):
            for j, p in enumerate(plots):
                self.curve_combo.addItem(
                    f'Ось {ai + 1}: {p["label"]}', (ai, j))
                if old == (ai, j):
                    sel = idx
                idx += 1
        if self.curve_combo.count() > 0:
            self.curve_combo.setCurrentIndex(sel)
        self.curve_combo.blockSignals(False)
        self._on_curve_changed()

    def _refresh_curves_tree(self):
        if not hasattr(self, 'curves_tree'):
            return
        self.curves_tree.blockSignals(True)
        self.curves_tree.clear()
        for ai, plots in enumerate(self.plots_per_axes):
            for j, p in enumerate(plots):
                item = QTreeWidgetItem()
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    0, Qt.Checked if p.get('visible', True) else Qt.Unchecked)
                item.setText(1, p.get('legend_label') or p['label'])
                item.setText(2, f'Ось {ai + 1}')
                item.setData(0, Qt.UserRole, (ai, j))
                self.curves_tree.addTopLevelItem(item)
        self.curves_tree.blockSignals(False)

    def _on_curve_visibility_changed(self, item, column):
        if column != 0:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        ai, j = data
        if ai >= len(self.plots_per_axes) or j >= len(self.plots_per_axes[ai]):
            return
        p = self.plots_per_axes[ai][j]
        visible = (item.checkState(0) == Qt.Checked)
        p['visible'] = visible

        curve = self._curve_objects.get((ai, j))
        if curve is not None:
            try:
                curve.setVisible(visible)
            except Exception:
                pass

        self._recompute_ranges_for_axis(ai)

        self.app.log(
            f'Кривая «{p["label"]}» (Ось {ai + 1}) '
            f'{"показана" if visible else "скрыта"}.')

    def _show_all_curves(self):
        for plots in self.plots_per_axes:
            for p in plots:
                p['visible'] = True
        for curve in self._curve_objects.values():
            try:
                curve.setVisible(True)
            except Exception:
                pass
        self._refresh_curves_tree()
        self._recompute_ranges()
        self.app.log('Показаны все кривые.')

    def _hide_all_curves(self):
        for plots in self.plots_per_axes:
            for p in plots:
                p['visible'] = False
        for curve in self._curve_objects.values():
            try:
                curve.setVisible(False)
            except Exception:
                pass
        self._refresh_curves_tree()
        self.app.log('Скрыты все кривые.')

    def _on_axis_changed(self):
        ai = self.axis_combo.currentIndex()
        has_color_btn = hasattr(self, 'axis_color_btn')
        if ai < 0 or ai >= len(self.axis_labels):
            self.axis_label_edit.clear()
            if has_color_btn:
                self.axis_color_btn.setText('Авто')
                self.axis_color_btn.setStyleSheet('')
            return
        self.axis_label_edit.setText(self.axis_labels[ai] or '')
        if not has_color_btn:
            return
        if ai < len(self.axes_colors) and self.axes_colors[ai] is not None:
            qc = rgb_to_qcolor(self.axes_colors[ai])
            self.axis_color_btn.setText('Выбран')
            self.axis_color_btn.setStyleSheet(
                f'background-color: {qc.name()}; color: '
                f'{"black" if qc.lightness() > 128 else "white"};')
        else:
            self.axis_color_btn.setText('Авто')
            self.axis_color_btn.setStyleSheet('')

    def _on_curve_changed(self):
        data = self.curve_combo.currentData()
        if not data:
            self.legend_edit.clear()
            return
        ai, j = data
        if ai >= len(self.plots_per_axes) or j >= len(self.plots_per_axes[ai]):
            return
        p = self.plots_per_axes[ai][j]
        self.legend_edit.setText(p.get('legend_label', ''))

        c = p.get('color')
        self._picked_color = c
        if c is None:
            self.color_btn.setText('Авто')
            self.color_btn.setStyleSheet('')
        else:
            qc = rgb_to_qcolor(c)
            self.color_btn.setText('Выбран')
            self.color_btn.setStyleSheet(
                f'background-color: {qc.name()}; color: '
                f'{"black" if qc.lightness() > 128 else "white"};')

        self.line_style_combo.setCurrentText(p.get('lineStyle', '-'))
        self.line_width_combo.setCurrentText(
            str(float(p.get('lineWidth', 1.5))))
        self.marker_combo.setCurrentText(p.get('marker', 'none'))
        self.marker_size_combo.setCurrentText(
            str(int(float(p.get('markerSize', 6)))))

        if hasattr(self, 'curve_axis_combo') and \
                0 <= ai < self.curve_axis_combo.count():
            self.curve_axis_combo.blockSignals(True)
            self.curve_axis_combo.setCurrentIndex(ai)
            self.curve_axis_combo.blockSignals(False)

    def apply_title(self):
        txt = self.title_edit.text().strip()
        self.custom_title = txt if txt else None
        self.redraw()
        self.app.log(f'Заголовок графика #{self.index} обновлён.')

    def apply_axis_label(self):
        ai = self.axis_combo.currentIndex()
        if ai < 0 or ai >= len(self.axis_labels):
            return
        txt = self.axis_label_edit.text().strip()
        self.axis_labels[ai] = txt if txt else None
        self.redraw()
        self.app.log(f'Подпись оси {ai + 1} графика #{self.index} обновлена.')

    def pick_axis_color(self):
        ai = self.axis_combo.currentIndex()
        if ai < 0 or ai >= len(self.axes_colors):
            return
        init = self.axes_colors[ai] or DEFAULT_COLOR
        c = QColorDialog.getColor(
            rgb_to_qcolor(init), self, 'Цвет оси')
        if not c.isValid():
            return
        self.axes_colors[ai] = (c.red() / 255.0, c.green() / 255.0,
                                c.blue() / 255.0)
        self.redraw()
        self.app.log(
            f'Цвет оси {ai + 1} графика #{self.index} обновлён: {c.name()}.')

    def reset_axis_color(self):
        ai = self.axis_combo.currentIndex()
        if ai < 0 or ai >= len(self.axes_colors):
            return
        self.axes_colors[ai] = None
        self.redraw()
        self.app.log(
            f'Цвет оси {ai + 1} графика #{self.index} сброшен (авто).')

    def _on_middle_click_auto_limits(self):
        for i in range(len(self.y_limits)):
            self.y_limits[i] = [None, None]
        self.x_limits = [None, None]
        self.redraw()
        self.app.log(
            f'Пределы осей графика #{self.index} сброшены в авто.')

    def apply_legend_label(self):
        data = self.curve_combo.currentData()
        if not data:
            return
        ai, j = data
        if ai >= len(self.plots_per_axes) or j >= len(self.plots_per_axes[ai]):
            return
        txt = self.legend_edit.text().strip()
        p = self.plots_per_axes[ai][j]
        if txt:
            p['legend_label'] = txt
        else:
            p.pop('legend_label', None)
        self.redraw()
        self.app.log(f'Подпись легенды обновлена: '
                     f'Ось {ai + 1} / {p["label"]}.')

    def pick_color(self):
        init = self._picked_color
        if init is None:
            init = DEFAULT_COLOR
        c = QColorDialog.getColor(
            rgb_to_qcolor(init), self, 'Цвет линии')
        if not c.isValid():
            return
        self._picked_color = (c.red() / 255.0, c.green() / 255.0,
                              c.blue() / 255.0)
        self.color_btn.setText('Выбран')
        self.color_btn.setStyleSheet(
            f'background-color: {c.name()}; color: '
            f'{"black" if c.lightness() > 128 else "white"};')

    def auto_color(self):
        self._picked_color = None
        self.color_btn.setText('Авто')
        self.color_btn.setStyleSheet('')

    def apply_style(self):
        data = self.curve_combo.currentData()
        if not data:
            return
        ai, j = data
        if ai >= len(self.plots_per_axes) or j >= len(self.plots_per_axes[ai]):
            return
        p = self.plots_per_axes[ai][j]
        p['color'] = self._picked_color
        p['lineStyle'] = self.line_style_combo.currentText()
        p['lineWidth'] = float(self.line_width_combo.currentText())
        p['marker'] = self.marker_combo.currentText()
        p['markerSize'] = float(self.marker_size_combo.currentText())
        self.redraw()
        self.app.log(f'Стиль применён: Ось {ai + 1} / {p["label"]}.')

    def delete_current_curve(self):
        data = self.curve_combo.currentData()
        if not data:
            return
        ai, j = data
        if ai >= len(self.plots_per_axes) or j >= len(self.plots_per_axes[ai]):
            return
        p = self.plots_per_axes[ai].pop(j)
        if ai > 0 and not self.plots_per_axes[ai]:
            self.plots_per_axes.pop(ai)
            self.axes_colors.pop(ai)
            self.y_limits.pop(ai)
            if ai < len(self.axis_labels):
                self.axis_labels.pop(ai)
        self.redraw()
        self.app.log(f'Кривая удалена: Ось {ai + 1} / {p["label"]}.')

    def clear_plot(self):
        self.plots_per_axes = [[]]
        self.axes_colors = [None]
        self.y_limits = [[None, None]]
        self.x_limits = [None, None]
        self.axis_labels = [None]
        self.custom_title = None
        self.redraw()
        self.app.log(f'Все данные удалены с графика #{self.index}.')

    # ---- Фон графика ----
    def pick_background(self):
        init = self.background_color or QColor('#ffffff')
        c = QColorDialog.getColor(init, self, 'Фон графика')
        if not c.isValid():
            return
        self.background_color = c
        self.redraw()
        self._update_bg_btn()
        self.app.log(f'Фон графика #{self.index}: {c.name()}.')

    def reset_background(self):
        self.background_color = None
        self.redraw()
        self._update_bg_btn()
        self.app.log(f'Фон графика #{self.index}: авто (белый).')

    def _update_bg_btn(self):
        if not hasattr(self, 'bg_btn'):
            return
        if self.background_color is None:
            self.bg_btn.setText('Авто')
            self.bg_btn.setStyleSheet('')
        else:
            c = self.background_color
            self.bg_btn.setText('Выбран')
            self.bg_btn.setStyleSheet(
                f'background-color: {c.name()}; color: '
                f'{"black" if c.lightness() > 128 else "white"};')

    # ---- Ось кривой ----
    def _refresh_curve_axis_combo(self):
        if not hasattr(self, 'curve_axis_combo'):
            return
        old = self.curve_axis_combo.currentIndex()
        self.curve_axis_combo.blockSignals(True)
        self.curve_axis_combo.clear()
        n = len(self.plots_per_axes)
        for k in range(n):
            self.curve_axis_combo.addItem(f'Ось {k + 1}')
        self.curve_axis_combo.addItem('Новая ось')
        if 0 <= old < self.curve_axis_combo.count():
            self.curve_axis_combo.setCurrentIndex(old)
        self.curve_axis_combo.blockSignals(False)

    def move_curve_to_axis(self):
        data = self.curve_combo.currentData()
        if not data:
            self.app.log('Не выбрана кривая для перемещения.')
            return
        ai_src, j_src = data
        if (ai_src >= len(self.plots_per_axes)
                or j_src >= len(self.plots_per_axes[ai_src])):
            return
        target_idx = self.curve_axis_combo.currentIndex()
        if target_idx < 0:
            return
        n_axes = len(self.plots_per_axes)
        if target_idx == ai_src:
            self.app.log('Кривая уже на выбранной оси.')
            return
        p = self.plots_per_axes[ai_src].pop(j_src)
        if target_idx >= n_axes:
            self.plots_per_axes.append([p])
            self.axes_colors.append(None)
            self.y_limits.append([None, None])
            self.axis_labels.append(None)
            new_axis = len(self.plots_per_axes) - 1
        else:
            self.plots_per_axes[target_idx].append(p)
            new_axis = target_idx
        if ai_src > 0 and not self.plots_per_axes[ai_src]:
            self.plots_per_axes.pop(ai_src)
            self.axes_colors.pop(ai_src)
            self.y_limits.pop(ai_src)
            if ai_src < len(self.axis_labels):
                self.axis_labels.pop(ai_src)
            if ai_src < new_axis:
                new_axis -= 1
        self.redraw()
        self.app.log(
            f'Кривая «{p["label"]}» перемещена на ось №{new_axis + 1}.')

    # ---- Пределы осей ----
    def _refresh_limit_axis_combo(self):
        if not hasattr(self, 'limit_axis_combo'):
            return
        old = self.limit_axis_combo.currentIndex()
        self.limit_axis_combo.blockSignals(True)
        self.limit_axis_combo.clear()
        n = len(self.plots_per_axes)
        for k in range(n):
            self.limit_axis_combo.addItem(f'Ось {k + 1}')
        if 0 <= old < n:
            self.limit_axis_combo.setCurrentIndex(old)
        self.limit_axis_combo.blockSignals(False)
        self.sync_limit_edits()

    def sync_limit_edits(self):
        if not hasattr(self, 'y_min_edit'):
            return
        for e in (self.y_min_edit, self.y_max_edit,
                  self.x_min_edit, self.x_max_edit):
            e.clear()
        ai = self.limit_axis_combo.currentIndex()
        if ai < 0 or ai >= len(self.y_limits):
            return
        ymin, ymax = self.y_limits[ai]
        xmin, xmax = self.x_limits
        if ymin is not None:
            self.y_min_edit.setText(str(ymin))
        if ymax is not None:
            self.y_max_edit.setText(str(ymax))
        if xmin is not None:
            self.x_min_edit.setText(str(xmin))
        if xmax is not None:
            self.x_max_edit.setText(str(xmax))

    def apply_limits(self):
        ai = self.limit_axis_combo.currentIndex()
        if ai < 0 or ai >= len(self.y_limits):
            return
        ymn, ok1 = _parse_limit(self.y_min_edit.text())
        ymx, ok2 = _parse_limit(self.y_max_edit.text())
        xmn, ok3 = _parse_limit(self.x_min_edit.text())
        xmx, ok4 = _parse_limit(self.x_max_edit.text())
        if not (ok1 and ok2 and ok3 and ok4):
            self.app.log('Ошибка: некорректное число в поле предела.')
            return
        if ymn is not None and ymx is not None and ymn >= ymx:
            self.app.log('Y min должно быть меньше Y max.')
            return
        if xmn is not None and xmx is not None and xmn >= xmx:
            self.app.log('X min должно быть меньше X max.')
            return
        self.y_limits[ai] = [ymn, ymx]
        self.x_limits = [xmn, xmx]
        self.redraw()
        self.app.log(
            f'Пределы оси №{ai + 1} графика #{self.index} применены.')

    def reset_limits(self):
        ai = self.limit_axis_combo.currentIndex()
        if ai < 0 or ai >= len(self.y_limits):
            return
        self.y_limits[ai] = [None, None]
        self.x_limits = [None, None]
        self.redraw()
        self.sync_limit_edits()
        self.app.log(
            f'Пределы оси №{ai + 1} графика #{self.index} сброшены в авто.')


# ---------------------------------------------------------------------------
# Разбор TRC в фоновом потоке
# ---------------------------------------------------------------------------
class TRCWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    done_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        try:
            msgs = parse_can_log(
                self.filename,
                progress_cb=self.log_signal.emit,
                progress_value_cb=lambda v: self.progress_signal.emit(
                    min(100, max(0, int(v)))),
            )
            self.done_signal.emit(msgs)
        except Exception as e:
            self.error_signal.emit(str(e))


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TRC Signal Plotter — PyQtGraph')
        self.resize(1400, 1000)

        self.trc_file = ''
        self.messages = []
        self.id_map = {}
        self.id_count_map = {}
        self.dbc = None
        self.fig_counter = 0
        self.plot_windows = []
        self.custom_axis_color = None
        self.export_entries = []
        self._trc_worker = None

        self.prefs = load_prefs()
        self.last_trc_dir = self.prefs.get('lastTrcDir', os.getcwd())
        self.last_dbc_dir = self.prefs.get('lastDbcDir', os.getcwd())
        self.last_export_dir = self.prefs.get('lastExportDir', os.getcwd())

        self._build_ui()
        self.log('Приложение готово.')
        self.log('Шаг 1: выберите .trc файл → «Читать TRC».')
        self.log('Шаг 2: выберите .dbc файл.')
        self.log('Шаг 3: выберите сообщение/сигналы → «Построить график».')
        self.log('В окне графика можно редактировать заголовок, '
                 'подписи осей, цвет осей, легенду и видимость кривых.')
        self.log('Функции контекстного меню ПКМ — в панели справа от графика.')
        self.log('СКМ по графику — сброс пределов осей в «авто».')
        if not HAS_SCIPY:
            self.log('Внимание: scipy не установлен — экспорт .mat недоступен.')

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        top = QGridLayout()
        top.setColumnStretch(1, 1)

        top.addWidget(QLabel('TRC файл:'), 0, 0)
        self.trc_entry = QLineEdit()
        self.trc_entry.setReadOnly(True)
        self.trc_entry.setText('(не выбран)')
        top.addWidget(self.trc_entry, 0, 1)
        b = QPushButton('Обзор...')
        b.clicked.connect(self.on_browse_trc)
        top.addWidget(b, 0, 2)
        self.read_trc_btn = QPushButton('Читать TRC')
        self.read_trc_btn.setEnabled(False)
        self.read_trc_btn.clicked.connect(self.on_read_trc)
        top.addWidget(self.read_trc_btn, 0, 3)

        top.addWidget(QLabel('DBC файл:'), 1, 0)
        self.dbc_entry = QLineEdit()
        self.dbc_entry.setReadOnly(True)
        self.dbc_entry.setText('(не выбран)')
        top.addWidget(self.dbc_entry, 1, 1)
        b = QPushButton('Обзор...')
        b.clicked.connect(self.on_browse_dbc)
        top.addWidget(b, 1, 2)

        top.addWidget(QLabel('Прогресс чтения:'), 2, 0)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        top.addWidget(self.progress_bar, 2, 1, 1, 3)

        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QSplitter(Qt.Vertical)

        msg_box = QWidget()
        mb = QVBoxLayout(msg_box)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.addWidget(QLabel('Сообщения DBC (двойной клик по шапке — сортировка):'))
        self.msg_tree = QTreeWidget()
        self.msg_tree.setHeaderLabels(['ID', 'Name', 'Count'])
        self.msg_tree.setRootIsDecorated(False)
        self.msg_tree.setAlternatingRowColors(True)
        self.msg_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.msg_tree.header().sectionClicked.connect(self._sort_msg_tree)
        self.msg_tree.itemSelectionChanged.connect(self.on_msg_select)
        mb.addWidget(self.msg_tree)
        left.addWidget(msg_box)

        sig_box = QWidget()
        sb = QVBoxLayout(sig_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.addWidget(QLabel('Сигналы (Ctrl+клик — несколько):'))
        self.sig_tree = QTreeWidget()
        self.sig_tree.setHeaderLabels(['Signal', 'Unit'])
        self.sig_tree.setRootIsDecorated(False)
        self.sig_tree.setAlternatingRowColors(True)
        self.sig_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        sb.addWidget(self.sig_tree)
        left.addWidget(sig_box)

        splitter.addWidget(left)

        self.tabs = QTabWidget()
        self.tab_plot = QWidget()
        self.tab_export = QWidget()
        self.tabs.addTab(self.tab_plot, 'Построение графика')
        self.tabs.addTab(self.tab_export, 'Экспорт')
        self._build_plot_tab(self.tab_plot)
        self._build_export_tab(self.tab_export)
        splitter.addWidget(self.tabs)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        log_box = QGroupBox('Журнал')
        lb = QVBoxLayout(log_box)
        lb.setContentsMargins(4, 4, 4, 4)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(2000)
        self.log_text.setFont(QFont('Consolas', 9))
        lb.addWidget(self.log_text)
        root.addWidget(log_box)

    def _build_plot_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        build_box = QGroupBox('Построение')
        bb = QVBoxLayout(build_box)

        self.plot_btn = QPushButton('Построить график')
        self.plot_btn.setStyleSheet('font-weight: bold; padding: 6px;')
        self.plot_btn.clicked.connect(self.on_plot)
        bb.addWidget(self.plot_btn)

        self.add_mode_cb = QCheckBox('Добавить к выбранному графику')
        bb.addWidget(self.add_mode_cb)

        row = QHBoxLayout()
        row.addWidget(QLabel('Ось:'))
        self.add_coords_combo = QComboBox()
        self.add_coords_combo.addItem('Ось 1')
        row.addWidget(self.add_coords_combo, stretch=1)
        bb.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Окно для добавления:'))
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        row.addWidget(self.target_combo, stretch=1)
        bb.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Цвет новой оси:'))
        self.color_btn = QPushButton('Авто')
        self.color_btn.clicked.connect(self.on_pick_axis_color)
        row.addWidget(self.color_btn, stretch=1)
        b = QPushButton('Сброс')
        b.clicked.connect(self.on_reset_axis_color)
        row.addWidget(b)
        bb.addLayout(row)

        hint = QLabel('Масштабирование мышью: ЛКМ-протяжка = область; '
                      'Shift+ЛКМ = только по X; Ctrl+ЛКМ = только по Y; '
                      'ПКМ = панорамирование; СКМ = авто-пределы осей. '
                      'Фон графика и пределы осей — в окне графика.')
        hint.setStyleSheet('color: #777; font-size: 11px;')
        hint.setWordWrap(True)
        bb.addWidget(hint)

        layout.addWidget(build_box)

        svc_box = QGroupBox('Сервис')
        sb = QVBoxLayout(svc_box)

        b = QPushButton('Очистить выбор сигналов')
        b.clicked.connect(self.sig_tree.clearSelection)
        sb.addWidget(b)

        b = QPushButton('Закрыть все графики')
        b.clicked.connect(self.on_close_all_plots)
        sb.addWidget(b)

        b = QPushButton('Сбросить пути')
        b.clicked.connect(self.on_reset_paths)
        sb.addWidget(b)

        b = QPushButton('Очистить журнал')
        b.clicked.connect(self._clear_log)
        sb.addWidget(b)

        layout.addWidget(svc_box)
        layout.addStretch()

    def _build_export_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        b = QPushButton('→  Добавить выбранные сигналы в список экспорта')
        b.setStyleSheet('font-weight: bold; padding: 6px;')
        b.clicked.connect(self._add_signals_to_export)
        layout.addWidget(b)

        lbl = QLabel('(выберите сигналы в левом дереве, затем нажмите кнопку)')
        lbl.setStyleSheet('color: #777;')
        layout.addWidget(lbl)

        self.export_tree = QTreeWidget()
        self.export_tree.setHeaderLabels(['Message', 'Signal', 'Unit'])
        self.export_tree.setRootIsDecorated(False)
        self.export_tree.setAlternatingRowColors(True)
        self.export_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.export_tree, stretch=1)

        row = QHBoxLayout()
        self.export_count_label = QLabel('Сигналов в списке: 0')
        self.export_count_label.setStyleSheet('font-weight: bold;')
        row.addWidget(self.export_count_label)
        row.addStretch()
        b = QPushButton('← Удалить выбранные')
        b.clicked.connect(self._remove_from_export)
        row.addWidget(b)
        b = QPushButton('Очистить список')
        b.clicked.connect(self._clear_export_list)
        row.addWidget(b)
        layout.addLayout(row)

        settings = QGroupBox('Настройки экспорта')
        form = QFormLayout(settings)

        self.export_fmt = QComboBox()
        self.export_fmt.addItems([
            'CSV files (folder)',
            'NPZ (single file)',
            'MAT file (folder)',
        ])
        self.export_fmt.currentIndexChanged.connect(
            self._on_export_fmt_changed)
        form.addRow('Формат:', self.export_fmt)

        self.mat_type_combo = QComboBox()
        self.mat_type_combo.addItems(['double', 'timeseries'])
        form.addRow('Тип (MAT):', self.mat_type_combo)

        self.export_prefix = QLineEdit()
        form.addRow('Префикс:', self.export_prefix)

        self.mat_filename_edit = QLineEdit('signals')
        form.addRow('Имя .mat:', self.mat_filename_edit)

        layout.addWidget(settings)

        b = QPushButton('ЭКСПОРТИРОВАТЬ')
        b.setStyleSheet('font-weight: bold; padding: 8px;')
        b.clicked.connect(self.on_export)
        layout.addWidget(b)

        self._on_export_fmt_changed()

    def _on_export_fmt_changed(self):
        is_mat = self.export_fmt.currentText().startswith('MAT')
        self.mat_filename_edit.setEnabled(is_mat)
        self.mat_type_combo.setEnabled(is_mat)

    def log(self, msg):
        self.log_text.appendPlainText(msg)

    def _clear_log(self):
        if getattr(self, 'log_text', None) is not None:
            self.log_text.clear()

    def on_browse_trc(self):
        start = self.last_trc_dir if os.path.isdir(self.last_trc_dir) else ''
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите .trc файл', start,
            'TRC files (*.trc);;All files (*)')
        if not path:
            return
        self.last_trc_dir = os.path.dirname(path)
        self.prefs['lastTrcDir'] = self.last_trc_dir
        save_prefs(self.prefs)

        self.trc_file = path
        self.trc_entry.setText(path)
        self.messages = []
        self.id_map = {}
        self.id_count_map = {}
        self.read_trc_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.refresh_message_list()
        self.log(f'Выбран TRC: {path}')

    def on_read_trc(self):
        if not self.trc_file or not os.path.isfile(self.trc_file):
            self.log('TRC файл не выбран или не найден.')
            return
        self.log(f'Загрузка TRC: {self.trc_file} ...')
        self.read_trc_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self._trc_worker = TRCWorker(self.trc_file)
        self._trc_worker.log_signal.connect(self.log)
        self._trc_worker.progress_signal.connect(self.progress_bar.setValue)
        self._trc_worker.done_signal.connect(self._on_trc_done)
        self._trc_worker.error_signal.connect(self._on_trc_error)
        self._trc_worker.start()

    def _on_trc_done(self, msgs):
        self.read_trc_btn.setEnabled(True)
        self.messages = msgs
        n = len(self.messages)
        if n == 0:
            self.log('TRC разобран, но сообщений не найдено.')
            return
        self.log(f'TRC разобран: {n} сообщений.')

        groups = defaultdict(list)
        for k, msg in enumerate(self.messages):
            groups[msg['id_val']].append(k)
        self.id_map = dict(groups)
        self.id_count_map = {k: len(v) for k, v in groups.items()}
        self.log(f'Уникальных ID: {len(self.id_map)}.')
        self.refresh_message_list()

    def _on_trc_error(self, msg):
        self.read_trc_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log(f'Ошибка при разборе TRC: {msg}')
        self.messages = []
        self.id_map = {}
        self.id_count_map = {}
        self.refresh_message_list()

    def on_browse_dbc(self):
        start = self.last_dbc_dir if os.path.isdir(self.last_dbc_dir) else ''
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите .dbc файл', start,
            'DBC files (*.dbc);;All files (*)')
        if not path:
            return
        self.last_dbc_dir = os.path.dirname(path)
        self.prefs['lastDbcDir'] = self.last_dbc_dir
        save_prefs(self.prefs)

        self.dbc_entry.setText(path)
        self.log(f'Загрузка DBC: {path} ...')

        self.export_entries = []
        self._refresh_export_list()

        try:
            self.dbc = parse_dbc(path)
        except Exception as e:
            self.log(f'Ошибка при разборе DBC: {e}')
            self.dbc = None
            self.msg_tree.clear()
            self.sig_tree.clear()
            return
        self.log(f'DBC разобран: {len(self.dbc["messages"])} сообщений.')
        self.refresh_message_list()

    def refresh_message_list(self):
        self.msg_tree.clear()
        if self.dbc is None:
            self.sig_tree.clear()
            return
        have_table = bool(self.id_count_map)
        for i, m in enumerate(self.dbc['messages']):
            cnt = self.id_count_map.get(m['id'], 0) if have_table else 0
            item = QTreeWidgetItem([
                f"0x{m['id']:08X}", m['name'],
                str(cnt) if have_table else '-',
            ])
            item.setData(0, Qt.UserRole, i)
            self.msg_tree.addTopLevelItem(item)
        if self.msg_tree.topLevelItemCount() > 0:
            self.msg_tree.setCurrentItem(self.msg_tree.topLevelItem(0))
            self.on_msg_select()

    def _sort_msg_tree(self, col):
        if self.dbc is None or self.msg_tree.topLevelItemCount() == 0:
            return
        items = []
        for i in range(self.msg_tree.topLevelItemCount()):
            it = self.msg_tree.topLevelItem(i)
            items.append((it, [it.text(0), it.text(1), it.text(2)]))
        if col == 2:
            keyf = lambda x: (int(x[1][2])
                              if x[1][2].lstrip('-').isdigit() else 0)
        else:
            keyf = lambda x: x[1][col]
        items.sort(key=keyf)
        self.msg_tree.clear()
        for it, _ in items:
            self.msg_tree.addTopLevelItem(it)

    def on_msg_select(self):
        self.sig_tree.clear()
        if self.dbc is None:
            return
        item = self.msg_tree.currentItem()
        if item is None:
            return
        m_idx = item.data(0, Qt.UserRole)
        if m_idx is None or m_idx >= len(self.dbc['messages']):
            return
        sigs = self.dbc['messages'][m_idx]['signals']
        for si, s in enumerate(sigs):
            it = QTreeWidgetItem([s['name'], s['unit'] or ''])
            it.setData(0, Qt.UserRole, si)
            self.sig_tree.addTopLevelItem(it)

    def on_pick_axis_color(self):
        init = self.custom_axis_color or DEFAULT_COLOR
        c = QColorDialog.getColor(
            rgb_to_qcolor(init), self, 'Цвет новой оси')
        if not c.isValid():
            return
        self.custom_axis_color = (c.red() / 255.0, c.green() / 255.0,
                                  c.blue() / 255.0)
        self.color_btn.setText('Выбран')
        self.color_btn.setStyleSheet(
            f'background-color: {c.name()}; color: '
            f'{"black" if c.lightness() > 128 else "white"};')
        self.log('Цвет новой оси обновлён.')

    def on_reset_axis_color(self):
        self.custom_axis_color = None
        self.color_btn.setText('Авто')
        self.color_btn.setStyleSheet('')
        self.log('Цвет новой оси: авто.')

    def on_plot_window_closed(self, pw):
        if pw in self.plot_windows:
            self.plot_windows.remove(pw)
        try:
            pw.deleteLater()
        except Exception:
            pass
        self.refresh_target_list()
        self.log('Окно графика закрыто.')

    def refresh_target_list(self):
        old = self.target_combo.currentIndex()
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        if not self.plot_windows:
            self.target_combo.addItem('(нет открытых графиков)')
        else:
            for pw in self.plot_windows:
                self.target_combo.addItem(
                    f'График #{pw.index} — {pw.window_title}')
            if 0 <= old < len(self.plot_windows):
                self.target_combo.setCurrentIndex(old)
        self.target_combo.blockSignals(False)
        self.refresh_coord_list()

    def on_target_changed(self):
        self.refresh_coord_list()

    def get_current_plot_window(self):
        if not self.plot_windows:
            return None
        idx = self.target_combo.currentIndex()
        if idx < 0 or idx >= len(self.plot_windows):
            return None
        return self.plot_windows[idx]

    def refresh_coord_list(self):
        pw = self.get_current_plot_window()
        old = self.add_coords_combo.currentIndex()
        self.add_coords_combo.blockSignals(True)
        self.add_coords_combo.clear()
        if pw is None:
            self.add_coords_combo.addItem('Ось 1')
            self.add_coords_combo.addItem('Новая ось')
        else:
            for k in range(len(pw.plots_per_axes)):
                self.add_coords_combo.addItem(f'Ось {k + 1}')
            self.add_coords_combo.addItem('Новая ось')
            if 0 <= old < self.add_coords_combo.count():
                self.add_coords_combo.setCurrentIndex(old)
        self.add_coords_combo.blockSignals(False)

    def on_close_all_plots(self):
        n = len(self.plot_windows)
        for pw in list(self.plot_windows):
            try:
                pw.close()
            except Exception:
                pass
        self.plot_windows = []
        self.fig_counter = 0
        self.refresh_target_list()
        self.log(f'Закрыто графиков: {n}')

    def on_plot(self):
        if not self.messages:
            self.log('Сначала нажмите «Читать TRC».')
            return
        if self.dbc is None:
            self.log('Сначала загрузите DBC файл.')
            return
        item = self.msg_tree.currentItem()
        if item is None:
            self.log('Выберите сообщение.')
            return
        m_idx = item.data(0, Qt.UserRole)
        if m_idx is None or m_idx >= len(self.dbc['messages']):
            return
        msg_def = self.dbc['messages'][m_idx]

        s_sel = []
        for it in self.sig_tree.selectedItems():
            si = it.data(0, Qt.UserRole)
            if si is not None and 0 <= si < len(msg_def['signals']):
                s_sel.append(si)
        if not s_sel:
            self.log('Выберите хотя бы один сигнал.')
            return

        match_idx = self.id_map.get(msg_def['id'], [])
        if not match_idx:
            self.log(f'В TRC не найдено сообщений с ID '
                     f'0x{msg_def["id"]:08X}.')
            return

        new_plots = []
        for si in s_sel:
            sig = msg_def['signals'][si]
            n = len(match_idx)
            t_arr = np.zeros(n)
            v_arr = np.zeros(n)
            for k, mi in enumerate(match_idx):
                m = self.messages[mi]
                data = m['data']
                if sig['byteorder'] == '1':
                    v = extract_intel(data, sig['start'], sig['len'],
                                      sig['sign'] == '-',
                                      sig['scale'], sig['offset'])
                else:
                    v = extract_motorola(data, sig['start'], sig['len'],
                                         sig['sign'] == '-',
                                         sig['scale'], sig['offset'])
                t_arr[k] = m['time']
                v_arr[k] = v
            label = sig['name']
            if sig['unit']:
                label = f"{sig['name']} [{sig['unit']}]"

            vv = np.asarray(v_arr, dtype=float)
            vv = vv[np.isfinite(vv)]
            if len(vv):
                self.log(
                    f'  Сигнал «{sig["name"]}»: n={len(v_arr)}, '
                    f'min={vv.min():.4g}, max={vv.max():.4g}, '
                    f'mean={vv.mean():.4g}')

            new_plots.append({
                't': t_arr, 'v': v_arr, 'label': label,
                'msgName': msg_def['name'],
                'color': None, 'lineStyle': '-', 'lineWidth': 1.5,
                'marker': 'none', 'markerSize': 6,
                'visible': True,
            })

        add_mode = self.add_mode_cb.isChecked()
        target = self.get_current_plot_window() if add_mode else None

        if target is not None:
            coord_choice = self.add_coords_combo.currentIndex()
            n_axes = len(target.plots_per_axes)
            if coord_choice < 0 or coord_choice > n_axes:
                coord_choice = n_axes
            target.add_plots(new_plots, coord_choice, self.custom_axis_color)
            if coord_choice >= n_axes:
                self.log(f'Добавлено {len(new_plots)} сигнал(ов) в новую ось '
                         f'№{n_axes + 1} графика #{target.index}.')
            else:
                self.log(f'Добавлено {len(new_plots)} сигнал(ов) в ось '
                         f'№{coord_choice + 1} графика #{target.index}.')
            self.refresh_coord_list()
        else:
            self.fig_counter += 1
            title = f"График #{self.fig_counter} — {msg_def['name']}"
            pw = PlotWindow(self, self.fig_counter, title)
            pw.plots_per_axes = [new_plots]
            pw.axes_colors = [self.custom_axis_color]
            pw.y_limits = [[None, None]]
            pw.axis_labels = [None]
            pw.closed.connect(self.on_plot_window_closed)
            pw.state_changed.connect(self.refresh_target_list)
            self.plot_windows.append(pw)
            pw.show()
            pw.redraw()
            self.refresh_target_list()
            self.log(f'Открыт {title}: {len(new_plots)} сигнал(ов) '
                     f'по {len(match_idx)} сообщениям.')

    def on_reset_paths(self):
        self.last_trc_dir = os.getcwd()
        self.last_dbc_dir = os.getcwd()
        self.last_export_dir = os.getcwd()
        self.prefs.pop('lastTrcDir', None)
        self.prefs.pop('lastDbcDir', None)
        self.prefs.pop('lastExportDir', None)
        save_prefs(self.prefs)
        self.log('Сохранённые пути сброшены.')

    def _add_signals_to_export(self):
        if self.dbc is None:
            self.log('Сначала загрузите DBC.')
            return
        item = self.msg_tree.currentItem()
        if item is None:
            self.log('Выберите сообщение.')
            return
        m_idx = item.data(0, Qt.UserRole)
        if m_idx is None or m_idx >= len(self.dbc['messages']):
            return
        msg_def = self.dbc['messages'][m_idx]

        s_sel = []
        for it in self.sig_tree.selectedItems():
            si = it.data(0, Qt.UserRole)
            if si is not None and 0 <= si < len(msg_def['signals']):
                s_sel.append(si)
        if not s_sel:
            self.log('Выберите сигналы.')
            return

        existing = {(e['msg_idx'], e['sig_idx']) for e in self.export_entries}
        added = 0
        for si in s_sel:
            key = (m_idx, si)
            if key in existing:
                continue
            sig = msg_def['signals'][si]
            self.export_entries.append({
                'msg_idx': m_idx, 'sig_idx': si,
                'msg_name': msg_def['name'], 'msg_id': msg_def['id'],
                'sig_name': sig['name'], 'unit': sig['unit'],
            })
            added += 1
            existing.add(key)
        self._refresh_export_list()
        self.log(f'Добавлено в экспорт: {added}.')

    def _remove_from_export(self):
        sel_items = self.export_tree.selectedItems()
        if not sel_items:
            return
        to_remove = set()
        for it in sel_items:
            idx = it.data(0, Qt.UserRole)
            if idx is not None:
                to_remove.add(idx)
        self.export_entries = [e for i, e in enumerate(self.export_entries)
                               if i not in to_remove]
        self._refresh_export_list()
        self.log(f'Удалено из экспорта: {len(to_remove)}.')

    def _clear_export_list(self):
        n = len(self.export_entries)
        self.export_entries = []
        self._refresh_export_list()
        self.log(f'Список экспорта очищен ({n}).')

    def _refresh_export_list(self):
        self.export_tree.clear()
        for i, e in enumerate(self.export_entries):
            it = QTreeWidgetItem([e['msg_name'], e['sig_name'],
                                  e['unit'] or ''])
            it.setData(0, Qt.UserRole, i)
            self.export_tree.addTopLevelItem(it)
        self.export_count_label.setText(
            f'Сигналов в списке: {len(self.export_entries)}')

    def _collect_signals(self, entries, prefix):
        if self.dbc is None:
            return {}
        out = {}
        for e in entries:
            m_idx = e['msg_idx']
            si = e['sig_idx']
            if m_idx >= len(self.dbc['messages']):
                continue
            msg_def = self.dbc['messages'][m_idx]
            if si >= len(msg_def['signals']):
                continue
            sig = msg_def['signals'][si]
            match_idx = self.id_map.get(msg_def['id'], [])
            if not match_idx:
                self.log(f'  ! Нет данных: {msg_def["name"]}.{sig["name"]}')
                continue
            n = len(match_idx)
            t_arr = np.zeros(n)
            v_arr = np.zeros(n)
            for k, mi in enumerate(match_idx):
                m = self.messages[mi]
                data = m['data']
                if sig['byteorder'] == '1':
                    v = extract_intel(data, sig['start'], sig['len'],
                                      sig['sign'] == '-',
                                      sig['scale'], sig['offset'])
                else:
                    v = extract_motorola(data, sig['start'], sig['len'],
                                         sig['sign'] == '-',
                                         sig['scale'], sig['offset'])
                t_arr[k] = m['time']
                v_arr[k] = v
            name = sig['name']
            if prefix:
                name = f'{prefix}_{name}'
            var_name = sanitize_var_name(name)
            if var_name in out:
                var_name = sanitize_var_name(f'{var_name}_{msg_def["name"]}')
            out[var_name] = {
                'name': sig['name'], 'unit': sig['unit'],
                't': t_arr, 'v': v_arr,
            }
        return out

    def _initial_export_dir(self):
        d = self.last_export_dir
        return d if os.path.isdir(d) else os.getcwd()

    def on_export(self):
        if not self.messages:
            self.log('Сначала нажмите «Читать TRC».')
            return
        if self.dbc is None:
            self.log('Сначала загрузите DBC.')
            return
        if not self.export_entries:
            self.log('Список экспорта пуст.')
            return
        prefix = self.export_prefix.text().strip()
        data_out = self._collect_signals(self.export_entries, prefix)
        if not data_out:
            self.log('Нечего экспортировать.')
            return
        fmt = self.export_fmt.currentText()
        if fmt.startswith('CSV'):
            self._export_csv(data_out)
        elif fmt.startswith('NPZ'):
            self._export_npz(data_out)
        elif fmt.startswith('MAT'):
            self._export_mat(data_out)

    def _export_csv(self, data_out):
        folder = QFileDialog.getExistingDirectory(
            self, 'Папка для CSV', self._initial_export_dir())
        if not folder:
            return
        self.last_export_dir = folder
        self.prefs['lastExportDir'] = folder
        save_prefs(self.prefs)
        for var, d in data_out.items():
            path = os.path.join(folder, f'{var}.csv')
            arr = np.column_stack([d['t'], d['v']])
            try:
                np.savetxt(path, arr, delimiter=',',
                           header='time,value', comments='', fmt='%.9g')
                self.log(f'  • {var}.csv ({len(d["t"])} точек)')
            except OSError as e:
                self.log(f'  ! Ошибка: {e}')
        QMessageBox.information(self, 'Готово',
                                f'CSV сохранено: {len(data_out)}')

    def _export_npz(self, data_out):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Сохранить NPZ', self._initial_export_dir(),
            'NumPy archive (*.npz)')
        if not path:
            return
        if not path.lower().endswith('.npz'):
            path += '.npz'
        self.last_export_dir = os.path.dirname(path)
        self.prefs['lastExportDir'] = self.last_export_dir
        save_prefs(self.prefs)
        payload = {}
        for var, d in data_out.items():
            payload[var + '_time'] = d['t']
            payload[var + '_value'] = d['v']
            payload[var + '_name'] = d['name']
            payload[var + '_unit'] = d['unit']
        try:
            np.savez(path, **payload)
            self.log(f'Сохранено: {path}')
            QMessageBox.information(self, 'Готово',
                                    f'Экспортировано: {len(data_out)}')
        except OSError as e:
            self.log(f'Ошибка: {e}')

    def _export_mat(self, data_out):
        if not HAS_SCIPY:
            QMessageBox.critical(self, 'Нет scipy',
                                 'Установите: pip install scipy')
            return
        folder = QFileDialog.getExistingDirectory(
            self, 'Папка для .mat', self._initial_export_dir())
        if not folder:
            return
        self.last_export_dir = folder
        self.prefs['lastExportDir'] = folder
        save_prefs(self.prefs)

        fname = sanitize_filename(
            self.mat_filename_edit.text(), default='signals', ext='.mat')
        path = os.path.join(folder, fname)
        base_name = os.path.splitext(fname)[0]
        dtype = self.mat_type_combo.currentText()

        if dtype == 'double':
            mat_dict = {}
            for var, d in data_out.items():
                mat_dict[var] = {
                    'time': np.asarray(d['t'], dtype=float),
                    'value': np.asarray(d['v'], dtype=float),
                    'name': d['name'],
                    'unit': d['unit'],
                }
            try:
                scipy_savemat(path, mat_dict, do_compression=True)
            except Exception as e:
                self.log(f'Ошибка .mat: {e}')
                return
            self.log(f'Сохранено: {path} [double]')
            QMessageBox.information(
                self, 'Готово',
                f'Файл: {fname}\nСигналов: {len(data_out)}\n'
                f'Тип: double')
        else:
            mat_dict = {}
            for var, d in data_out.items():
                mat_dict[var] = {
                    'Time': np.asarray(d['t'], dtype=float),
                    'Data': np.asarray(d['v'], dtype=float),
                    'Name': d['name'],
                    'Unit': d['unit'],
                }
            try:
                scipy_savemat(path, mat_dict, do_compression=True)
            except Exception as e:
                self.log(f'Ошибка .mat: {e}')
                return

            m_path = os.path.join(folder, base_name + '_load.m')
            try:
                with open(m_path, 'w', encoding='utf-8') as fm:
                    fm.write(f"% Auto-generated\nS = load('{fname}');\n")
                    for var in data_out:
                        fm.write(
                            f'\n{var} = timeseries(S.{var}.Data, '
                            f'S.{var}.Time);\n'
                            f'{var}.Name = S.{var}.Name;\n'
                        )
                    fm.write('\nclear S;\n')
            except OSError:
                pass
            self.log(f'Сохранено: {path} [timeseries]')
            QMessageBox.information(
                self, 'Готово',
                f'Файл: {fname}\nСигналов: {len(data_out)}\n'
                f'Тип: timeseries\nСкрипт: {base_name}_load.m')


# ---------------------------------------------------------------------------
def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()