import pandas as pd
import numpy as np
from typing import Dict, Tuple

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib не установлен. Используются собственные реализации индикаторов.")
    print("Для установки TA-Lib: pip install TA-Lib")

class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Простая скользящая средняя"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.SMA(data.values, timeperiod=period), index=data.index)
        else:
            return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Экспоненциальная скользящая средняя"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.EMA(data.values, timeperiod=period), index=data.index)
        else:
            return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Индекс относительной силы"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)
        else:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD индикатор"""
        if TALIB_AVAILABLE:
            macd_line, macd_signal, macd_hist = talib.MACD(data.values, fastperiod=fast, 
                                                          slowperiod=slow, signalperiod=signal)
            return (pd.Series(macd_line, index=data.index),
                   pd.Series(macd_signal, index=data.index),
                   pd.Series(macd_hist, index=data.index))
        else:
            ema_fast = TechnicalIndicators.ema(data, fast)
            ema_slow = TechnicalIndicators.ema(data, slow)
            macd_line = ema_fast - ema_slow
            macd_signal = TechnicalIndicators.ema(macd_line, signal)
            macd_hist = macd_line - macd_signal
            return macd_line, macd_signal, macd_hist
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Полосы Боллинджера"""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(data.values, timeperiod=period, 
                                              nbdevup=std, nbdevdn=std)
            return (pd.Series(upper, index=data.index),
                   pd.Series(middle, index=data.index),
                   pd.Series(lower, index=data.index))
        else:
            sma = TechnicalIndicators.sma(data, period)
            std_dev = data.rolling(window=period).std()
            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)
            return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), 
                           index=close.index)
        else:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Стохастический осциллятор"""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values,
                                     fastk_period=k_period, slowk_period=3, 
                                     slowk_matype=0, slowd_period=d_period, slowd_matype=0)
            return (pd.Series(slowk, index=close.index),
                   pd.Series(slowd, index=close.index))
        else:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=period), 
                           index=close.index)
        else:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return wr
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.CCI(high.values, low.values, close.values, timeperiod=period), 
                           index=close.index)
        else:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), 
                           index=close.index)
        else:
            # Упрощенная реализация ADX
            tr = TechnicalIndicators.atr(high, low, close, 1)
            
            dm_plus = high.diff()
            dm_minus = -low.diff()
            
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            di_plus = 100 * (dm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum())
            di_minus = 100 * (dm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum())
            
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=period).mean()
            
            return adx
    
    @staticmethod
    def volume_spike(volume: pd.Series, period: int = 20, threshold: float = 1.5) -> pd.Series:
        """Определение всплесков объема"""
        volume_ma = volume.rolling(window=period).mean()
        return volume > (volume_ma * threshold)
    
    @staticmethod
    def support_resistance(data: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Уровни поддержки и сопротивления"""
        resistance = data.rolling(window=window, center=True).max()
        support = data.rolling(window=window, center=True).min()
        return support, resistance
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Точки разворота (Pivot Points)"""
        pivot = (high + low + close) / 3
        
        # Уровни сопротивления
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        # Уровни поддержки
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, 
                 tenkan_period: int = 9, kijun_period: int = 26, 
                 senkou_b_period: int = 52) -> Dict[str, pd.Series]:
        """Облако Ишимоку"""
        # Tenkan-sen (быстрая линия)
        tenkan_sen = (high.rolling(window=tenkan_period).max() + 
                     low.rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (медленная линия)
        kijun_sen = (high.rolling(window=kijun_period).max() + 
                    low.rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (первая граница облака)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Senkou Span B (вторая граница облака)
        senkou_span_b = ((high.rolling(window=senkou_b_period).max() + 
                         low.rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)
        
        # Chikou Span (запаздывающая линия)
        chikou_span = close.shift(-kijun_period)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        else:
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period), 
                           index=close.index)
        else:
            typical_price = (high + low + close) / 3
            raw_money_flow = typical_price * volume
            
            positive_flow = pd.Series(0.0, index=close.index)
            negative_flow = pd.Series(0.0, index=close.index)
            
            for i in range(1, len(typical_price)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = raw_money_flow.iloc[i]
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    negative_flow.iloc[i] = raw_money_flow.iloc[i]
            
            positive_money_flow = positive_flow.rolling(window=period).sum()
            negative_money_flow = negative_flow.rolling(window=period).sum()
            
            money_ratio = positive_money_flow / negative_money_flow
            mfi = 100 - (100 / (1 + money_ratio))
            
            return mfi


# Пример использования и тестирования
if __name__ == "__main__":
    # Создаем тестовые данные
    print("🔄 Тестирование технических индикаторов...")
    
    # Генерируем случайные данные для теста
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    
    # Имитируем ценовые данные
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    # Тестируем индикаторы
    indicators = TechnicalIndicators()
    
    print("\n📊 Расчет основных индикаторов...")
    
    # Базовые индикаторы
    df['sma_20'] = indicators.sma(df['close'], 20)
    df['ema_20'] = indicators.ema(df['close'], 20)
    df['rsi'] = indicators.rsi(df['close'])
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = indicators.macd(df['close'])
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = indicators.bollinger_bands(df['close'])
    
    # ATR
    df['atr'] = indicators.atr(df['high'], df['low'], df['close'])
    
    # Volume indicators
    df['volume_spike'] = indicators.volume_spike(df['volume'])
    df['obv'] = indicators.obv(df['close'], df['volume'])
    
    print("✅ Индикаторы рассчитаны успешно!")
    
    # Показываем последние значения
    print("\n📈 Последние значения индикаторов:")
    last_row = df.iloc[-1]
    print(f"Цена: ${last_row['close']:.2f}")
    print(f"SMA(20): ${last_row['sma_20']:.2f}")
    print(f"EMA(20): ${last_row['ema_20']:.2f}")
    print(f"RSI: {last_row['rsi']:.1f}")
    print(f"MACD: {last_row['macd']:.3f}")
    print(f"BB Upper: ${last_row['bb_upper']:.2f}")
    print(f"BB Lower: ${last_row['bb_lower']:.2f}")
    print(f"ATR: {last_row['atr']:.2f}")
    
    print(f"\n🔧 TA-Lib доступен: {'Да' if TALIB_AVAILABLE else 'Нет'}")
    print("✅ Тестирование завершено!")