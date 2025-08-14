import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from indicators import TechnicalIndicators

@dataclass
class TradeSignal:
    """Класс для торгового сигнала"""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    confidence: float  # 0.0 - 1.0
    strategy: str
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class BaseStrategy(ABC):
    """Базовый класс для всех торговых стратегий"""
    
    def __init__(self, name: str):
        self.name = name
        self.indicators = TechnicalIndicators()
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        """Генерация торговых сигналов"""
        pass

class TradingStrategies:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
        # ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ для большего количества сигналов
        self.params = {
            'mean_reversion': {
                'rsi_oversold': 35,      # было 30 - более чувствительно
                'rsi_overbought': 65,    # было 70 - более чувствительно
                'rsi_period': 14,
                'bb_period': 20,
                'bb_std': 1.8,           # было 2.0 - чаще касания границ
                'volume_threshold': 1.3  # было 2.0 - меньше требования к объему
            },
            'trend_following': {
                'ema_fast': 8,           # было 12 - быстрее реакция
                'ema_slow': 21,          # было 26 - быстрее реакция
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'adx_threshold': 20,     # было 25 - ловим начало трендов
                'volume_threshold': 1.2  # было 1.5 - мягче требования
            },
            'breakout': {
                'bb_period': 20,
                'bb_std': 1.5,           # было 2.0 - больше сигналов
                'volume_spike': 1.5,     # было 2.5 - чаще срабатывает
                'atr_period': 14,
                'atr_multiplier': 1.0,   # было 1.5 - ближе стопы
                'support_resistance_period': 20,
                'min_touches': 2         # было 3 - меньше подтверждений
            },
            'scalping': {
                'rsi_period': 7,         # короткий период для скальпинга
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'ema_fast': 5,
                'ema_slow': 10,
                'volume_threshold': 1.1,
                'min_volatility': 0.3    # минимальная волатильность в %
            },
            'momentum': {
                'lookback': 10,          # период для расчета моментума
                'threshold': 2.0,        # порог в % для входа
                'volume_confirm': 1.3,
                'rsi_filter': 45         # не входим в перекупленность
            }
        }
        
    def calculate_rsi(self, df, period=14):
        """Расчет RSI"""
        return self.indicators.rsi(df['close'], period)
    
    def calculate_bollinger_bands(self, df, period=20, std=2):
        """Расчет полос Боллинджера"""
        upper, middle, lower = self.indicators.bollinger_bands(df['close'], period, std)
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Расчет MACD"""
        macd, macd_signal, histogram = self.indicators.macd(df['close'], fast, slow, signal)
        return {'macd': macd, 'signal': macd_signal, 'histogram': histogram}
    
    def calculate_adx(self, df, period=14):
        """Расчет ADX"""
        return self.indicators.adx(df['high'], df['low'], df['close'], period)
    
    def calculate_atr(self, df, period=14):
        """Расчет ATR"""
        return self.indicators.atr(df['high'], df['low'], df['close'], period)
    
    def analyze_market_conditions(self, df):
        """Анализ рыночных условий"""
        if len(df) < 50:
            return {'trend': 'UNKNOWN', 'volatility': 'UNKNOWN', 'volume': 'UNKNOWN'}
        
        # Анализ тренда
        ema_short = df['close'].ewm(span=20).mean().iloc[-1]
        ema_long = df['close'].ewm(span=50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price > ema_short > ema_long:
            trend = 'UPTREND'
        elif current_price < ema_short < ema_long:
            trend = 'DOWNTREND'
        else:
            trend = 'SIDEWAYS'
        
        # Анализ волатильности
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * 100
        if volatility > 5:
            vol_level = 'HIGH'
        elif volatility > 2:
            vol_level = 'MEDIUM'
        else:
            vol_level = 'LOW'
        
        # Анализ объема
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        if current_volume > volume_ma * 1.5:
            volume_level = 'HIGH'
        elif current_volume > volume_ma * 0.7:
            volume_level = 'NORMAL'
        else:
            volume_level = 'LOW'
        
        return {
            'trend': trend,
            'volatility': vol_level,
            'volume': volume_level,
            'recommended_strategies': self._get_recommended_strategies(trend, vol_level, volume_level)
        }
    
    def _get_recommended_strategies(self, trend, volatility, volume):
        """Рекомендации стратегий по рыночным условиям"""
        recommendations = []
        
        if trend == 'UPTREND' or trend == 'DOWNTREND':
            recommendations.append('trend_following')
        
        if volatility == 'HIGH':
            recommendations.append('breakout')
            recommendations.append('scalping')
        
        if trend == 'SIDEWAYS':
            recommendations.append('mean_reversion')
        
        if volatility == 'LOW':
            recommendations.append('momentum')
        
        return recommendations if recommendations else ['mean_reversion']
        
    def mean_reversion_strategy(self, df):
        """Стратегия возврата к среднему - ОПТИМИЗИРОВАННАЯ"""
        params = self.params['mean_reversion']
        signals = []
        
        # Расчет индикаторов
        df['RSI'] = self.calculate_rsi(df, params['rsi_period'])
        bb_data = self.calculate_bollinger_bands(df, params['bb_period'], params['bb_std'])
        df['BB_upper'] = bb_data['upper']
        df['BB_lower'] = bb_data['lower']
        df['BB_middle'] = bb_data['middle']
        
        # Расчет среднего объема
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Дополнительный фильтр - не торгуем во флете
        df['volatility'] = (df['high'] - df['low']) / df['close'] * 100
        df['avg_volatility'] = df['volatility'].rolling(window=10).mean()
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Условия для LONG
            if (current['RSI'] < params['rsi_oversold'] and
                current['close'] < current['BB_lower'] and
                current['volume'] > current['volume_ma'] * params['volume_threshold'] and
                current['avg_volatility'] > 0.5):  # Добавлен фильтр волатильности
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'BUY',
                    'price': current['close'],
                    'rsi': current['RSI'],
                    'reason': 'Oversold + Below BB',
                    'confidence': min(90, 50 + (params['rsi_oversold'] - current['RSI']) * 2),
                    'strategy': 'mean_reversion'
                })
            
            # Условия для SHORT
            elif (current['RSI'] > params['rsi_overbought'] and
                  current['close'] > current['BB_upper'] and
                  current['volume'] > current['volume_ma'] * params['volume_threshold'] and
                  current['avg_volatility'] > 0.5):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'SELL',
                    'price': current['close'],
                    'rsi': current['RSI'],
                    'reason': 'Overbought + Above BB',
                    'confidence': min(90, 50 + (current['RSI'] - params['rsi_overbought']) * 2),
                    'strategy': 'mean_reversion'
                })
        
        return signals
    
    def trend_following_strategy(self, df):
        """Стратегия следования за трендом - ОПТИМИЗИРОВАННАЯ"""
        params = self.params['trend_following']
        signals = []
        
        # Расчет индикаторов
        df['EMA_fast'] = df['close'].ewm(span=params['ema_fast']).mean()
        df['EMA_slow'] = df['close'].ewm(span=params['ema_slow']).mean()
        
        macd_data = self.calculate_macd(df, params['macd_fast'], params['macd_slow'], params['macd_signal'])
        df['MACD'] = macd_data['macd']
        df['MACD_signal'] = macd_data['signal']
        df['MACD_hist'] = macd_data['histogram']
        
        df['ADX'] = self.calculate_adx(df)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Добавляем фильтр по наклону EMA
        df['EMA_slope'] = (df['EMA_fast'] - df['EMA_fast'].shift(3)) / df['EMA_fast'].shift(3) * 100
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # LONG сигнал
            if (prev['EMA_fast'] <= prev['EMA_slow'] and
                current['EMA_fast'] > current['EMA_slow'] and
                current['MACD'] > current['MACD_signal'] and
                current['ADX'] > params['adx_threshold'] and
                current['volume'] > current['volume_ma'] * params['volume_threshold'] and
                current['EMA_slope'] > 0.1):  # Тренд должен быть восходящим
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'BUY',
                    'price': current['close'],
                    'adx': current['ADX'],
                    'reason': 'Trend Start + MACD Cross',
                    'confidence': min(95, 60 + current['ADX']),
                    'strategy': 'trend_following'
                })
            
            # SHORT сигнал
            elif (prev['EMA_fast'] >= prev['EMA_slow'] and
                  current['EMA_fast'] < current['EMA_slow'] and
                  current['MACD'] < current['MACD_signal'] and
                  current['ADX'] > params['adx_threshold'] and
                  current['volume'] > current['volume_ma'] * params['volume_threshold'] and
                  current['EMA_slope'] < -0.1):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'SELL',
                    'price': current['close'],
                    'adx': current['ADX'],
                    'reason': 'Trend Reversal + MACD Cross',
                    'confidence': min(95, 60 + current['ADX']),
                    'strategy': 'trend_following'
                })
        
        return signals
    
    def volatility_breakout_strategy(self, df):
        """Стратегия пробоя волатильности - ОПТИМИЗИРОВАННАЯ"""
        params = self.params['breakout']
        signals = []
        
        # Расчет индикаторов
        bb_data = self.calculate_bollinger_bands(df, params['bb_period'], params['bb_std'])
        df['BB_upper'] = bb_data['upper']
        df['BB_lower'] = bb_data['lower']
        df['BB_width'] = bb_data['upper'] - bb_data['lower']
        
        df['ATR'] = self.calculate_atr(df, params['atr_period'])
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Поиск уровней поддержки/сопротивления
        df['resistance'] = df['high'].rolling(window=params['support_resistance_period']).max()
        df['support'] = df['low'].rolling(window=params['support_resistance_period']).min()
        
        # Сжатие Bollinger Bands (предшествует пробою)
        df['BB_squeeze'] = df['BB_width'] / df['close'] * 100
        df['BB_squeeze_ma'] = df['BB_squeeze'].rolling(window=20).mean()
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Проверка на сжатие волатильности
            volatility_squeeze = current['BB_squeeze'] < current['BB_squeeze_ma'] * 0.8
            
            # LONG - пробой вверх
            if (current['close'] > prev['BB_upper'] and
                prev['close'] <= prev['BB_upper'] and
                current['volume'] > current['volume_ma'] * params['volume_spike'] and
                volatility_squeeze and
                current['close'] > current['resistance'] * 0.995):  # Близко к сопротивлению
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'BUY',
                    'price': current['close'],
                    'stop_loss': current['close'] - current['ATR'] * params['atr_multiplier'],
                    'take_profit': current['close'] + current['ATR'] * params['atr_multiplier'] * 2,
                    'reason': 'Volatility Breakout UP',
                    'confidence': min(85, 60 + (current['volume'] / current['volume_ma'] - 1) * 10),
                    'strategy': 'breakout'
                })
            
            # SHORT - пробой вниз
            elif (current['close'] < prev['BB_lower'] and
                  prev['close'] >= prev['BB_lower'] and
                  current['volume'] > current['volume_ma'] * params['volume_spike'] and
                  volatility_squeeze and
                  current['close'] < current['support'] * 1.005):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'SELL',
                    'price': current['close'],
                    'stop_loss': current['close'] + current['ATR'] * params['atr_multiplier'],
                    'take_profit': current['close'] - current['ATR'] * params['atr_multiplier'] * 2,
                    'reason': 'Volatility Breakout DOWN',
                    'confidence': min(85, 60 + (current['volume'] / current['volume_ma'] - 1) * 10),
                    'strategy': 'breakout'
                })
        
        return signals
    
    def scalping_strategy(self, df):
        """Новая стратегия - Скальпинг на коротких таймфреймах"""
        params = self.params['scalping']
        signals = []
        
        # Быстрые индикаторы для скальпинга
        df['RSI'] = self.calculate_rsi(df, params['rsi_period'])
        df['EMA_fast'] = df['close'].ewm(span=params['ema_fast']).mean()
        df['EMA_slow'] = df['close'].ewm(span=params['ema_slow']).mean()
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        
        # Волатильность для фильтрации
        df['volatility'] = (df['high'] - df['low']) / df['close'] * 100
        
        for i in range(20, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Быстрый LONG
            if (current['RSI'] < params['rsi_oversold'] and
                current['EMA_fast'] > current['EMA_slow'] and
                prev['EMA_fast'] <= prev['EMA_slow'] and
                current['volatility'] > params['min_volatility'] and
                current['volume'] > current['volume_ma'] * params['volume_threshold']):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'BUY',
                    'price': current['close'],
                    'stop_loss': current['close'] * 0.997,  # Tight stop для скальпинга
                    'take_profit': current['close'] * 1.005,  # Быстрая прибыль
                    'reason': 'Scalping BUY',
                    'confidence': 70,
                    'strategy': 'scalping'
                })
            
            # Быстрый SHORT
            elif (current['RSI'] > params['rsi_overbought'] and
                  current['EMA_fast'] < current['EMA_slow'] and
                  prev['EMA_fast'] >= prev['EMA_slow'] and
                  current['volatility'] > params['min_volatility'] and
                  current['volume'] > current['volume_ma'] * params['volume_threshold']):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'SELL',
                    'price': current['close'],
                    'stop_loss': current['close'] * 1.003,
                    'take_profit': current['close'] * 0.995,
                    'reason': 'Scalping SELL',
                    'confidence': 70,
                    'strategy': 'scalping'
                })
        
        return signals
    
    def momentum_strategy(self, df):
        """Новая стратегия - Momentum (импульсная торговля)"""
        params = self.params['momentum']
        signals = []
        
        # Расчет моментума
        df['momentum'] = (df['close'] - df['close'].shift(params['lookback'])) / df['close'].shift(params['lookback']) * 100
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df, 14)
        
        # Скользящее среднее моментума
        df['momentum_ma'] = df['momentum'].rolling(window=5).mean()
        
        for i in range(params['lookback'] + 10, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Сильный импульс вверх
            if (current['momentum'] > params['threshold'] and
                current['momentum'] > current['momentum_ma'] and
                current['RSI'] < 70 and  # Не перекуплено
                current['volume'] > current['volume_ma'] * params['volume_confirm']):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'BUY',
                    'price': current['close'],
                    'momentum': current['momentum'],
                    'reason': f'Strong Momentum +{current["momentum"]:.2f}%',
                    'confidence': min(90, 70 + current['momentum'] * 2),
                    'strategy': 'momentum'
                })
            
            # Сильный импульс вниз
            elif (current['momentum'] < -params['threshold'] and
                  current['momentum'] < current['momentum_ma'] and
                  current['RSI'] > 30 and  # Не перепродано
                  current['volume'] > current['volume_ma'] * params['volume_confirm']):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'SELL',
                    'price': current['close'],
                    'momentum': current['momentum'],
                    'reason': f'Strong Momentum {current["momentum"]:.2f}%',
                    'confidence': min(90, 70 + abs(current['momentum']) * 2),
                    'strategy': 'momentum'
                })
        
        return signals
    
    def combine_signals(self, df, strategies=['all']):
        """Комбинирование сигналов от разных стратегий"""
        all_signals = []
        
        if 'all' in strategies:
            strategies = ['mean_reversion', 'trend_following', 'breakout', 'scalping', 'momentum']
        
        strategy_map = {
            'mean_reversion': self.mean_reversion_strategy,
            'trend_following': self.trend_following_strategy,
            'breakout': self.volatility_breakout_strategy,
            'scalping': self.scalping_strategy,
            'momentum': self.momentum_strategy
        }
        
        for strategy_name in strategies:
            if strategy_name in strategy_map:
                try:
                    strategy_signals = strategy_map[strategy_name](df.copy())
                    for signal in strategy_signals:
                        signal['strategy'] = strategy_name
                        all_signals.append(signal)
                except Exception as e:
                    print(f"Ошибка в стратегии {strategy_name}: {e}")
                    continue
        
        # Сортируем по времени
        all_signals.sort(key=lambda x: x['timestamp'])
        
        return all_signals


# Дополнительные стратегии для совместимости
class EMACrossoverStrategy(BaseStrategy):
    """Стратегия пересечения EMA"""
    
    def __init__(self, fast_ema: int = 12, slow_ema: int = 26):
        super().__init__("EMA_Crossover")
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        
        # Расчет EMA
        df['ema_fast'] = self.indicators.ema(df['close'], self.fast_ema)
        df['ema_slow'] = self.indicators.ema(df['close'], self.slow_ema)
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Пересечение вверх
            if (previous['ema_fast'] <= previous['ema_slow'] and 
                current['ema_fast'] > current['ema_slow']):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='BUY',
                    price=current['close'],
                    confidence=0.7,
                    strategy=self.name,
                    reason=f"EMA({self.fast_ema}) пересекла EMA({self.slow_ema}) вверх"
                ))
            
            # Пересечение вниз
            elif (previous['ema_fast'] >= previous['ema_slow'] and 
                  current['ema_fast'] < current['ema_slow']):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='SELL',
                    price=current['close'],
                    confidence=0.7,
                    strategy=self.name,
                    reason=f"EMA({self.fast_ema}) пересекла EMA({self.slow_ema}) вниз"
                ))
        
        return signals


class MACDBollingerStrategy(BaseStrategy):
    """Стратегия MACD + Bollinger Bands"""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2):
        super().__init__("MACD_Bollinger")
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        
        # Расчет индикаторов
        df['macd'], df['macd_signal'], df['macd_hist'] = self.indicators.macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.indicators.bollinger_bands(df['close'], self.bb_period, self.bb_std)
        
        for i in range(26, len(df)):  # Начинаем с 26 для стабильности MACD
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Сигнал на покупку
            if (current['macd'] > current['macd_signal'] and
                previous['macd'] <= previous['macd_signal'] and
                current['close'] < current['bb_upper']):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='BUY',
                    price=current['close'],
                    confidence=0.75,
                    strategy=self.name,
                    reason="MACD пересечение + цена ниже верхней BB"
                ))
            
            # Сигнал на продажу
            elif (current['macd'] < current['macd_signal'] and
                  previous['macd'] >= previous['macd_signal'] and
                  current['close'] > current['bb_lower']):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='SELL',
                    price=current['close'],
                    confidence=0.75,
                    strategy=self.name,
                    reason="MACD пересечение вниз + цена выше нижней BB"
                ))
        
        return signals


class ComboStrategy(BaseStrategy):
    """Комбинированная стратегия"""
    
    def __init__(self, trend_ema: int = 50, fast_ema: int = 12, slow_ema: int = 26):
        super().__init__("Combo_Strategy")
        self.trend_ema = trend_ema
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        
        # Расчет индикаторов
        df['trend_ema'] = self.indicators.ema(df['close'], self.trend_ema)
        df['fast_ema'] = self.indicators.ema(df['close'], self.fast_ema)
        df['slow_ema'] = self.indicators.ema(df['close'], self.slow_ema)
        df['rsi'] = self.indicators.rsi(df['close'])
        df['macd'], df['macd_signal'], _ = self.indicators.macd(df['close'])
        
        for i in range(self.trend_ema, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Определение тренда
            uptrend = current['close'] > current['trend_ema']
            
            # Сигнал на покупку
            if (uptrend and
                previous['fast_ema'] <= previous['slow_ema'] and
                current['fast_ema'] > current['slow_ema'] and
                current['rsi'] < 70 and
                current['macd'] > current['macd_signal']):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='BUY',
                    price=current['close'],
                    confidence=0.8,
                    strategy=self.name,
                    reason="Восходящий тренд + пересечение EMA + подтверждение индикаторов"
                ))
            
            # Сигнал на продажу
            elif (not uptrend and
                  previous['fast_ema'] >= previous['slow_ema'] and
                  current['fast_ema'] < current['slow_ema'] and
                  current['rsi'] > 30 and
                  current['macd'] < current['macd_signal']):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='SELL',
                    price=current['close'],
                    confidence=0.8,
                    strategy=self.name,
                    reason="Нисходящий тренд + пересечение EMA + подтверждение индикаторов"
                ))
        
        return signals


class BreakoutStrategy(BaseStrategy):
    """Стратегия пробоя"""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__("Breakout_Strategy")
        self.lookback_period = lookback_period
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        
        # Расчет уровней поддержки и сопротивления
        df['resistance'] = df['high'].rolling(window=self.lookback_period).max()
        df['support'] = df['low'].rolling(window=self.lookback_period).min()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        for i in range(self.lookback_period, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Пробой сопротивления
            if (current['close'] > current['resistance'] and
                previous['close'] <= previous['resistance'] and
                current['volume'] > current['volume_ma'] * 1.5):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='BUY',
                    price=current['close'],
                    confidence=0.75,
                    strategy=self.name,
                    reason=f"Пробой сопротивления {current['resistance']:.2f} с объемом"
                ))
            
            # Пробой поддержки
            elif (current['close'] < current['support'] and
                  previous['close'] >= previous['support'] and
                  current['volume'] > current['volume_ma'] * 1.5):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='SELL',
                    price=current['close'],
                    confidence=0.75,
                    strategy=self.name,
                    reason=f"Пробой поддержки {current['support']:.2f} с объемом"
                ))
        
        return signals


class MeanReversionStrategy(BaseStrategy):
    """Стратегия возврата к среднему"""
    
    def __init__(self, bb_period: int = 20, rsi_period: int = 14):
        super().__init__("Mean_Reversion")
        self.bb_period = bb_period
        self.rsi_period = rsi_period
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        
        # Расчет индикаторов
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.indicators.bollinger_bands(df['close'], self.bb_period)
        df['rsi'] = self.indicators.rsi(df['close'], self.rsi_period)
        
        for i in range(max(self.bb_period, self.rsi_period), len(df)):
            current = df.iloc[i]
            
            # Перепроданность
            if (current['close'] < current['bb_lower'] and
                current['rsi'] < 30):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='BUY',
                    price=current['close'],
                    confidence=0.7,
                    strategy=self.name,
                    reason=f"Перепроданность: цена ниже BB ({current['bb_lower']:.2f}), RSI={current['rsi']:.1f}"
                ))
            
            # Перекупленность
            elif (current['close'] > current['bb_upper'] and
                  current['rsi'] > 70):
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='SELL',
                    price=current['close'],
                    confidence=0.7,
                    strategy=self.name,
                    reason=f"Перекупленность: цена выше BB ({current['bb_upper']:.2f}), RSI={current['rsi']:.1f}"
                ))
        
        return signals