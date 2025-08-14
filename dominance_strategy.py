import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from strategies import BaseStrategy, TradeSignal
from indicators import TechnicalIndicators

@dataclass
class MarketSentiment:
    """Класс для анализа рыночного сентимента"""
    btc_dominance: float
    fear_greed_index: float
    market_trend: str  # 'BULL', 'BEAR', 'SIDEWAYS'
    volatility_level: str  # 'LOW', 'MEDIUM', 'HIGH'

class DominanceCorrelationStrategy(BaseStrategy):
    """
    🔥 АВТОРСКАЯ СТРАТЕГИЯ: Bitcoin Dominance + Correlation Analysis
    
    Инновационный подход, анализирующий:
    1. Доминацию Bitcoin на рынке
    2. Корреляцию альткоинов с BTC
    3. Рыночный сентимент и циклы
    4. Сезонные паттерны
    """
    
    def __init__(self, btc_dominance_threshold: float = 2.0, 
                 correlation_period: int = 30, risk_level: str = 'MEDIUM'):
        super().__init__("Dominance_Correlation_Strategy")
        self.btc_dominance_threshold = btc_dominance_threshold
        self.correlation_period = correlation_period
        self.risk_level = risk_level  # 'LOW', 'MEDIUM', 'HIGH'
        self.btc_data = None
        
    def get_btc_dominance(self) -> float:
        """Получение текущей доминации Bitcoin"""
        try:
            # Используем CoinGecko API для получения доминации BTC
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                dominance = data['data']['market_cap_percentage']['btc']
                return dominance
            else:
                # Fallback: примерная доминация если API недоступен
                return 42.0
                
        except Exception as e:
            print(f"Ошибка получения доминации BTC: {e}")
            return 42.0  # Средняя историческая доминация
    
    def calculate_correlation_with_btc(self, symbol_data: pd.DataFrame, 
                                     btc_data: pd.DataFrame) -> float:
        """Расчет корреляции альткоина с Bitcoin"""
        try:
            # Берем только пересекающиеся временные периоды
            common_index = symbol_data.index.intersection(btc_data.index)
            
            if len(common_index) < 20:  # Минимум данных для корреляции
                return 0.5  # Нейтральная корреляция
            
            symbol_returns = symbol_data.loc[common_index, 'close'].pct_change().dropna()
            btc_returns = btc_data.loc[common_index, 'close'].pct_change().dropna()
            
            # Берем последние N периодов
            if len(symbol_returns) > self.correlation_period:
                symbol_returns = symbol_returns.tail(self.correlation_period)
                btc_returns = btc_returns.tail(self.correlation_period)
            
            correlation = symbol_returns.corr(btc_returns)
            
            return correlation if not pd.isna(correlation) else 0.5
            
        except Exception as e:
            print(f"Ошибка расчета корреляции: {e}")
            return 0.5
    
    def analyze_market_sentiment(self, data: pd.DataFrame) -> MarketSentiment:
        """Анализ рыночного сентимента"""
        
        # Получаем доминацию BTC
        btc_dominance = self.get_btc_dominance()
        
        # Анализ волатильности
        recent_returns = data['close'].pct_change().tail(14).std()
        
        if recent_returns < 0.02:
            volatility = 'LOW'
        elif recent_returns < 0.05:
            volatility = 'MEDIUM'
        else:
            volatility = 'HIGH'
        
        # Определение тренда по EMA
        ema_short = self.indicators.ema(data['close'], 10).iloc[-1]
        ema_long = self.indicators.ema(data['close'], 30).iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if current_price > ema_short > ema_long:
            trend = 'BULL'
        elif current_price < ema_short < ema_long:
            trend = 'BEAR'
        else:
            trend = 'SIDEWAYS'
        
        # Примерный Fear & Greed индекс (упрощенный)
        rsi = self.indicators.rsi(data['close']).iloc[-1]
        if rsi > 70:
            fear_greed = 75  # Жадность
        elif rsi < 30:
            fear_greed = 25  # Страх
        else:
            fear_greed = 50  # Нейтрально
        
        return MarketSentiment(
            btc_dominance=btc_dominance,
            fear_greed_index=fear_greed,
            market_trend=trend,
            volatility_level=volatility
        )
    
    def get_seasonal_factor(self) -> float:
        """Сезонный фактор (упрощенный анализ)"""
        current_month = datetime.now().month
        
        # Исторически сильные месяцы для крипто
        strong_months = [1, 2, 10, 11, 12]  # Январь, февраль, октябрь-декабрь
        weak_months = [6, 7, 8, 9]  # Летние месяцы
        
        if current_month in strong_months:
            return 1.2  # +20% к сигналу
        elif current_month in weak_months:
            return 0.8  # -20% к сигналу
        else:
            return 1.0  # Нейтрально
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка данных с индикаторами"""
        df = df.copy()
        
        # Базовые индикаторы
        df['rsi'] = self.indicators.rsi(df['close'])
        df['ema_10'] = self.indicators.ema(df['close'], 10)
        df['ema_30'] = self.indicators.ema(df['close'], 30)
        df['ema_50'] = self.indicators.ema(df['close'], 50)
        
        # MACD для подтверждения
        df['macd'], df['macd_signal'], df['macd_hist'] = self.indicators.macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.indicators.bollinger_bands(df['close'])
        
        # ATR для расчета стопов
        df['atr'] = self.indicators.atr(df['high'], df['low'], df['close'])
        
        # Объемный анализ
        df['volume_sma'] = self.indicators.sma(df['volume'], 20)
        df['volume_spike'] = df['volume'] > (df['volume_sma'] * 1.5)
        
        # Momentum
        df['momentum'] = df['close'].pct_change(10)
        
        return df
    
    def get_risk_multiplier(self) -> float:
        """Множитель риска в зависимости от настройки"""
        multipliers = {
            'LOW': 0.5,    # Консервативный подход
            'MEDIUM': 1.0,  # Стандартный
            'HIGH': 1.5    # Агрессивный
        }
        return multipliers.get(self.risk_level, 1.0)
    
    def generate_signals(self, df: pd.DataFrame, symbol: str = 'UNKNOWN') -> List[TradeSignal]:
        """Генерация сигналов на основе доминации и корреляции"""
        
        signals = []
        df = self.prepare_data(df)
        
        # Получаем данные BTC для корреляции (если символ не BTC)
        btc_correlation = 0.5
        if symbol != 'BTCUSDT' and symbol != 'UNKNOWN':
            try:
                from data_provider import DataProvider
                provider = DataProvider()
                btc_data = provider.get_binance_data('BTCUSDT', '1h', len(df))
                if not btc_data.empty:
                    self.btc_data = btc_data
                    btc_correlation = self.calculate_correlation_with_btc(df, btc_data)
            except:
                pass
        
        # Анализ рыночного сентимента
        sentiment = self.analyze_market_sentiment(df)
        
        # Сезонный фактор
        seasonal_factor = self.get_seasonal_factor()
        
        # Множитель риска
        risk_multiplier = self.get_risk_multiplier()
        
        # Основная логика генерации сигналов
        for i in range(50, len(df)):  # Начинаем с 50-й свечи для стабильности индикаторов
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            if pd.isna(current['rsi']) or pd.isna(current['ema_30']):
                continue
            
            # === УСЛОВИЯ ДЛЯ ПОКУПКИ ===
            
            # 1. Базовые технические условия
            rsi_oversold = current['rsi'] < 35
            price_above_ema = current['close'] > current['ema_10']
            macd_bullish = current['macd'] > current['macd_signal']
            volume_confirm = current['volume_spike']
            
            # 2. Условия доминации (для альткоинов)
            dominance_favorable = True
            if symbol != 'BTCUSDT':
                # Для альткоинов: покупаем когда доминация BTC снижается
                dominance_favorable = sentiment.btc_dominance < 45.0
                
                # Учитываем корреляцию
                if btc_correlation > 0.7:  # Высокая корреляция с BTC
                    # Нужен сильный сигнал при высокой корреляции
                    dominance_favorable = dominance_favorable and sentiment.market_trend == 'BULL'
            
            # 3. Сентимент и сезонность
            sentiment_bullish = (sentiment.fear_greed_index < 40 or  # Покупаем на страхе
                               sentiment.market_trend == 'BULL')
            
            seasonal_bullish = seasonal_factor >= 1.0
            
            # 4. Проверка цены относительно Bollinger Bands
            price_near_support = current['close'] <= current['bb_middle']
            
            # === СИГНАЛ НА ПОКУПКУ ===
            if (rsi_oversold and macd_bullish and dominance_favorable and 
                sentiment_bullish and seasonal_bullish and price_near_support and volume_confirm):
                
                # Расчет стоп-лосса и тейк-профита с учетом риска
                atr_multiplier = risk_multiplier * 2.0
                stop_loss = current['close'] - (current['atr'] * atr_multiplier)
                take_profit = current['close'] + (current['atr'] * atr_multiplier * 2)
                
                # Уверенность сигнала
                confidence_factors = [
                    0.2 if dominance_favorable else 0.0,
                    0.2 if volume_confirm else 0.0,
                    0.2 if sentiment_bullish else 0.0,
                    0.2 if seasonal_bullish else 0.0,
                    0.2 if btc_correlation < 0.8 else 0.1  # Лучше низкая корреляция для альткоинов
                ]
                confidence = sum(confidence_factors)
                
                # Причина сигнала
                reason_parts = []
                if dominance_favorable:
                    reason_parts.append(f"BTC dom: {sentiment.btc_dominance:.1f}%")
                if sentiment.market_trend == 'BULL':
                    reason_parts.append("Bull trend")
                if btc_correlation < 0.5:
                    reason_parts.append("Low BTC corr")
                if seasonal_bullish:
                    reason_parts.append("Seasonal+")
                
                reason = f"DOM Strategy: {', '.join(reason_parts)}"
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='BUY',
                    price=current['close'],
                    confidence=min(confidence, 0.95),
                    strategy=self.name,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=reason
                ))
            
            # === УСЛОВИЯ ДЛЯ ПРОДАЖИ ===
            
            rsi_overbought = current['rsi'] > 65
            price_below_ema = current['close'] < current['ema_10']
            macd_bearish = current['macd'] < current['macd_signal']
            
            # Для альткоинов: продаем когда доминация BTC растет
            dominance_bearish = True
            if symbol != 'BTCUSDT':
                dominance_bearish = sentiment.btc_dominance > 50.0
            
            sentiment_bearish = (sentiment.fear_greed_index > 75 or  # Продаем на жадности
                               sentiment.market_trend == 'BEAR')
            
            price_near_resistance = current['close'] >= current['bb_upper']
            
            # === СИГНАЛ НА ПРОДАЖУ ===
            if (rsi_overbought and macd_bearish and dominance_bearish and 
                sentiment_bearish and price_near_resistance and volume_confirm):
                
                atr_multiplier = risk_multiplier * 2.0
                stop_loss = current['close'] + (current['atr'] * atr_multiplier)
                take_profit = current['close'] - (current['atr'] * atr_multiplier * 2)
                
                confidence_factors = [
                    0.2 if dominance_bearish else 0.0,
                    0.2 if volume_confirm else 0.0,
                    0.2 if sentiment_bearish else 0.0,
                    0.2 if price_near_resistance else 0.0,
                    0.2 if btc_correlation > 0.7 else 0.1  # Высокая корреляция плохо для продажи альткоинов
                ]
                confidence = sum(confidence_factors)
                
                reason_parts = []
                if dominance_bearish:
                    reason_parts.append(f"BTC dom: {sentiment.btc_dominance:.1f}%↑")
                if sentiment.market_trend == 'BEAR':
                    reason_parts.append("Bear trend")
                if sentiment.fear_greed_index > 75:
                    reason_parts.append("Greed high")
                
                reason = f"DOM Strategy: {', '.join(reason_parts)}"
                
                signals.append(TradeSignal(
                    timestamp=current.name,
                    action='SELL',
                    price=current['close'],
                    confidence=min(confidence, 0.95),
                    strategy=self.name,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=reason
                ))
        
        return signals
    
    def get_strategy_info(self) -> Dict:
        """Информация о стратегии для интерфейса"""
        return {
            'name': 'Bitcoin Dominance + Correlation',
            'description': 'Инновационная стратегия анализа доминации BTC и корреляций',
            'risk_level': self.risk_level,
            'features': [
                '📊 Анализ доминации Bitcoin',
                '🔗 Корреляция с BTC',
                '📈 Рыночный сентимент',
                '📅 Сезонные факторы',
                f'⚠️ Уровень риска: {self.risk_level}'
            ]
        }


# Пример использования и тестирования
if __name__ == "__main__":
    print("🔥 Тестирование авторской стратегии Bitcoin Dominance...")
    
    # Создаем стратегии с разными уровнями риска
    strategies = [
        DominanceCorrelationStrategy(risk_level='LOW'),
        DominanceCorrelationStrategy(risk_level='MEDIUM'),
        DominanceCorrelationStrategy(risk_level='HIGH')
    ]
    
    # Тестируем на примере данных
    try:
        from data_provider import DataProvider
        
        provider = DataProvider()
        
        # Тестируем на нескольких монетах
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in test_symbols:
            print(f"\n📊 Тестирование {symbol}:")
            data = provider.get_binance_data(symbol, '1h', 200)
            
            if not data.empty:
                for strategy in strategies:
                    signals = strategy.generate_signals(data, symbol)
                    
                    if signals:
                        last_signal = signals[-1]
                        print(f"  {strategy.risk_level} риск: {last_signal.action} "
                              f"(уверенность: {last_signal.confidence:.2%})")
                        print(f"    Причина: {last_signal.reason}")
                    else:
                        print(f"  {strategy.risk_level} риск: Нет сигналов")
            else:
                print(f"  ❌ Не удалось загрузить данные")
    
    except ImportError:
        print("⚠️ Для полного тестирования нужен data_provider.py")
    
    print("\n✅ Авторская стратегия готова к использованию!")