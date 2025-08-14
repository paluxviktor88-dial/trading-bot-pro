import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataProvider:
    """Класс для получения рыночных данных из различных источников"""
    
    def __init__(self):
        self.binance_base_url = "https://api.binance.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def test_connection(self) -> bool:
        """Тестирование подключения к Binance API"""
        try:
            response = self.session.get(f"{self.binance_base_url}/api/v3/ping", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Ошибка подключения к Binance: {e}")
            return False
    
    def get_binance_data(self, symbol: str, interval: str = '1h', limit: int = 500) -> pd.DataFrame:
        """
        Получение данных с Binance
        
        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            interval: Таймфрейм ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Количество свечей (максимум 1000)
        """
        try:
            # Очистка символа
            symbol = symbol.replace('/', '').upper()
            
            # Параметры запроса
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance лимит
            }
            
            # Запрос к API
            url = f"{self.binance_base_url}/api/v3/klines"
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"Ошибка API Binance: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not data:
                print(f"Нет данных для {symbol}")
                return pd.DataFrame()
            
            # Создание DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Преобразование типов данных
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Преобразование времени
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Оставляем только нужные колонки
            df = df[numeric_columns]
            
            # Удаляем пустые строки
            df.dropna(inplace=True)
            
            print(f"✅ Загружено {len(df)} свечей для {symbol} ({interval})")
            
            return df
            
        except requests.exceptions.Timeout:
            print(f"❌ Таймаут при получении данных для {symbol}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка сети при получении данных для {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Неожиданная ошибка при получении данных для {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены"""
        try:
            symbol = symbol.replace('/', '').upper()
            
            url = f"{self.binance_base_url}/api/v3/ticker/price"
            params = {'symbol': symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                print(f"Ошибка получения цены для {symbol}: {response.status_code}")
                return 0.0
                
        except Exception as e:
            print(f"Ошибка получения цены для {symbol}: {e}")
            return 0.0
    
    def get_ticker_24h(self, symbol: str) -> Dict:
        """Получение 24-часовой статистики"""
        try:
            symbol = symbol.replace('/', '').upper()
            
            url = f"{self.binance_base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Ошибка получения статистики для {symbol}: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Ошибка получения статистики для {symbol}: {e}")
            return {}
    
    def get_market_data(self, symbol: str, interval: str = '15m', limit: int = 200) -> pd.DataFrame:
        """Универсальный метод получения рыночных данных"""
        return self.get_binance_data(symbol, interval, limit)
    
    def get_multiple_symbols_data(self, symbols: List[str], interval: str = '1h', 
                                 limit: int = 200) -> Dict[str, pd.DataFrame]:
        """Получение данных для нескольких символов"""
        results = {}
        
        for symbol in symbols:
            print(f"📥 Загрузка данных для {symbol}...")
            data = self.get_binance_data(symbol, interval, limit)
            
            if not data.empty:
                results[symbol] = data
                time.sleep(0.1)  # Небольшая задержка между запросами
            else:
                print(f"❌ Не удалось загрузить данные для {symbol}")
        
        return results
    
    def get_btc_dominance(self) -> float:
        """Получение доминации Bitcoin"""
        try:
            # Используем CoinGecko API
            url = "https://api.coingecko.com/api/v3/global"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                dominance = data['data']['market_cap_percentage']['btc']
                return float(dominance)
            else:
                # Fallback значение
                return 42.0
                
        except Exception as e:
            print(f"Ошибка получения доминации BTC: {e}")
            return 42.0  # Средняя историческая доминация
    
    def get_fear_greed_index(self) -> Dict:
        """Получение индекса страха и жадности"""
        try:
            # Alternative API для Fear & Greed индекса
            url = "https://api.alternative.me/fng/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    return {
                        'value': int(data['data'][0]['value']),
                        'classification': data['data'][0]['value_classification'],
                        'timestamp': data['data'][0]['timestamp']
                    }
            
            # Fallback
            return {'value': 50, 'classification': 'Neutral', 'timestamp': str(int(time.time()))}
            
        except Exception as e:
            print(f"Ошибка получения Fear & Greed индекса: {e}")
            return {'value': 50, 'classification': 'Neutral', 'timestamp': str(int(time.time()))}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Проверка валидности торгового символа"""
        try:
            symbol = symbol.replace('/', '').upper()
            
            url = f"{self.binance_base_url}/api/v3/exchangeInfo"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                symbols = [s['symbol'] for s in data['symbols']]
                return symbol in symbols
            
            return False
            
        except Exception as e:
            print(f"Ошибка проверки символа {symbol}: {e}")
            return False
    
    def get_available_symbols(self, quote_asset: str = 'USDT') -> List[str]:
        """Получение списка доступных торговых пар"""
        try:
            url = f"{self.binance_base_url}/api/v3/exchangeInfo"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                symbols = []
                
                for symbol_info in data['symbols']:
                    if (symbol_info['status'] == 'TRADING' and 
                        symbol_info['quoteAsset'] == quote_asset):
                        symbols.append(symbol_info['symbol'])
                
                return sorted(symbols)
            
            return []
            
        except Exception as e:
            print(f"Ошибка получения списка символов: {e}")
            return []
    
    def get_top_volume_symbols(self, quote_asset: str = 'USDT', limit: int = 50) -> List[str]:
        """Получение топ символов по объему торгов"""
        try:
            url = f"{self.binance_base_url}/api/v3/ticker/24hr"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Фильтруем по quote asset и сортируем по объему
                filtered_data = [
                    item for item in data 
                    if item['symbol'].endswith(quote_asset) and float(item['quoteVolume']) > 0
                ]
                
                sorted_data = sorted(filtered_data, key=lambda x: float(x['quoteVolume']), reverse=True)
                
                return [item['symbol'] for item in sorted_data[:limit]]
            
            return []
            
        except Exception as e:
            print(f"Ошибка получения топ символов: {e}")
            return []
    
    def create_sample_data(self, symbol: str = 'BTCUSDT', days: int = 30) -> pd.DataFrame:
        """Создание примерных данных для тестирования (если API недоступен)"""
        print(f"⚠️ Создание тестовых данных для {symbol}")
        
        # Генерируем случайные данные похожие на реальные
        np.random.seed(42)
        
        # Временной диапазон
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start_time, end_time, freq='1H')
        
        # Базовая цена (примерная для BTC)
        base_price = 50000 if 'BTC' in symbol else 2000
        
        # Генерация цен с случайным блужданием
        prices = [base_price]
        for _ in range(len(timestamps) - 1):
            change = np.random.normal(0, 0.02)  # 2% стандартное отклонение
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.5))  # Минимальная цена
        
        # Создание OHLC данных
        df_data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Генерируем high, low, open на основе close
            volatility = abs(np.random.normal(0, 0.01))
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = prices[i-1] if i > 0 else close
            
            # Объем
            volume = np.random.randint(100, 10000)
            
            df_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(df_data, index=timestamps)
        
        print(f"✅ Создано {len(df)} тестовых свечей")
        return df


# Пример использования и тестирования
if __name__ == "__main__":
    print("🔄 Тестирование провайдера данных...")
    
    # Создаем провайдер
    provider = DataProvider()
    
    # Тест подключения
    if provider.test_connection():
        print("✅ Подключение к Binance API успешно")
        
        # Тест получения данных
        print("\n📊 Тестирование получения данных...")
        
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        for symbol in test_symbols:
            print(f"\n🔍 Тестирование {symbol}:")
            
            # Основные данные
            data = provider.get_binance_data(symbol, '1h', 100)
            if not data.empty:
                print(f"  ✅ OHLCV данные: {len(data)} свечей")
                print(f"  📈 Диапазон цен: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
                print(f"  📊 Последняя цена: ${data['close'].iloc[-1]:.2f}")
            
            # Текущая цена
            current_price = provider.get_current_price(symbol)
            if current_price > 0:
                print(f"  💰 Текущая цена: ${current_price:.2f}")
            
            # 24h статистика
            ticker = provider.get_ticker_24h(symbol)
            if ticker:
                change = float(ticker.get('priceChangePercent', 0))
                print(f"  📈 Изменение за 24ч: {change:+.2f}%")
            
            time.sleep(0.5)  # Пауза между запросами
        
        # Тест дополнительных данных
        print(f"\n🌍 Дополнительные данные:")
        
        btc_dominance = provider.get_btc_dominance()
        print(f"  📊 Доминация BTC: {btc_dominance:.1f}%")
        
        fear_greed = provider.get_fear_greed_index()
        print(f"  😨 Fear & Greed: {fear_greed['value']} ({fear_greed['classification']})")
        
        # Топ символы по объему
        top_symbols = provider.get_top_volume_symbols(limit=10)
        if top_symbols:
            print(f"  🔥 Топ-10 по объему: {', '.join(top_symbols[:5])}...")
        
    else:
        print("❌ Не удалось подключиться к Binance API")
        print("🔧 Создание тестовых данных...")
        
        # Создаем тестовые данные
        test_data = provider.create_sample_data('BTCUSDT', 7)
        print(f"✅ Тестовые данные готовы: {len(test_data)} записей")
    
    print("\n✅ Тестирование провайдера данных завершено!")